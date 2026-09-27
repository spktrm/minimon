"""Battle mechanics: team generation, damage, move execution, turn resolution.

Every function is pure and traceable. The player index ``p`` may be a traced
integer, so the same code path handles both players (and turn orderings
decided at runtime). Generation-dependent branches are resolved at trace time
from the static ``EnvConfig``; all content (statuses, fields, items, abilities)
is read from the tables in ``EnvParams``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from minimon.config import EnvConfig, Transform
from minimon.constants import (
    B_ATK,
    B_DEF,
    B_SPA,
    B_SPD,
    B_SPE,
    HP,
    MAX_BOOST,
    NONE,
    NUM_BOOSTS,
    SPE,
    Category,
    Effect,
    Phase,
)
from minimon.dex import EnvParams, ModifierTable
from minimon.state import State

# Priority of a status move turned into a guard by the transform.
GUARD_PRIORITY = 4
# Modifier sources: state attribute -> EnvParams table attribute.
_TABLES = {"item": "items", "ability": "abilities"}


def select_tree(pred, on_true, on_false):
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), on_true, on_false)


def boost_multiplier(stage):
    stage = jnp.asarray(stage, jnp.float32)
    return jnp.where(stage >= 0, (2.0 + stage) / 2.0, 2.0 / (2.0 - stage))


def transform_power(tf: Transform, power):
    return jnp.clip(tf.power_base + tf.power_scale * power, tf.power_min, tf.power_max)


def _boost_index(cfg: EnvConfig, b):
    """Without the special split there is one special stat: SpD aliases SpA."""
    if cfg.rules.special_stat_split:
        return b
    return jnp.where(b == B_SPD, B_SPA, b)


def _floor_frac(frac, max_hp):
    """Fraction of max HP, floored, at least 1."""
    return jnp.maximum(1.0, jnp.floor(frac * max_hp))


def _clip_boost(b):
    return jnp.clip(b, -MAX_BOOST, MAX_BOOST)


# ---------------------------------------------------------------------------
# Modifier (item / ability) helpers
# ---------------------------------------------------------------------------


def sources(cfg: EnvConfig) -> list[str]:
    """Modifier sources enabled by the rules."""
    out = []
    if cfg.rules.items:
        out.append("item")
    if cfg.rules.abilities:
        out.append("ability")
    return out


def modifier(params: EnvParams, state: State, source: str, p, i) -> ModifierTable:
    table = getattr(params, _TABLES[source])
    idx = getattr(state, source)[p, i]
    return jax.tree.map(lambda x: x[idx], table)


def _reveal(state: State, source: str, p, i, pred) -> State:
    name = f"revealed_{source}"
    arr = getattr(state, name)
    return state.replace(**{name: arr.at[p, i].set(arr[p, i] | pred)})


def _consume(state: State, source: str, p, i, pred) -> State:
    arr = getattr(state, source)
    return state.replace(**{source: arr.at[p, i].set(jnp.where(pred, NONE, arr[p, i]))})


def stat_multipliers(cfg: EnvConfig, params: EnvParams, state: State, p, i):
    """``f32[5]`` product of status and modifier stat multipliers."""
    m = params.statuses.stat_mult[state.status[p, i]]
    for s in sources(cfg):
        m = m * modifier(params, state, s, p, i).stat_mult
    return m


def is_locked(cfg: EnvConfig, params: EnvParams, state: State, p):
    """Whether player ``p`` is restricted to ``locked_move``."""
    a = state.active[p]
    lock = jnp.bool_(False)
    for s in sources(cfg):
        lock = lock | modifier(params, state, s, p, a).lock
    if cfg.rules.transform.timed:
        lock = lock & (state.transform_turns[p] == 0)
    return lock & (state.locked_move[p] >= 0)


# ---------------------------------------------------------------------------
# Team generation
# ---------------------------------------------------------------------------


def calc_stats(base, level: int):
    """Level-scaled stats from base stats (fixed investment)."""
    core = jnp.floor((2.0 * base + 52.0) * level / 100.0)
    return core.at[..., HP].add(level + 10.0).at[..., 1:].add(5.0)


def sample_team(cfg: EnvConfig, params: EnvParams, key: jax.Array):
    """Sample one player's team: species, moves, items, alternate types."""
    N, M = cfg.team_size, cfg.num_moves
    k_sp, k_mv, k_it, k_alt, k_alt_t = jax.random.split(key, 5)

    g = jax.random.gumbel(k_sp, (params.num_species,))
    _, species = jax.lax.top_k(g, N)

    weights = params.learnset[species]  # [N, Md]
    logits = jnp.log(weights + 1e-6) + jax.random.gumbel(k_mv, weights.shape)
    _, moves = jax.lax.top_k(logits, M)

    if cfg.rules.items:
        item = jax.random.randint(k_it, (N,), 1, cfg.num_items + 1)
    else:
        item = jnp.zeros((N,), jnp.int32)

    types = params.species_types[species]
    random_type = jax.random.randint(k_alt_t, (N,), 0, params.num_types)
    alt = jnp.where(jax.random.uniform(k_alt, (N,)) < 0.5, types[:, 0], random_type)

    stats = calc_stats(params.species_base[species], cfg.level)
    ability = params.species_ability[species] if cfg.rules.abilities else jnp.zeros_like(species)
    return dict(
        species=species.astype(jnp.int32),
        types=types.astype(jnp.int32),
        stats=stats,
        hp=stats[:, HP],
        moves=moves.astype(jnp.int32),
        item=item.astype(jnp.int32),
        ability=ability.astype(jnp.int32),
        alt_type=alt.astype(jnp.int32),
    )


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def defensive_types(cfg: EnvConfig, params: EnvParams, state: State, p, i):
    if not cfg.rules.transform.retype:
        return state.types[p, i]
    retyped = jnp.stack([state.alt_type[p, i], jnp.int32(params.num_types)])
    return jnp.where(state.transformed[p, i], retyped, state.types[p, i])


def effective_speed(cfg: EnvConfig, params: EnvParams, state: State, p):
    a = state.active[p]
    return (state.stats[p, a, SPE] * boost_multiplier(state.boosts[p, B_SPE])
            * stat_multipliers(cfg, params, state, p, a)[B_SPE])


def move_category(cfg: EnvConfig, params: EnvParams, move):
    cat = params.move_category[move]
    if cfg.rules.physical_special_split:
        return cat
    by_type = jnp.where(params.special_types[params.move_type[move]],
                        Category.SPECIAL, Category.PHYSICAL)
    return jnp.where(cat == Category.STATUS, Category.STATUS, by_type)


def stab_multiplier(cfg: EnvConfig, params: EnvParams, state: State, p, i, mtype):
    rules, tf = cfg.rules, cfg.rules.transform
    orig = state.types[p, i]
    in_orig = (orig[0] == mtype) | (orig[1] == mtype)
    stab = jnp.float32(rules.stab_multiplier)
    for s in sources(cfg):
        override = modifier(params, state, s, p, i).stab
        stab = jnp.where(override > 0, jnp.maximum(stab, override), stab)
    normal = jnp.where(in_orig, stab, 1.0)
    if not tf.retype:
        return normal
    alt = state.alt_type[p, i] == mtype
    retyped = jnp.where(alt & in_orig, tf.retype_stab,
                        jnp.where(alt | in_orig, rules.stab_multiplier, 1.0))
    return jnp.where(state.transformed[p, i], retyped, normal)


def damage(cfg: EnvConfig, params: EnvParams, state: State, p, move, power, crit, roll):
    """Damage player ``p``'s active deals to the opposing active with ``move``.

    Returns ``(damage, type_effectiveness)``. ``damage`` is not clipped to the
    defender's remaining HP.
    """
    rules = cfg.rules
    q = 1 - p
    a, d = state.active[p], state.active[q]
    mtype = params.move_type[move]
    special = move_category(cfg, params, move) == Category.SPECIAL

    atk_idx = jnp.where(special, B_SPA, B_ATK)
    def_idx = jnp.where(special, _boost_index(cfg, B_SPD), B_DEF)
    atk_b = state.boosts[p, atk_idx]
    def_b = state.boosts[q, def_idx]
    atk_b = jnp.where(crit, jnp.maximum(atk_b, 0), atk_b)
    def_b = jnp.where(crit, jnp.minimum(def_b, 0), def_b)
    A = (state.stats[p, a, atk_idx + 1] * boost_multiplier(atk_b)
         * stat_multipliers(cfg, params, state, p, a)[atk_idx])
    D = (state.stats[q, d, def_idx + 1] * boost_multiplier(def_b)
         * stat_multipliers(cfg, params, state, q, d)[def_idx])

    mult = jnp.float32(1.0)
    for s in sources(cfg):
        mult = mult * modifier(params, state, s, p, a).damage_mult
    if rules.fields:
        mult = mult * params.fields.type_mult[state.field, mtype]

    dtypes = defensive_types(cfg, params, state, q, d)
    eff = params.type_chart[mtype, dtypes[0]] * params.type_chart[mtype, dtypes[1]]

    base = (2.0 * cfg.level / 5.0 + 2.0) * power * A / D / 50.0 + 2.0
    mult = (mult * jnp.where(crit, rules.crit_multiplier, 1.0) * roll
            * stab_multiplier(cfg, params, state, p, a, mtype) * eff)
    dmg = jnp.maximum(1.0, jnp.floor(base * mult))
    return jnp.where((power > 0) & (eff > 0), dmg, 0.0), eff


def _empowered_move(cfg: EnvConfig, params: EnvParams, state: State, p, move, empowered):
    """(power, accuracy) after the transform's move changes."""
    tf = cfg.rules.transform
    power = params.move_power[move]
    acc = params.move_accuracy[move]
    if not tf.enabled:
        return power, acc
    if tf.changes_power:
        power = jnp.where(empowered & (power > 0), transform_power(tf, power), power)
    if tf.sure_hit:
        acc = jnp.where(empowered, 1.0, acc)
    return power, acc


def _is_empowered(cfg: EnvConfig, state: State, p, burst):
    tf = cfg.rules.transform
    if not tf.enabled:
        return jnp.bool_(False)
    if tf.single_move:
        return jnp.asarray(burst, bool)
    return state.transformed[p, state.active[p]]


def expected_damage(cfg: EnvConfig, params: EnvParams, state: State, p, slot):
    """Accuracy-weighted mean damage of ``p`` using move ``slot`` (no crits)."""
    move = state.moves[p, state.active[p], slot]
    power, acc = _empowered_move(cfg, params, state, p, move, _is_empowered(cfg, state, p, False))
    dmg, _ = damage(cfg, params, state, p, move, power, False, 0.925)
    return dmg * (acc if cfg.chance.misses else 1.0)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def _set_field(state: State, field, turns, pred) -> State:
    return state.replace(
        field=jnp.where(pred, field, state.field).astype(jnp.int32),
        field_turns=jnp.where(pred, turns, state.field_turns).astype(jnp.int32),
    )


def end_transform(cfg: EnvConfig, state: State, p) -> State:
    """Revert a *timed* transform of player ``p``'s active."""
    tf = cfg.rules.transform
    if not tf.timed:
        return state
    a = state.active[p]
    on = state.transformed[p, a] & (state.transform_turns[p] > 0)
    stats = state.stats[p, a]
    reverted = stats.at[1:].divide(jnp.asarray(tf.stat_mult)).at[HP].divide(tf.hp_mult)
    return state.replace(
        hp=state.hp.at[p, a].set(
            jnp.where(on, jnp.ceil(state.hp[p, a] / tf.hp_mult), state.hp[p, a])),
        stats=state.stats.at[p, a].set(jnp.where(on, jnp.round(reverted), stats)),
        transformed=state.transformed.at[p, a].set(state.transformed[p, a] & ~on),
        transform_turns=state.transform_turns.at[p].set(0),
    )


def on_entry(cfg: EnvConfig, params: EnvParams, state: State, p) -> State:
    """Entry effects of player ``p``'s (new) active."""
    q = 1 - p
    a = state.active[p]
    alive = state.hp[p, a] > 0
    foe_alive = state.hp[q, state.active[q]] > 0
    for s in sources(cfg):
        row = modifier(params, state, s, p, a)
        debuff = alive & foe_alive & jnp.any(row.entry_foe_boost != 0)
        state = state.replace(boosts=state.boosts.at[q].set(
            jnp.where(debuff, _clip_boost(state.boosts[q] + row.entry_foe_boost), state.boosts[q])))
        triggered = debuff
        if cfg.rules.fields:
            turns = cfg.rules.entry_field_turns if cfg.rules.entry_field_turns > 0 else -1
            sets = alive & (row.entry_field > 0)
            state = _set_field(state, row.entry_field, turns, sets)
            triggered = triggered | sets
        state = _reveal(state, s, p, a, triggered)
    return state


def do_switch(cfg: EnvConfig, params: EnvParams, state: State, p, slot) -> State:
    old = state.active[p]
    state = end_transform(cfg, state, p)
    for s in sources(cfg):
        row = modifier(params, state, s, p, old)
        heals = (row.switch_heal > 0) & (state.hp[p, old] > 0)
        max_hp = state.stats[p, old, HP]
        healed = jnp.minimum(max_hp, state.hp[p, old] + jnp.floor(max_hp * row.switch_heal))
        state = state.replace(hp=state.hp.at[p, old].set(jnp.where(heals, healed,
                                                                   state.hp[p, old])))
        state = _reveal(state, s, p, old, heals)
    state = state.replace(
        active=state.active.at[p].set(slot),
        boosts=state.boosts.at[p].set(0),
        locked_move=state.locked_move.at[p].set(-1),
        revealed_species=state.revealed_species.at[p, slot].set(True),
    )
    return on_entry(cfg, params, state, p)


def activate_transform(cfg: EnvConfig, state: State, p) -> State:
    tf = cfg.rules.transform
    a = state.active[p]
    state = state.replace(gimmick_used=state.gimmick_used.at[p].set(True))
    if tf.single_move:
        return state
    stats = state.stats[p, a].at[1:].multiply(jnp.asarray(tf.stat_mult))
    stats = stats.at[HP].set(jnp.floor(stats[HP] * tf.hp_mult))
    state = state.replace(
        stats=state.stats.at[p, a].set(stats),
        hp=state.hp.at[p, a].set(jnp.floor(state.hp[p, a] * tf.hp_mult)),
        transformed=state.transformed.at[p, a].set(True),
    )
    if tf.timed:
        state = state.replace(transform_turns=state.transform_turns.at[p].set(tf.duration))
    return state


def move_priority(cfg: EnvConfig, params: EnvParams, state: State, p, slot, burst=False):
    move = state.moves[p, state.active[p], slot]
    prio = params.move_priority[move]
    if cfg.rules.transform.status_guard:
        guard = (_is_empowered(cfg, state, p, burst)
                 & (params.move_category[move] == Category.STATUS))
        prio = jnp.where(guard, GUARD_PRIORITY, prio)
    return prio


def _status_gate(cfg: EnvConfig, params: EnvParams, state: State, p, key):
    """Resolve the user's status before acting. Returns (state, blocked)."""
    chance, table = cfg.chance, params.statuses
    a = state.active[p]
    s, turns = state.status[p, a], state.status_turns[p, a]
    has = s != NONE
    k_skip, k_rec = jax.random.split(key)
    timed = table.max_turns[s] > 0
    if chance.status_rolls:
        recover = has & ~timed & (jax.random.uniform(k_rec) < table.recover[s])
        skip = jax.random.uniform(k_skip) < table.skip[s]
    else:
        recover = jnp.bool_(False)
        skip = table.skip[s] >= 0.5
        timed = timed | skip  # blocking statuses get a fixed duration
    expired = has & timed & (turns <= 0)
    cured = expired | recover
    blocked = has & ~cured & skip
    alive = state.hp[p, a] > 0
    state = state.replace(
        status=state.status.at[p, a].set(jnp.where(alive & cured, NONE, s)),
        status_turns=state.status_turns.at[p, a].set(
            jnp.where(alive & has & timed & ~cured, turns - 1, turns)),
    )
    return state, blocked


def _status_duration(cfg: EnvConfig, params: EnvParams, s, key):
    table = params.statuses
    if cfg.chance.status_rolls:
        dur = jax.random.randint(key, (), table.min_turns[s], table.max_turns[s] + 1)
        return jnp.where(table.max_turns[s] > 0, dur, 0)
    blocking = (table.max_turns[s] > 0) | (table.skip[s] >= 0.5)
    return jnp.where(blocking, cfg.rules.deterministic_status_turns, 0)


def use_move(cfg: EnvConfig, params: EnvParams, state: State, p, slot, burst, key) -> State:
    """Player ``p``'s active uses move ``slot`` against the opposing active.

    ``burst`` marks a move used with a ``single_move`` transform.
    """
    rules, chance, tf = cfg.rules, cfg.chance, cfg.rules.transform
    q = 1 - p
    a, d = state.active[p], state.active[q]
    k = jax.random.split(key, 8)

    # --- Can the user act? ----------------------------------------------------
    alive = state.hp[p, a] > 0
    state, blocked = _status_gate(cfg, params, state, p, k[0])
    can_act = alive & ~blocked

    # --- Move properties ---------------------------------------------------------
    move = state.moves[p, a, slot]
    kind = params.move_effect[move]
    arg = params.move_effect_arg[move]
    mag = params.move_effect_mag[move]
    is_status = params.move_category[move] == Category.STATUS
    empowered = _is_empowered(cfg, state, p, burst)
    power, acc = _empowered_move(cfg, params, state, p, move, empowered)
    guard = (empowered & is_status) if tf.status_guard else jnp.bool_(False)
    secondary_ok = ~empowered if tf.suppress_secondary else jnp.bool_(True)

    targets_foe = ~is_status | (kind == Effect.STATUS) | (kind == Effect.DROP_TARGET)
    target_ok = (state.hp[q, d] > 0) & ~state.protected[q]
    hit = (jax.random.uniform(k[2]) < acc) if chance.misses else jnp.bool_(True)
    lands = can_act & ~guard & jnp.where(targets_foe, hit & target_ok, True)

    # --- Damage --------------------------------------------------------------------
    if chance.crits:
        if rules.crit_speed_based:
            base_spe = params.species_base[state.species[p, a], SPE]
            rate = jnp.minimum(base_spe * rules.crit_speed_scale, 255 / 256)
        else:
            rate = rules.crit_rate
        crit = jax.random.uniform(k[3]) < rate
    else:
        crit = jnp.bool_(False)
    roll = (jax.random.uniform(k[4], minval=0.85, maxval=1.0)
            if chance.damage_rolls else jnp.float32(0.925))
    raw, _ = damage(cfg, params, state, p, move, power, crit, roll)
    dmg = jnp.where(lands & ~is_status, raw, 0.0)

    def_hp, def_max = state.hp[q, d], state.stats[q, d, HP]
    for s in sources(cfg):
        row = modifier(params, state, s, q, d)
        endure = row.endure & (def_hp >= def_max) & (dmg >= def_hp) & (dmg > 0)
        dmg = jnp.where(endure, def_hp - 1, dmg)
        state = _reveal(state, s, q, d, endure)
        state = _consume(state, s, q, d, endure & row.consumable)
    dmg = jnp.minimum(dmg, def_hp)
    new_def_hp = def_hp - dmg
    state = state.replace(hp=state.hp.at[q, d].set(new_def_hp))
    for s in sources(cfg):
        row = modifier(params, state, s, q, d)
        heal = ((row.low_hp_heal > 0) & (state.hp[q, d] > 0)
                & (state.hp[q, d] <= def_max / 2) & (dmg > 0))
        amount = jnp.floor(def_max * row.low_hp_heal)
        state = state.replace(hp=state.hp.at[q, d].set(
            jnp.where(heal, jnp.minimum(def_max, state.hp[q, d] + amount), state.hp[q, d])))
        state = _reveal(state, s, q, d, heal)
        state = _consume(state, s, q, d, heal & row.consumable)

    # --- Effects -------------------------------------------------------------------
    if chance.secondary:
        proc = jax.random.uniform(k[5]) < params.move_effect_chance[move]
    else:
        proc = params.move_effect_chance[move] >= 0.5
    apply = lands & proc & secondary_ok & (is_status | (dmg > 0))
    max_hp = state.stats[p, a, HP]
    bi = _boost_index(cfg, jnp.clip(arg, 0, NUM_BOOSTS - 1))
    stages = mag.astype(jnp.int32)

    # Status on target.
    inflict = (apply & (kind == Effect.STATUS) & (state.status[q, d] == NONE)
               & (state.hp[q, d] > 0))
    state = state.replace(
        status=state.status.at[q, d].set(jnp.where(inflict, arg, state.status[q, d])),
        status_turns=state.status_turns.at[q, d].set(
            jnp.where(inflict, _status_duration(cfg, params, arg, k[6]),
                      state.status_turns[q, d])),
    )
    # Stat stages.
    up = apply & (kind == Effect.BOOST_SELF)
    down = apply & (kind == Effect.DROP_TARGET) & (state.hp[q, d] > 0)
    boosts = state.boosts
    boosts = boosts.at[p, bi].set(jnp.where(up, _clip_boost(boosts[p, bi] + stages), boosts[p, bi]))
    boosts = boosts.at[q, bi].set(jnp.where(down, _clip_boost(boosts[q, bi] - stages),
                                            boosts[q, bi]))
    if tf.status_boost_all:
        all_up = empowered & is_status & can_act
        boosts = boosts.at[p].set(jnp.where(all_up, _clip_boost(boosts[p] + tf.status_boost_all),
                                            boosts[p]))
    state = state.replace(boosts=boosts)
    # HP effects on the user.
    heal = jnp.where(apply & (kind == Effect.HEAL), jnp.floor(mag * max_hp), 0.0)
    heal = heal + jnp.where(apply & (kind == Effect.DRAIN), jnp.floor(mag * dmg), 0.0)
    recoil = jnp.where(apply & (kind == Effect.RECOIL), jnp.maximum(1.0, jnp.floor(mag * dmg)), 0.0)
    for s in sources(cfg):
        row = modifier(params, state, s, p, a)
        hurt = (row.attack_recoil > 0) & (dmg > 0)
        recoil = recoil + jnp.where(hurt, _floor_frac(row.attack_recoil, max_hp), 0.0)
        state = _reveal(state, s, p, a, hurt)
    user_hp = jnp.clip(state.hp[p, a] + heal - recoil, 0.0, max_hp)
    state = state.replace(hp=state.hp.at[p, a].set(jnp.where(alive, user_hp, state.hp[p, a])))
    if rules.fields:
        state = _set_field(state, arg, rules.move_field_turns, apply & (kind == Effect.FIELD))

    # --- Bookkeeping ---------------------------------------------------------------
    lock = jnp.bool_(False)
    for s in sources(cfg):
        lock = lock | modifier(params, state, s, p, a).lock
    if tf.timed:
        lock = lock & (state.transform_turns[p] == 0)
    state = state.replace(locked_move=state.locked_move.at[p].set(
        jnp.where(can_act & lock & (state.locked_move[p] < 0), slot, state.locked_move[p])))
    return state.replace(
        protected=state.protected.at[p].set(state.protected[p] | (guard & can_act)),
        revealed_moves=state.revealed_moves.at[p, a, slot].set(
            state.revealed_moves[p, a, slot] | can_act),
    )


def end_of_turn(cfg: EnvConfig, params: EnvParams, state: State) -> State:
    rules = cfg.rules
    for p in (0, 1):
        a = state.active[p]
        max_hp = state.stats[p, a, HP]
        alive = state.hp[p, a] > 0
        chip_frac = params.statuses.chip[state.status[p, a]]
        if rules.fields:
            dtypes = defensive_types(cfg, params, state, p, a)
            immune = params.fields.chip_immune[state.field, dtypes].any()
            chip_frac = chip_frac + jnp.where(immune, 0.0, params.fields.chip[state.field])
        chip = jnp.where(alive & (chip_frac > 0), _floor_frac(chip_frac, max_hp), 0.0)
        hp = jnp.maximum(0.0, state.hp[p, a] - chip)
        for s in sources(cfg):
            row = modifier(params, state, s, p, a)
            regen = (row.turn_heal > 0) & (hp > 0) & (hp < max_hp)
            hp = jnp.where(regen, jnp.minimum(max_hp, hp + _floor_frac(row.turn_heal, max_hp)), hp)
            grow = jnp.any(row.turn_boost != 0) & (hp > 0)
            state = state.replace(boosts=state.boosts.at[p].set(
                jnp.where(grow, _clip_boost(state.boosts[p] + row.turn_boost), state.boosts[p])))
            state = _reveal(state, s, p, a, regen | grow)
        state = state.replace(hp=state.hp.at[p, a].set(hp))
        if rules.transform.timed:
            left = jnp.maximum(state.transform_turns[p] - 1, 0)
            expiring = state.transform_turns[p] == 1
            state = select_tree(expiring, end_transform(cfg, state, p),
                                state.replace(transform_turns=state.transform_turns.at[p].set(left)))
    if rules.fields:
        ft = state.field_turns
        ft = jnp.where(ft > 0, ft - 1, ft)
        state = state.replace(
            field=jnp.where(ft == 0, NONE, state.field).astype(jnp.int32),
            field_turns=ft,
        )
    return state.replace(turn=state.turn + 1)


def resolve_faints(cfg: EnvConfig, state: State) -> State:
    """Clean up fainted actives and decide phase / termination."""
    for p in (0, 1):
        fainted = state.hp[p, state.active[p]] <= 0
        state = select_tree(fainted, end_transform(cfg, state, p), state)
    idx = jnp.arange(2)
    fainted = state.hp[idx, state.active] <= 0
    remaining = jnp.any(state.hp > 0, axis=1)
    out = ~remaining
    timeout = state.turn >= cfg.max_turns
    done = out[0] | out[1] | timeout
    winner = jnp.where(out[0] & ~out[1], 1, jnp.where(out[1] & ~out[0], 0, -1))
    if cfg.turn_limit == "hp":
        frac = jnp.sum(state.hp / state.stats[..., HP], axis=1)
        by_hp = jnp.where(frac[0] > frac[1], 0, jnp.where(frac[1] > frac[0], 1, -1))
        winner = jnp.where(timeout & ~out[0] & ~out[1], by_hp, winner)
    needs_switch = fainted & remaining & ~done
    return state.replace(
        done=done,
        winner=winner.astype(jnp.int32),
        needs_switch=needs_switch,
        phase=jnp.where(jnp.any(needs_switch), Phase.SWITCH, Phase.TURN).astype(jnp.int32),
    )
