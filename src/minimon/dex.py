"""Procedurally generated content (``EnvParams``).

``EnvParams`` holds every array the dynamics read that is *not* per-episode
state: the type chart, species, moves, and the status / field / item / ability
tables. Nothing is copied from the real games; each system is a generic,
parameterised analogue sampled from a key. Because it is a pytree, a batch of
different metagames can be built with ``jax.vmap(make_params)`` and passed to a
vmapped ``step`` for domain randomisation.

Row ``0`` of every table is "none" (neutral). Users may also construct any of
these dataclasses directly from their own arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from minimon import struct, typechart
from minimon.config import EnvConfig, Rules
from minimon.constants import (
    B_ATK,
    B_DEF,
    B_SPA,
    B_SPD,
    B_SPE,
    NUM_BOOSTS,
    SPA,
    SPD,
    Category,
    Effect,
)


@struct.dataclass
class StatusTable:
    """Non-volatile statuses. Shapes ``[S+1, ...]``."""

    chip: jax.Array  # f32  end-of-turn damage (fraction of max HP)
    stat_mult: jax.Array  # f32[., 5]  multiplier on ATK..SPE
    skip: jax.Array  # f32  chance to lose the turn
    min_turns: jax.Array  # i32  duration bounds (max 0 = until cured)
    max_turns: jax.Array  # i32
    recover: jax.Array  # f32  per-turn recovery chance (untimed statuses)


@struct.dataclass
class FieldTable:
    """Field conditions (weather / terrain analogues). Shapes ``[F+1, ...]``."""

    type_mult: jax.Array  # f32[., T+1]  damage multiplier by move type
    chip: jax.Array  # f32  end-of-turn damage to non-immune mons
    chip_immune: jax.Array  # bool[., T+1]  types immune to the chip


@struct.dataclass
class ModifierTable:
    """Held items / abilities: each row is a bundle of passive and triggered effects.

    Shapes ``[K+1, ...]``. A row is revealed to the opponent when one of its
    visible effects triggers (heals, recoil, entry effects, boosts, endure).
    """

    stat_mult: jax.Array  # f32[., 5]  passive stat multipliers
    damage_mult: jax.Array  # f32  passive multiplier on damage dealt
    stab: jax.Array  # f32  same-type bonus override (0 = default)
    lock: jax.Array  # bool  locked into the first move used until switching
    attack_recoil: jax.Array  # f32  max-HP fraction lost after dealing damage
    turn_heal: jax.Array  # f32  end-of-turn heal fraction
    turn_boost: jax.Array  # i32[., 5]  end-of-turn self stat stages
    entry_foe_boost: jax.Array  # i32[., 5]  stat stages applied to the foe on entry
    entry_field: jax.Array  # i32  field row set on entry (0 = none)
    switch_heal: jax.Array  # f32  heal fraction on switching out
    endure: jax.Array  # bool  survive a KO from full HP with 1 HP
    low_hp_heal: jax.Array  # f32  heal when dropping to <= half HP
    consumable: jax.Array  # bool  removed after endure / low-HP heal triggers


@struct.dataclass
class EnvParams:
    # Types (index ``T`` = "no type", neutral in the padded chart).
    type_chart: jax.Array  # f32[T+1, T+1]  [attacking, defending]
    special_types: jax.Array  # bool[T+1]  category-by-type without the phys/spec split
    # Species.
    species_types: jax.Array  # i32[S, 2]
    species_base: jax.Array  # f32[S, 6]  base stats
    species_ability: jax.Array  # i32[S]
    learnset: jax.Array  # f32[S, Md]  sampling weight, 0 = cannot learn
    # Moves.
    move_type: jax.Array  # i32[Md]
    move_power: jax.Array  # f32[Md]
    move_accuracy: jax.Array  # f32[Md]  in (0, 1]
    move_category: jax.Array  # i32[Md]
    move_priority: jax.Array  # i32[Md]
    move_effect: jax.Array  # i32[Md]
    move_effect_arg: jax.Array  # i32[Md]
    move_effect_mag: jax.Array  # f32[Md]
    move_effect_chance: jax.Array  # f32[Md]
    # Tables.
    statuses: StatusTable
    fields: FieldTable
    items: ModifierTable
    abilities: ModifierTable

    @property
    def num_types(self) -> int:
        return self.type_chart.shape[-1] - 1

    @property
    def num_species(self) -> int:
        return self.species_base.shape[-2]

    @property
    def num_dex_moves(self) -> int:
        return self.move_power.shape[-1]


def _pick(key, values, probs, shape):
    return jax.random.choice(key, jnp.asarray(values), shape=shape, p=jnp.asarray(probs))


def _stat_hot(key, rows, choices):
    """``[rows, 5]`` one-hot of a stat drawn from ``choices`` per row."""
    idx = jax.random.choice(key, jnp.asarray(choices), (rows,))
    return jax.nn.one_hot(idx, NUM_BOOSTS) > 0


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

NUM_STATUS_ARCHETYPES = 5  # weaken, chip, slow, dormant, stun


def make_statuses(key: jax.Array, rules: Rules, num: int) -> StatusTable:
    """Rows 1-5 follow the archetypes with ``rules`` magnitudes; further rows
    cycle through them with randomised magnitudes."""
    rows = num + 1
    r = jnp.arange(rows)
    arch = (r - 1) % NUM_STATUS_ARCHETYPES
    live = r > 0
    k_jit, k_stat = jax.random.split(key)
    jitter = jnp.where(r > NUM_STATUS_ARCHETYPES,
                       jax.random.uniform(k_jit, (rows,), minval=0.5, maxval=1.5), 1.0)

    weaken, chip_a, slow, dormant, stun = (live & (arch == i) for i in range(5))
    chip = jnp.where(weaken, rules.weaken_chip, 0.0) + jnp.where(chip_a, rules.chip_damage, 0.0)
    extra_stat = jnp.where(r > NUM_STATUS_ARCHETYPES,
                           _stat_hot(k_stat, rows, [B_ATK, B_SPA]).argmax(-1), B_ATK)
    stat_mult = jnp.ones((rows, NUM_BOOSTS))
    stat_mult = jnp.where(weaken[:, None] & (jnp.arange(NUM_BOOSTS) == extra_stat[:, None]),
                          rules.weaken_stat_mult, stat_mult)
    stat_mult = stat_mult.at[:, B_SPE].set(jnp.where(slow, rules.slow_speed_mult, 1.0))
    skip = jnp.where(slow, jnp.clip(rules.slow_skip_chance * jitter, 0.0, 1.0), 0.0)
    skip = jnp.where(dormant | stun, 1.0, skip)
    return StatusTable(
        chip=jnp.clip(chip * jitter, 0.0, 0.5).astype(jnp.float32),
        stat_mult=stat_mult.astype(jnp.float32),
        skip=skip.astype(jnp.float32),
        min_turns=jnp.where(dormant, rules.dormant_min, 0).astype(jnp.int32),
        max_turns=jnp.where(dormant, rules.dormant_max, 0).astype(jnp.int32),
        recover=jnp.where(stun, jnp.clip(rules.stun_recover_chance * jitter, 0.0, 1.0),
                          0.0).astype(jnp.float32),
    )


# ---------------------------------------------------------------------------
# Field table
# ---------------------------------------------------------------------------


def make_fields(key: jax.Array, rules: Rules, num: int, num_types: int) -> FieldTable:
    """Each field boosts one type and weakens another. Fields 1 and 2 are a
    mirrored pair (sun / rain analogue); later ones may also chip non-immune mons."""
    rows = num + 1
    T = num_types
    k_b, k_n, k_chip, k_imm = jax.random.split(key, 4)
    boost = jax.random.randint(k_b, (rows,), 0, T)
    nerf = jax.random.randint(k_n, (rows,), 0, T - 1)
    nerf = jnp.where(nerf >= boost, nerf + 1, nerf)
    if rows > 2:
        boost, nerf = boost.at[2].set(nerf[1]), nerf.at[2].set(boost[1])
    live = (jnp.arange(rows) > 0)[:, None]
    cols = jnp.arange(T + 1)[None, :]
    m = rules.field_multiplier
    type_mult = jnp.where(live & (cols == boost[:, None]), m, 1.0)
    type_mult = jnp.where(live & (cols == nerf[:, None]), 1.0 / m, type_mult)
    chips = (jnp.arange(rows) > 2) & (jax.random.uniform(k_chip, (rows,)) < 0.5)
    immune = jax.random.uniform(k_imm, (rows, T + 1)) < 2.0 / T
    immune = immune | (cols == boost[:, None]) | (cols == T)
    return FieldTable(
        type_mult=type_mult.astype(jnp.float32),
        chip=jnp.where(chips, rules.field_chip, 0.0).astype(jnp.float32),
        chip_immune=immune,
    )


# ---------------------------------------------------------------------------
# Item / ability tables
# ---------------------------------------------------------------------------


def _neutral_modifiers(rows: int) -> dict[str, jax.Array]:
    zf, zb = jnp.zeros((rows,)), jnp.zeros((rows,), bool)
    zi5 = jnp.zeros((rows, NUM_BOOSTS), jnp.int32)
    return dict(
        stat_mult=jnp.ones((rows, NUM_BOOSTS)), damage_mult=jnp.ones((rows,)), stab=zf,
        lock=zb, attack_recoil=zf, turn_heal=zf, turn_boost=zi5, entry_foe_boost=zi5,
        entry_field=jnp.zeros((rows,), jnp.int32), switch_heal=zf, endure=zb,
        low_hp_heal=zf, consumable=zb,
    )


def _u(key, rows, lo, hi):
    return jax.random.uniform(key, (rows,), minval=lo, maxval=hi)


# Each archetype maps (key, rows, num_fields) -> field overrides.
ITEM_ARCHETYPES = {
    "regen": lambda k, n, F: dict(turn_heal=_u(k, n, 1 / 16, 1 / 8)),
    "lock": lambda k, n, F: dict(
        lock=jnp.ones((n,), bool),
        stat_mult=jnp.where(_stat_hot(k, n, [B_ATK, B_SPA, B_SPE]), 1.5, 1.0)),
    "power_recoil": lambda k, n, F: dict(damage_mult=_u(k, n, 1.2, 1.3),
                                         attack_recoil=jnp.full((n,), 0.1)),
    "endure": lambda k, n, F: dict(endure=jnp.ones((n,), bool), consumable=jnp.ones((n,), bool)),
    "low_hp_heal": lambda k, n, F: dict(low_hp_heal=_u(k, n, 0.25, 1 / 3),
                                        consumable=jnp.ones((n,), bool)),
    "bulk": lambda k, n, F: dict(stat_mult=jnp.where(_stat_hot(k, n, [B_DEF, B_SPD]), 1.5, 1.0)),
    "power": lambda k, n, F: dict(damage_mult=_u(k, n, 1.1, 1.2)),
}

ABILITY_ARCHETYPES = {
    "entry_debuff": lambda k, n, F: dict(
        entry_foe_boost=-_stat_hot(k, n, list(range(NUM_BOOSTS))).astype(jnp.int32)),
    "entry_field": lambda k, n, F: dict(entry_field=jax.random.randint(k, (n,), 1, F + 1)),
    "turn_boost": lambda k, n, F: dict(
        turn_boost=_stat_hot(k, n, list(range(NUM_BOOSTS))).astype(jnp.int32)),
    "stab_up": lambda k, n, F: dict(stab=jnp.full((n,), 2.0)),
    "switch_heal": lambda k, n, F: dict(switch_heal=_u(k, n, 0.25, 1 / 3)),
    "stat_up": lambda k, n, F: dict(
        stat_mult=jnp.where(_stat_hot(k, n, list(range(NUM_BOOSTS))), 1.3, 1.0)),
    "regen": lambda k, n, F: dict(turn_heal=jnp.full((n,), 1 / 16)),
}


def make_modifiers(key: jax.Array, num: int, num_fields: int, archetypes: dict) -> ModifierTable:
    """Rows 1..len(archetypes) cover each archetype once; later rows sample one."""
    rows = num + 1
    builders = list(archetypes.values())
    keys = jax.random.split(key, len(builders) + 1)
    r = jnp.arange(rows)
    arch = jnp.where(r - 1 < len(builders), r - 1,
                     jax.random.randint(keys[0], (rows,), 0, len(builders)))
    out = _neutral_modifiers(rows)
    for j, build in enumerate(builders):
        sel = (arch == j) & (r > 0)
        for name, val in build(keys[j + 1], rows, num_fields).items():
            mask = sel.reshape(sel.shape + (1,) * (val.ndim - 1))
            out[name] = jnp.where(mask, val, out[name])
    ints = {"turn_boost", "entry_foe_boost", "entry_field"}
    bools = {"lock", "endure", "consumable"}
    return ModifierTable(**{
        k: v.astype(jnp.int32 if k in ints else bool if k in bools else jnp.float32)
        for k, v in out.items()
    })


# ---------------------------------------------------------------------------
# Moves and species
# ---------------------------------------------------------------------------


def _make_moves(key: jax.Array, n: int, num_types: int, num_statuses: int,
                num_fields: int) -> dict[str, jax.Array]:
    ks = iter(jax.random.split(key, 24))
    u = lambda: jax.random.uniform(next(ks), (n,))  # noqa: E731

    mtype = jax.random.randint(next(ks), (n,), 0, num_types)
    is_status = u() < 0.2

    # --- damaging moves ----------------------------------------------------
    power = _pick(next(ks), [40, 50, 60, 70, 80, 90, 100, 110, 120, 150],
                  [.08, .08, .12, .12, .16, .16, .1, .08, .07, .03], (n,)).astype(jnp.float32)
    acc_strong = _pick(next(ks), [1.0, 0.95, 0.9, 0.85, 0.8, 0.7],
                       [.4, .15, .15, .1, .1, .1], (n,))
    acc = jnp.where(power <= 80, jnp.where(u() < 0.85, 1.0, 0.9), acc_strong)
    cat = jax.random.randint(next(ks), (n,), 0, 2)
    priority = jnp.where((power <= 40) & (u() < 0.5), 1, 0)

    sec = _pick(next(ks), [Effect.NONE, Effect.STATUS, Effect.DROP_TARGET, Effect.BOOST_SELF,
                           Effect.DRAIN, Effect.RECOIL],
                [.55, .15, .12, .05, .06, .07], (n,))
    status_arg = jax.random.randint(next(ks), (n,), 1, num_statuses + 1)
    boost_arg = jax.random.randint(next(ks), (n,), 0, NUM_BOOSTS)
    sec_chance = _pick(next(ks), [0.1, 0.2, 0.3], [.5, .3, .2], (n,))
    dmg_arg = jnp.where(sec == Effect.STATUS, status_arg,
                        jnp.where((sec == Effect.DROP_TARGET) | (sec == Effect.BOOST_SELF),
                                  boost_arg, 0))
    dmg_mag = jnp.select(
        [sec == Effect.DRAIN, sec == Effect.RECOIL,
         (sec == Effect.DROP_TARGET) | (sec == Effect.BOOST_SELF)],
        [jnp.full((n,), 0.5), jnp.full((n,), 1 / 3), jnp.ones((n,))],
        jnp.zeros((n,)),
    )
    dmg_chance = jnp.where((sec == Effect.DRAIN) | (sec == Effect.RECOIL), 1.0,
                           jnp.where(sec == Effect.NONE, 0.0, sec_chance))
    power = jnp.where(sec == Effect.RECOIL, jnp.maximum(power, 120.0), power)
    power = jnp.where(sec == Effect.DRAIN, jnp.minimum(power, 75.0), power)

    # --- status moves ------------------------------------------------------
    kind = _pick(next(ks), [Effect.STATUS, Effect.BOOST_SELF, Effect.DROP_TARGET, Effect.HEAL,
                            Effect.FIELD],
                 [.35, .3, .15, .1, .1], (n,))
    st_status = jax.random.randint(next(ks), (n,), 1, num_statuses + 1)
    st_boost = jax.random.randint(next(ks), (n,), 0, NUM_BOOSTS)
    st_field = jax.random.randint(next(ks), (n,), 1, num_fields + 1)
    st_arg = jnp.select(
        [kind == Effect.STATUS, kind == Effect.FIELD],
        [st_status, st_field],
        jnp.where(kind == Effect.HEAL, 0, st_boost),
    )
    st_mag = jnp.where(kind == Effect.HEAL, 0.5, 2.0)
    st_acc = jnp.where(kind == Effect.STATUS, _pick(next(ks), [0.75, 0.85, 0.9, 1.0],
                                                    [.25, .25, .3, .2], (n,)), 1.0)

    return dict(
        move_type=mtype.astype(jnp.int32),
        move_power=jnp.where(is_status, 0.0, power).astype(jnp.float32),
        move_accuracy=jnp.where(is_status, st_acc, acc).astype(jnp.float32),
        move_category=jnp.where(is_status, Category.STATUS, cat).astype(jnp.int32),
        move_priority=jnp.where(is_status, 0, priority).astype(jnp.int32),
        move_effect=jnp.where(is_status, kind, sec).astype(jnp.int32),
        move_effect_arg=jnp.where(is_status, st_arg, dmg_arg).astype(jnp.int32),
        move_effect_mag=jnp.where(is_status, st_mag, dmg_mag).astype(jnp.float32),
        move_effect_chance=jnp.where(is_status, 1.0, dmg_chance).astype(jnp.float32),
    )


def _make_species(key, config: EnvConfig, num_types: int, move_type, move_power):
    S = config.num_species
    ks = jax.random.split(key, 8)
    t1 = jax.random.randint(ks[0], (S,), 0, num_types)
    t2 = jax.random.randint(ks[1], (S,), 0, num_types - 1)
    t2 = jnp.where(t2 >= t1, t2 + 1, t2)  # distinct from t1
    t2 = jnp.where(jax.random.uniform(ks[2], (S,)) < 0.5, t2, num_types)
    types = jnp.stack([t1, t2], axis=-1).astype(jnp.int32)

    bst = jax.random.uniform(ks[3], (S, 1), minval=330.0, maxval=600.0)
    w = jax.random.dirichlet(ks[4], jnp.full((6,), 6.0), shape=(S,))
    base = jnp.clip(bst * w, 20.0, 200.0)
    if not config.rules.special_stat_split:
        base = base.at[:, SPD].set(base[:, SPA])

    ability = jax.random.randint(ks[5], (S,), 1, config.num_abilities + 1)

    Md = move_type.shape[0]
    p_learn = min(1.0, max(0.15, 3.0 * config.num_moves / Md))
    learnable = jax.random.uniform(ks[6], (S, Md)) < p_learn
    stab = (move_type[None, :] == t1[:, None]) | (move_type[None, :] == t2[:, None])
    weight = learnable * (1.0 + 2.0 * stab) * jnp.where(move_power[None, :] > 0, 1.5, 1.0)
    return dict(
        species_types=types,
        species_base=base.astype(jnp.float32),
        species_ability=ability.astype(jnp.int32),
        learnset=weight.astype(jnp.float32),
    )


def make_params(config: EnvConfig, key: jax.Array) -> EnvParams:
    """Generate a metagame (type chart, dex and mechanic tables) from ``key``.

    Traceable, so ``jax.vmap(lambda k: make_params(config, k))(keys)`` builds a
    batch of independent metagames.
    """
    ks = jax.random.split(key, 8)
    T = config.num_types
    if config.type_chart == "generational":
        chart = typechart.generational_chart(ks[0], config.rules.gen, config.base_types)
    else:
        chart = typechart.random_chart(ks[0], T)
    special = jax.random.bernoulli(ks[1], 0.5, (T,))

    moves = _make_moves(ks[2], config.num_dex_moves, T, config.num_statuses, config.num_fields)
    species = _make_species(ks[3], config, T, moves["move_type"], moves["move_power"])
    return EnvParams(
        type_chart=typechart.pad_chart(chart),
        special_types=jnp.pad(special, (0, 1), constant_values=False),
        **species,
        **moves,
        statuses=make_statuses(ks[4], config.rules, config.num_statuses),
        fields=make_fields(ks[5], config.rules, config.num_fields, T),
        items=make_modifiers(ks[6], config.num_items, config.num_fields, ITEM_ARCHETYPES),
        abilities=make_modifiers(ks[7], config.num_abilities, config.num_fields,
                                 ABILITY_ARCHETYPES),
    )
