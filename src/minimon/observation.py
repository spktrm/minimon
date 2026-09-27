"""Egocentric, information-set-respecting observations.

``observe(cfg, params, state, player)`` returns a flat ``f32[D]`` vector laid
out as ``[global | own team (N x F) | opponent team (N x F)]``.

Because content is procedurally generated, every table entry is encoded by its
*properties* (e.g. an item's multipliers and triggers), not only its ID, so
policies can generalise across metagames.

With ``cfg.hidden_info`` the opponent's unrevealed attributes are zeroed and
accompanied by explicit "known" bits:

* species (types / stats)  - revealed on switch-in, or at start with team preview
* each move                - revealed when first used
* item / ability           - revealed when a visible effect triggers
* alternate type           - revealed when the type transform is used
* status turn counters     - never revealed
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from minimon.config import EnvConfig
from minimon.constants import HP, MAX_BOOST, NUM_BOOSTS, Category, Effect, Phase
from minimon.dex import EnvParams, ModifierTable
from minimon.state import State


def _oh(x, n):
    return jax.nn.one_hot(x, n, dtype=jnp.float32)


def _f(x):
    return jnp.asarray(x, jnp.float32)[None]


def _move_features(cfg: EnvConfig, params: EnvParams, move):
    T = params.num_types
    arg_dim = max(cfg.num_statuses + 1, NUM_BOOSTS, cfg.num_fields + 1)
    return jnp.concatenate([
        _oh(params.move_type[move], T),
        _f(params.move_power[move] / 100.0),
        _f(params.move_accuracy[move]),
        _oh(params.move_category[move], Category.NUM),
        _f(params.move_priority[move]),
        _oh(params.move_effect[move], Effect.NUM),
        _oh(params.move_effect_arg[move], arg_dim),
        _f(params.move_effect_mag[move]),
        _f(params.move_effect_chance[move]),
    ])


def _modifier_features(cfg: EnvConfig, table: ModifierTable, idx, rows: int):
    r = jax.tree.map(lambda x: x[idx], table)
    return jnp.concatenate([
        _oh(idx, rows),
        r.stat_mult - 1.0,
        _f(r.damage_mult - 1.0),
        _f(r.stab),
        _f(r.lock),
        _f(r.attack_recoil),
        _f(r.turn_heal),
        r.turn_boost.astype(jnp.float32),
        r.entry_foe_boost.astype(jnp.float32),
        _oh(r.entry_field, cfg.num_fields + 1),
        _f(r.switch_heal),
        _f(r.endure),
        _f(r.low_hp_heal),
        _f(r.consumable),
    ])


def _status_features(cfg: EnvConfig, params: EnvParams, s):
    t = params.statuses
    return jnp.concatenate([
        _oh(s, cfg.num_statuses + 1),
        _f(t.chip[s]),
        t.stat_mult[s] - 1.0,
        _f(t.skip[s]),
        _f(t.max_turns[s] > 0),
        _f(t.recover[s]),
    ])


def _mon_features(cfg: EnvConfig, params: EnvParams, state: State, p, i, own: bool):
    T = params.num_types
    hidden = cfg.hidden_info and not own

    if hidden:
        k_species = state.revealed_species[p, i]
        k_item = state.revealed_item[p, i]
        k_ability = state.revealed_ability[p, i]
        k_alt = state.transformed[p, i] & cfg.rules.transform.retype
        k_moves = state.revealed_moves[p, i]
    else:
        k_species = k_item = k_ability = k_alt = jnp.bool_(True)
        k_moves = jnp.ones((cfg.num_moves,), bool)

    types = state.types[p, i]
    type_mh = (_oh(types[0], T + 1) + _oh(types[1], T + 1))[:T]
    moves = jax.vmap(lambda m: _move_features(cfg, params, m))(state.moves[p, i])
    moves = jnp.concatenate([k_moves[:, None].astype(jnp.float32), moves * k_moves[:, None]], 1)
    status_turns = 0.0 if hidden else state.status_turns[p, i] / 8.0
    item = _modifier_features(cfg, params.items, state.item[p, i], cfg.num_items + 1)
    ability = _modifier_features(cfg, params.abilities, state.ability[p, i],
                                 cfg.num_abilities + 1)

    return jnp.concatenate([
        _f(k_species),
        type_mh * k_species,
        state.stats[p, i] / 200.0 * k_species,
        _f(state.hp[p, i] / state.stats[p, i, HP]),
        _f(state.hp[p, i] > 0),
        _f(state.active[p] == i),
        _status_features(cfg, params, state.status[p, i]),
        _f(status_turns),
        _f(k_item),
        item * k_item,
        _f(k_ability),
        ability * k_ability,
        _f(k_alt),
        _oh(state.alt_type[p, i], T) * k_alt,
        _f(state.transformed[p, i]),
        moves.reshape(-1),
    ])


def _team_features(cfg, params, state, p, own):
    idx = jnp.arange(cfg.team_size)
    return jax.vmap(lambda i: _mon_features(cfg, params, state, p, i, own))(idx).reshape(-1)


def _global_features(cfg: EnvConfig, params: EnvParams, state: State, p):
    q = 1 - p
    ft = jnp.where(state.field_turns < 0, 1.0, state.field_turns / 8.0)
    duration = max(cfg.rules.transform.duration, 1)
    return jnp.concatenate([
        _oh(state.field, cfg.num_fields + 1),
        params.fields.type_mult[state.field] - 1.0,
        _f(params.fields.chip[state.field]),
        _f(ft),
        _f(state.turn / cfg.max_turns),
        _oh(state.phase, Phase.NUM),
        _f(state.needs_switch[p]),
        _f(state.needs_switch[q]),
        _f(state.gimmick_used[p]),
        _f(state.gimmick_used[q]),
        _f(state.transform_turns[p] / duration),
        _f(state.transform_turns[q] / duration),
        state.boosts[p] / MAX_BOOST,
        state.boosts[q] / MAX_BOOST,
        _oh(state.locked_move[p], cfg.num_moves),
    ])


def observe(cfg: EnvConfig, params: EnvParams, state: State, player) -> jax.Array:
    """Flat observation for ``player`` (may be traced)."""
    return jnp.concatenate([
        _global_features(cfg, params, state, player),
        _team_features(cfg, params, state, player, own=True),
        _team_features(cfg, params, state, 1 - player, own=False),
    ])
