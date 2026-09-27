"""Battle state pytree. Leading axis ``2`` is always the player."""

from __future__ import annotations

import jax

from minimon import struct


@struct.dataclass
class State:
    # --- Teams: [2, N, ...] --------------------------------------------------
    species: jax.Array  # i32[2, N]
    types: jax.Array  # i32[2, N, 2]   original types (T = none)
    stats: jax.Array  # f32[2, N, 6]   current stats; stats[..., HP] is max HP
    hp: jax.Array  # f32[2, N]
    moves: jax.Array  # i32[2, N, M]   indices into the movedex
    item: jax.Array  # i32[2, N]      item-table row (0 = none / consumed)
    ability: jax.Array  # i32[2, N]   ability-table row
    alt_type: jax.Array  # i32[2, N]  hidden alternate type (type transform)
    status: jax.Array  # i32[2, N]    status-table row
    status_turns: jax.Array  # i32[2, N]   remaining turns of a timed status
    transformed: jax.Array  # bool[2, N]
    # --- Field: per player's active slot --------------------------------------
    active: jax.Array  # i32[2]
    boosts: jax.Array  # i32[2, 5]
    locked_move: jax.Array  # i32[2]   lock-item slot, -1 if none
    transform_turns: jax.Array  # i32[2]   remaining turns of a timed transform
    protected: jax.Array  # bool[2]  guarded this turn
    gimmick_used: jax.Array  # bool[2]
    field: jax.Array  # i32[]   field-table row
    field_turns: jax.Array  # i32[]   -1 = permanent
    # --- Information sets -------------------------------------------------------
    revealed_species: jax.Array  # bool[2, N]
    revealed_moves: jax.Array  # bool[2, N, M]
    revealed_item: jax.Array  # bool[2, N]
    revealed_ability: jax.Array  # bool[2, N]
    # --- Control ------------------------------------------------------------------
    phase: jax.Array  # i32[]   Phase.TURN or Phase.SWITCH
    needs_switch: jax.Array  # bool[2]
    turn: jax.Array  # i32[]
    done: jax.Array  # bool[]
    winner: jax.Array  # i32[]   0 / 1, or -1 for draw / undecided
