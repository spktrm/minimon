"""The MiniMon environment: a functional, jittable two-player zero-sum game.

Each call to ``step`` is one simultaneous-move stage game: both players submit
an action, chance is resolved from ``key``, and the state advances. Episodes
therefore form a *sequential matrix game* (a stochastic game with imperfect
information).

Action layout (``A = 2M + N + 1``)::

    [0, M)          use move i
    [M, M+N)        switch to team slot j
    [M+N, 2M+N)     activate the once-per-battle transform, then use move i
    2M+N            no-op (only legal when the player has nothing to decide)

Illegal actions are replaced by the lowest-index legal action.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from minimon import mechanics as mech
from minimon.config import EnvConfig
from minimon.constants import HP, NUM_BOOSTS, Phase
from minimon.dex import EnvParams, make_params
from minimon.observation import observe
from minimon.state import State


class StepOutput(NamedTuple):
    state: State
    obs: jax.Array  # f32[2, D]
    reward: jax.Array  # f32[2], zero-sum
    done: jax.Array  # bool[]
    info: dict[str, Any]


class MiniMon:
    """Functional environment. Methods are pure and safe to ``jit``/``vmap``.

    >>> env = MiniMon(make_config("medium", gen=9))
    >>> params = env.default_params()
    >>> state, obs = env.reset(key, params)
    >>> out = env.step(key, state, jnp.array([0, 0]), params)
    """

    num_players = 2

    def __init__(self, config: EnvConfig | None = None):
        self.config = config if config is not None else EnvConfig()

    def __repr__(self) -> str:
        return f"MiniMon({self.config!r})"

    # --- Sizes -------------------------------------------------------------------

    @property
    def num_actions(self) -> int:
        return self.config.num_actions

    @property
    def noop_action(self) -> int:
        return self.num_actions - 1

    @property
    def obs_size(self) -> int:
        def obs_shape(key):
            return self.reset(key, self.default_params(key))[1]

        return jax.eval_shape(obs_shape, jax.random.key(0)).shape[-1]

    # --- Core API --------------------------------------------------------------------

    def default_params(self, key: jax.Array | None = None) -> EnvParams:
        """Generate a metagame. The same ``key`` always yields the same content."""
        return make_params(self.config, jax.random.key(0) if key is None else key)

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[State, jax.Array]:
        cfg = self.config
        N, M = cfg.team_size, cfg.num_moves
        k_teams, k_order = jax.random.split(key)
        team = jax.vmap(lambda k: mech.sample_team(cfg, params, k))(jax.random.split(k_teams))

        z_i = lambda *s: jnp.zeros(s, jnp.int32)  # noqa: E731
        z_b = lambda *s: jnp.zeros(s, bool)  # noqa: E731
        first_seen = jnp.zeros((2, N), bool).at[:, 0].set(True)
        preview = cfg.rules.team_preview or not cfg.hidden_info
        state = State(
            **team,
            status=z_i(2, N),
            status_turns=z_i(2, N),
            transformed=z_b(2, N),
            active=z_i(2),
            boosts=z_i(2, NUM_BOOSTS),
            locked_move=jnp.full((2,), -1, jnp.int32),
            transform_turns=z_i(2),
            protected=z_b(2),
            gimmick_used=z_b(2),
            field=jnp.int32(0),
            field_turns=jnp.int32(0),
            revealed_species=jnp.ones((2, N), bool) if preview else first_seen,
            revealed_moves=z_b(2, N, M),
            revealed_item=z_b(2, N),
            revealed_ability=z_b(2, N),
            phase=jnp.int32(Phase.TURN),
            needs_switch=z_b(2),
            turn=jnp.int32(0),
            done=jnp.bool_(False),
            winner=jnp.int32(-1),
        )
        # Lead entry effects, faster lead first.
        first = self._faster(state, params, k_order)
        state = mech.on_entry(cfg, params, state, first)
        state = mech.on_entry(cfg, params, state, 1 - first)
        return state, self.observe(state, params)

    def step(self, key: jax.Array, state: State, actions: jax.Array,
             params: EnvParams) -> StepOutput:
        legal = self.legal_actions(state, params)
        actions = jnp.asarray(actions, jnp.int32)
        actions = jnp.where(legal[jnp.arange(2), actions], actions, jnp.argmax(legal, axis=-1))

        advanced = jax.lax.cond(
            state.phase == Phase.TURN,
            lambda s: self._turn(key, s, actions, params),
            lambda s: self._forced_switch(key, s, actions, params),
            state,
        )
        next_state = mech.select_tree(state.done, state, advanced)
        just_done = next_state.done & ~state.done
        r0 = jnp.where(next_state.winner == 0, 1.0, jnp.where(next_state.winner == 1, -1.0, 0.0))
        reward = jnp.where(just_done, jnp.stack([r0, -r0]), 0.0)
        info = {
            "legal_actions": self.legal_actions(next_state, params),
            "winner": next_state.winner,
            "turn": next_state.turn,
            "actions": actions,
        }
        return StepOutput(next_state, self.observe(next_state, params), reward,
                          next_state.done, info)

    def legal_actions(self, state: State, params: EnvParams) -> jax.Array:
        """``bool[2, A]`` mask of legal actions per player."""
        cfg = self.config
        N, M = cfg.team_size, cfg.num_moves

        def per_player(p):
            a = state.active[p]
            bench = (state.hp[p] > 0) & (jnp.arange(N) != a)
            locked = mech.is_locked(cfg, params, state, p)
            move_ok = jnp.where(locked, jnp.arange(M) == state.locked_move[p], True)
            if cfg.rules.transform.enabled:
                gmk_ok = move_ok & ~state.gimmick_used[p]
            else:
                gmk_ok = jnp.zeros((M,), bool)
            f_m, f_n = jnp.zeros((M,), bool), jnp.zeros((N,), bool)
            turn_mask = jnp.concatenate([move_ok, bench, gmk_ok, jnp.array([False])])
            switch_mask = jnp.concatenate([f_m, bench, f_m, jnp.array([False])])
            noop_mask = jnp.concatenate([f_m, f_n, f_m, jnp.array([True])])
            mask = jnp.where(state.phase == Phase.TURN, turn_mask,
                             jnp.where(state.needs_switch[p], switch_mask, noop_mask))
            return jnp.where(state.done, noop_mask, mask)

        return jax.vmap(per_player)(jnp.arange(2))

    def observe(self, state: State, params: EnvParams) -> jax.Array:
        """``f32[2, D]``: row ``p`` is player ``p``'s egocentric observation."""
        return jax.vmap(lambda p: observe(self.config, params, state, p))(jnp.arange(2))

    # --- Internals ---------------------------------------------------------------------

    def _faster(self, state: State, params: EnvParams, key: jax.Array, prio=None):
        """Index of the player who acts first (priority, then speed, then tie-break)."""
        spe = jnp.stack([mech.effective_speed(self.config, params, state, p) for p in (0, 1)])
        if self.config.chance.speed_ties:
            tie = jax.random.bernoulli(key).astype(jnp.int32)
        else:
            tie = state.turn % 2
        by_speed = jnp.where(spe[0] > spe[1], 0, jnp.where(spe[1] > spe[0], 1, tie))
        if prio is None:
            return by_speed
        return jnp.where(prio[0] > prio[1], 0, jnp.where(prio[1] > prio[0], 1, by_speed))

    def _turn(self, key, state: State, actions, params: EnvParams) -> State:
        cfg = self.config
        M, N = cfg.num_moves, cfg.team_size
        k_sw, k_order, k_m0, k_m1 = jax.random.split(key, 4)
        move_keys = jnp.stack([k_m0, k_m1])

        is_switch = (actions >= M) & (actions < M + N)
        is_gimmick = (actions >= M + N) & (actions < 2 * M + N)
        is_move = (actions < M) | is_gimmick
        move_slot = jnp.where(is_gimmick, actions - M - N, actions)
        switch_slot = actions - M

        state = state.replace(protected=jnp.zeros((2,), bool))

        # 1. Switches (faster side first, matters for entry effects).
        first = self._faster(state, params, k_sw)
        for p in (first, 1 - first):
            state = mech.select_tree(is_switch[p],
                                     mech.do_switch(cfg, params, state, p, switch_slot[p]), state)

        # 2. Transform activation.
        if cfg.rules.transform.enabled:
            for p in (0, 1):
                state = mech.select_tree(is_gimmick[p],
                                         mech.activate_transform(cfg, state, p), state)

        # 3. Moves in priority / speed order.
        prio = jnp.stack([mech.move_priority(cfg, params, state, p, move_slot[p], is_gimmick[p])
                          for p in (0, 1)])
        prio = jnp.where(is_move, prio, -99)
        first = self._faster(state, params, k_order, prio)
        for p in (first, 1 - first):
            used = mech.use_move(cfg, params, state, p, move_slot[p], is_gimmick[p],
                                 move_keys[p])
            state = mech.select_tree(is_move[p], used, state)

        # 4. End of turn, faints, termination.
        state = mech.end_of_turn(cfg, params, state)
        return mech.resolve_faints(cfg, state)

    def _forced_switch(self, key, state: State, actions, params: EnvParams) -> State:
        cfg = self.config
        M, N = cfg.num_moves, cfg.team_size
        is_switch = (actions >= M) & (actions < M + N)
        first = self._faster(state, params, key)
        for p in (first, 1 - first):
            do = state.needs_switch[p] & is_switch[p]
            state = mech.select_tree(do, mech.do_switch(cfg, params, state, p, actions[p] - M),
                                     state)
        return mech.resolve_faints(cfg, state)

    # --- Debugging helpers (not jittable) ------------------------------------------------

    def describe_action(self, action: int) -> str:
        M, N = self.config.num_moves, self.config.team_size
        action = int(action)
        if action < M:
            return f"move {action}"
        if action < M + N:
            return f"switch {action - M}"
        if action < 2 * M + N:
            return f"{self.config.rules.transform.name}-transform + move {action - M - N}"
        return "no-op"

    def render(self, state: State, params: EnvParams) -> str:
        """Human-readable summary of a (single, unbatched) state."""
        import numpy as np

        s = jax.tree.map(np.asarray, state)
        lines = [f"turn {int(s.turn)}  phase={'TURN' if s.phase == Phase.TURN else 'SWITCH'}"
                 f"  field={int(s.field)}  done={bool(s.done)}  winner={int(s.winner)}"]
        for p in (0, 1):
            lines.append(f"player {p}:")
            for i in range(self.config.team_size):
                mark = "*" if s.active[p] == i else " "
                hp, mx = s.hp[p, i], s.stats[p, i, HP]
                st = f" status={int(s.status[p, i])}" if s.status[p, i] else ""
                tf = " transformed" if s.transformed[p, i] else ""
                lines.append(f"  {mark} #{int(s.species[p, i]):<4d} types={s.types[p, i].tolist()}"
                             f" hp={hp:.0f}/{mx:.0f}{st}{tf}")
        return "\n".join(lines)
