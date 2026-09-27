"""Core API and battle mechanics."""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import (
    ATTACK,
    BOOST,
    INFLICT,
    SET_FIELD,
    all_moves,
    bare_rules,
    det_env,
    jitted,
    set_move,
)

import minimon
from minimon import mechanics as mech
from minimon.constants import B_ATK, HP, SPE, Phase

KEY = jax.random.key(0)


def fresh(env, params, key=KEY):
    reset, _ = jitted(env)
    return reset(key, params)[0]


def step(env, state, actions, params, key=KEY):
    _, step_fn = jitted(env)
    return step_fn(key, state, jnp.asarray(actions), params)


# --- API ---------------------------------------------------------------------------


def test_reset_shapes_and_initial_state():
    env = minimon.make("medium")
    params = env.default_params()
    state, obs = jax.jit(env.reset)(KEY, params)
    N, M, A = env.config.team_size, env.config.num_moves, env.num_actions
    assert obs.shape == (2, env.obs_size)
    assert bool(jnp.isfinite(obs).all())
    assert np.allclose(state.hp, state.stats[..., HP])
    assert state.active.tolist() == [0, 0]
    assert not bool(state.done) and int(state.phase) == Phase.TURN
    legal = env.legal_actions(state, params)
    assert legal.shape == (2, A)
    assert bool(legal[:, :M].all())
    assert not bool(legal[:, M].any())  # cannot switch to the active slot
    assert bool(legal[:, M + 1:M + N].all())
    assert not bool(legal[:, -1].any())  # no-op only when nothing to decide


def test_step_is_deterministic_given_key():
    env = minimon.make("medium")
    params = env.default_params()
    state = fresh(env, params)
    a = step(env, state, [0, 1], params, jax.random.key(3))
    b = step(env, state, [0, 1], params, jax.random.key(3))
    for x, y in zip(jax.tree.leaves(a.state), jax.tree.leaves(b.state), strict=True):
        assert np.array_equal(x, y)


def test_vmap_over_batch_of_envs_and_metagames():
    env = minimon.make("medium")
    keys = jax.random.split(KEY, 8)
    params = jax.vmap(env.default_params)(keys)
    states, obs = jax.vmap(env.reset)(keys, params)
    out = jax.jit(jax.vmap(env.step))(keys, states, jnp.zeros((8, 2), jnp.int32), params)
    assert obs.shape == (8, 2, env.obs_size)
    assert out.reward.shape == (8, 2)
    assert int(out.state.turn.min()) == 1


def test_illegal_action_is_sanitised():
    env = minimon.make("medium")
    params = env.default_params()
    state = fresh(env, params)
    M = env.config.num_moves
    out = step(env, state, [M, env.noop_action], params)  # both illegal
    legal = env.legal_actions(state, params)
    assert bool(legal[jnp.arange(2), out.info["actions"]].all())


# --- Mechanics -----------------------------------------------------------------------


def test_damage_and_forced_switch(toy_params):
    env = det_env(rules=bare_rules(9))
    params = toy_params(env)
    M = env.config.num_moves
    state = fresh(env, params)
    state = all_moves(all_moves(state, 0, ATTACK), 1, BOOST)
    state = state.replace(hp=state.hp.at[1, 0].set(1.0))

    out = step(env, state, [0, 0], params)
    s = out.state
    assert float(s.hp[1, 0]) == 0.0
    assert int(s.phase) == Phase.SWITCH and s.needs_switch.tolist() == [False, True]
    legal = env.legal_actions(s, params)
    assert legal[0].tolist() == [False] * (env.num_actions - 1) + [True]
    assert legal[1, M + 1] and legal[1, M + 2] and not legal[1, :M].any()

    out2 = step(env, s, [env.noop_action, M + 2], params)
    assert int(out2.state.active[1]) == 2
    assert int(out2.state.phase) == Phase.TURN
    assert int(out2.state.turn) == int(s.turn)  # forced switches are not turns
    assert float(out2.reward.sum()) == 0.0


def test_termination_is_zero_sum_and_absorbing(toy_params):
    env = det_env(difficulty="trivial")
    params = toy_params(env)
    state = fresh(env, params)
    state = all_moves(all_moves(state, 0, ATTACK), 1, BOOST)
    state = state.replace(hp=state.hp.at[1, 0].set(1.0))
    out = step(env, state, [0, 0], params)
    assert bool(out.done) and int(out.state.winner) == 0
    assert out.reward.tolist() == [1.0, -1.0]
    again = step(env, out.state, [env.noop_action] * 2, params)
    assert again.reward.tolist() == [0.0, 0.0]
    assert np.array_equal(again.state.hp, out.state.hp)


def test_turn_limit_draw_and_hp_tiebreak(toy_params):
    for limit, expected in (("draw", -1), ("hp", 0)):
        env = det_env(difficulty="trivial", max_turns=1, turn_limit=limit)
        params = toy_params(env)
        state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
        state = state.replace(hp=state.hp.at[1, 0].multiply(0.5))
        out = step(env, state, [0, 0], params)
        assert bool(out.done) and int(out.state.winner) == expected


def test_speed_and_priority_order(toy_params):
    env = det_env(difficulty="trivial")
    params = toy_params(env)
    state = all_moves(all_moves(fresh(env, params), 0, ATTACK), 1, ATTACK)
    state = state.replace(hp=jnp.ones_like(state.hp),
                          stats=state.stats.at[0, 0, SPE].set(999.0).at[1, 0, SPE].set(1.0))
    assert int(step(env, state, [0, 0], params).state.winner) == 0  # faster side wins
    params = set_move(params, 5, type=0, power=40.0, priority=1)
    state = all_moves(state, 1, 5)
    assert int(step(env, state, [0, 0], params).state.winner) == 1  # priority beats speed


def test_boost_status_and_field_moves(toy_params):
    env = det_env(rules=bare_rules(9, fields=True))
    params = toy_params(env)
    state = fresh(env, params)
    state = all_moves(state, 0, BOOST)
    state = state.replace(moves=state.moves.at[1, 0, 0].set(INFLICT).at[1, 0, 1].set(SET_FIELD))
    s = step(env, state, [0, 0], params).state
    assert int(s.boosts[0, B_ATK]) == 2
    assert int(s.status[0, 0]) == 1
    s = step(env, s, [0, 1], params).state
    assert int(s.field) == 1
    assert int(s.field_turns) == env.config.rules.move_field_turns - 1


def test_fields_disabled_in_gen1(toy_params):
    env = det_env(gen=1, rules=bare_rules(1))
    params = toy_params(env)
    state = all_moves(all_moves(fresh(env, params), 0, SET_FIELD), 1, SET_FIELD)
    assert int(step(env, state, [0, 0], params).state.field) == 0


def test_type_immunity(toy_params):
    env = det_env(rules=bare_rules(9))
    params = toy_params(env)
    params = params.replace(type_chart=params.type_chart.at[0, :].set(0.0))
    state = all_moves(all_moves(fresh(env, params), 0, ATTACK), 1, BOOST)
    s = step(env, state, [0, 0], params).state
    assert float(s.hp[1, 0]) == float(state.hp[1, 0])


def test_stab_multiplier(toy_params):
    env = det_env(rules=bare_rules(9))
    params = toy_params(env)
    T = params.num_types
    state = fresh(env, params)

    def dmg(types):
        s = state.replace(types=state.types.at[0, 0].set(jnp.array(types)))
        return float(mech.damage(env.config, params, s, 0, ATTACK, 80.0, False, 1.0)[0])

    ratio = dmg([0, T]) / dmg([1, T])
    assert 1.4 < ratio < 1.6
