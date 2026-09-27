"""Observations must not leak unrevealed opponent information."""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import BOOST, all_moves, bare_rules, det_env
from test_env import fresh, step

import minimon
from minimon.observation import _global_features


def obs0(env, state, params):
    return np.asarray(env.observe(state, params)[0])


def test_unrevealed_opponent_attributes_do_not_leak():
    env = minimon.make("medium", gen=9)
    params = env.default_params()
    state = fresh(env, params)
    base = obs0(env, state, params)
    hidden_changes = dict(
        moves=state.moves.at[1].set((state.moves[1] + 7) % params.num_dex_moves),
        item=state.item.at[1].set(state.item[1] % env.config.num_items + 1),
        ability=state.ability.at[1].set(state.ability[1] % env.config.num_abilities + 1),
        alt_type=state.alt_type.at[1].set((state.alt_type[1] + 1) % params.num_types),
        status_turns=state.status_turns.at[1].set(5),
    )
    for name, value in hidden_changes.items():
        assert np.array_equal(obs0(env, state.replace(**{name: value}), params), base), name
    # The same changes to one's own team are visible.
    own = state.replace(moves=state.moves.at[0].set((state.moves[0] + 7) % params.num_dex_moves))
    assert not np.array_equal(obs0(env, own, params), base)


def test_moves_revealed_when_used(toy_params):
    env = det_env(rules=bare_rules(9))
    params = toy_params(env)
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    s = step(env, state, [0, 0], params).state
    assert bool(s.revealed_moves[1, 0, 0]) and not bool(s.revealed_moves[1, 0, 1:].any())
    changed = s.replace(moves=s.moves.at[1, 0, 0].set(5))
    assert not np.array_equal(obs0(env, changed, params), obs0(env, s, params))
    changed = s.replace(moves=s.moves.at[1, 0, 1].set(5))
    assert np.array_equal(obs0(env, changed, params), obs0(env, s, params))


def test_species_hidden_without_team_preview():
    env = minimon.make("medium", gen=4)
    params = env.default_params()
    state = fresh(env, params)
    T = params.num_types
    bench = state.replace(types=state.types.at[1, 1].set(jnp.array([0, T])))
    lead = state.replace(types=state.types.at[1, 0].set(jnp.array([0, T])))
    if not np.array_equal(state.types[1, 1], bench.types[1, 1]):
        assert np.array_equal(obs0(env, bench, params), obs0(env, state, params))
    if not np.array_equal(state.types[1, 0], lead.types[1, 0]):
        assert not np.array_equal(obs0(env, lead, params), obs0(env, state, params))


def test_full_information_mode_is_symmetric():
    env = minimon.make("medium", gen=9, hidden_info=False)
    params = env.default_params()
    state = fresh(env, params)
    obs = np.asarray(env.observe(state, params))
    G = jax.eval_shape(lambda: _global_features(env.config, params, state, 0)).shape[0]
    team = (obs.shape[1] - G) // 2
    assert np.array_equal(obs[0, G + team:], obs[1, G:G + team])
    assert np.array_equal(obs[1, G + team:], obs[0, G:G + team])
