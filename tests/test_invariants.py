"""State invariants under random self-play, for every generation analogue."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import minimon
from minimon import policies
from minimon.constants import HP, MAX_BOOST


def random_selfplay(env, params, key, steps):
    pol = policies.random_policy(env)
    k_reset, k_run = jax.random.split(key)
    state, _ = env.reset(k_reset, params)

    def body(state, key):
        k0, k1, k_step, k_reset = jax.random.split(key, 4)
        actions = jnp.stack([pol(k0, state, params, 0), pol(k1, state, params, 1)])
        out = env.step(k_step, state, actions, params)
        fresh, _ = env.reset(k_reset, params)
        nxt = minimon.mechanics.select_tree(out.done, fresh, out.state)
        return nxt, (out.state, out.obs, out.reward)

    _, traj = jax.lax.scan(body, state, jax.random.split(k_run, steps))
    return traj


@pytest.mark.parametrize("gen", range(1, 10))
def test_invariants(gen):
    env = minimon.make("medium", gen=gen)
    params = env.default_params()
    states, obs, rewards = jax.jit(lambda k: random_selfplay(env, params, k, 200))(
        jax.random.key(gen))
    hp, max_hp = np.asarray(states.hp), np.asarray(states.stats[..., HP])
    assert (hp >= 0).all() and (hp <= max_hp + 1e-4).all()
    assert (np.abs(np.asarray(states.boosts)) <= MAX_BOOST).all()
    assert np.isfinite(np.asarray(obs)).all()
    assert np.allclose(np.asarray(rewards).sum(-1), 0.0)
    assert (np.asarray(states.gimmick_used).sum(-1) <= 2).all()
    assert (np.asarray(states.field) <= env.config.num_fields).all()
    assert (np.asarray(states.status) <= env.config.num_statuses).all()
    # Any finished game has a well-defined outcome.
    done = np.asarray(states.done)
    assert done.any()
    assert np.isin(np.asarray(states.winner)[done], [-1, 0, 1]).all()
