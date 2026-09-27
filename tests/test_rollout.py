"""Jitted trajectory collection with a network apply_fn."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import minimon
from minimon import policies

KEY = jax.random.key(0)


def linear_apply(params, obs):
    return obs @ params["w"] + params["b"]


def init_linear(env, key=KEY):
    return {"w": 0.01 * jax.random.normal(key, (env.obs_size, env.num_actions)),
            "b": jnp.zeros((env.num_actions,))}


def test_selfplay_rollout_shapes_and_consistency():
    env = minimon.make("medium")
    env_params = env.default_params()
    net = init_linear(env)
    T = 64
    collect = jax.jit(functools.partial(minimon.rollout, env, apply_fn=linear_apply, num_steps=T))
    state, traj = collect(env_params, KEY, params=net)
    assert traj.obs.shape == (T, 2, env.obs_size)
    assert traj.legal.shape == (T, 2, env.num_actions)
    assert traj.action.shape == (T, 2) and traj.reward.shape == (T, 2)
    t, p = np.meshgrid(np.arange(T), np.arange(2), indexing="ij")
    assert bool(traj.legal[t, p, traj.action].all())
    assert bool((traj.log_prob <= 1e-6).all()) and bool(jnp.isfinite(traj.log_prob).all())
    assert np.allclose(traj.reward.sum(-1), 0.0)
    assert bool((traj.reward[~traj.done] == 0).all())


def test_rollout_against_opponent_and_vmapped():
    env = minimon.make("trivial")
    env_params = env.default_params()
    net, opp = init_linear(env), init_linear(env, jax.random.key(1))

    def run(key):
        return minimon.rollout(env, env_params, key, linear_apply, net, 100,
                               opponent_params=opp, opponent_apply_fn=linear_apply)

    _, traj = jax.jit(jax.vmap(run))(jax.random.split(KEY, 4))
    assert traj.obs.shape[:3] == (4, 100, 2)
    assert int(traj.done.sum()) > 0  # short 1v1 games finish and auto-reset


def test_rollout_without_auto_reset_stays_terminal():
    env = minimon.make("trivial")
    env_params = env.default_params()
    net = init_linear(env)
    state, traj = minimon.rollout(env, env_params, KEY, linear_apply, net, 60, auto_reset=False)
    assert bool(state.done)
    first = int(jnp.argmax(traj.done))
    assert np.allclose(traj.reward[first + 1:], 0.0)
    assert bool(traj.legal[first + 1:, :, -1].all())  # only the no-op remains


def test_greedy_beats_random():
    env = minimon.make("medium")
    env_params = env.default_params()
    res = jax.jit(lambda k: minimon.evaluate(env, env_params, k, policies.greedy_policy(env),
                                             policies.random_policy(env), 64))(KEY)
    assert bool(res["all_done"])
    assert float(res["win_rate"]) > 0.6


def test_network_policy_in_evaluate():
    env = minimon.make("easy")
    env_params = env.default_params()
    pol = policies.network_policy(env, linear_apply, init_linear(env))
    res = minimon.evaluate(env, env_params, KEY, pol, policies.random_policy(env), 16)
    total = float(res["win_rate"] + res["loss_rate"] + res["draw_rate"])
    assert abs(total - 1.0) < 1e-6
