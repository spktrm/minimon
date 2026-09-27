"""Minimal tour: build an env, step it, and collect a jitted self-play rollout."""

import functools

import jax
import jax.numpy as jnp

import minimon

env = minimon.make("medium", gen=9)  # 3v3, hidden info, stochastic, type transform
params = env.default_params(jax.random.key(0))  # procedurally generated metagame

state, obs = env.reset(jax.random.key(1), params)
print(env.render(state, params))
print("obs", obs.shape, "actions", env.num_actions)

out = jax.jit(env.step)(jax.random.key(2), state, jnp.array([0, 0]), params)
print(env.render(out.state, params))


def apply_fn(net, obs):  # any (params, obs[D]) -> logits[A] works, e.g. flax Module.apply
    return obs @ net


net = 0.01 * jax.random.normal(jax.random.key(3), (env.obs_size, env.num_actions))
collect = jax.jit(functools.partial(minimon.rollout, env, apply_fn=apply_fn, num_steps=256))
final_state, traj = collect(params, jax.random.key(4), params=net)
print("episodes finished:", int(traj.done.sum()), "reward sum:", traj.reward.sum(0))
