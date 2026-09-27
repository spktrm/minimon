"""Fully-jitted trajectory collection with a network ``apply_fn``.

``apply_fn(params, obs) -> logits`` maps a single ``f32[D]`` observation to
``f32[A]`` action logits (e.g. ``flax.linen.Module.apply`` or a plain
function). Illegal actions are masked before sampling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from minimon.env import MiniMon
from minimon.mechanics import select_tree
from minimon.state import State

ApplyFn = Callable[[Any, jax.Array], jax.Array]


class Trajectory(NamedTuple):
    """Time-major batch of transitions. Player axis is 1."""

    obs: jax.Array  # f32[T, 2, D]
    legal: jax.Array  # bool[T, 2, A]
    action: jax.Array  # i32[T, 2]
    log_prob: jax.Array  # f32[T, 2]
    reward: jax.Array  # f32[T, 2]
    done: jax.Array  # bool[T]   episode ended at this step (next obs is a reset)


def masked_categorical(key: jax.Array, logits: jax.Array, legal: jax.Array):
    """Sample from ``softmax(logits)`` restricted to ``legal``; returns (action, log_prob)."""
    logits = jnp.where(legal, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    action = jax.random.categorical(key, logits, axis=-1)
    return action, jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def rollout(
    env: MiniMon,
    env_params,
    key: jax.Array,
    apply_fn: ApplyFn,
    params,
    num_steps: int,
    *,
    opponent_params=None,
    opponent_apply_fn: ApplyFn | None = None,
    init_state: State | None = None,
    auto_reset: bool = True,
) -> tuple[State, Trajectory]:
    """Collect ``num_steps`` joint transitions in a single ``lax.scan``.

    Self-play when no opponent is given; otherwise player 0 uses
    ``(apply_fn, params)`` and player 1 uses ``(opponent_apply_fn or apply_fn,
    opponent_params)``. With ``auto_reset`` finished episodes are immediately
    restarted, so ``Trajectory.done`` marks episode boundaries.

    Jit with ``env``, ``apply_fn``, ``num_steps`` static, e.g.::

        collect = jax.jit(functools.partial(rollout, env, apply_fn=net.apply, num_steps=128))
        state, traj = collect(env_params, key, params=net_params)

    and ``jax.vmap`` over ``key`` (and ``init_state``) for parallel environments.
    """
    if init_state is None:
        key, k_reset = jax.random.split(key)
        init_state, _ = env.reset(k_reset, env_params)

    self_play = opponent_params is None and opponent_apply_fn is None
    opp_fn = opponent_apply_fn or apply_fn
    opp_params = params if opponent_params is None else opponent_params

    def logits_fn(obs):
        if self_play:
            return jax.vmap(apply_fn, in_axes=(None, 0))(params, obs)
        return jnp.stack([apply_fn(params, obs[0]), opp_fn(opp_params, obs[1])])

    def body(state, key):
        k_act, k_step, k_reset = jax.random.split(key, 3)
        obs = env.observe(state, env_params)
        legal = env.legal_actions(state, env_params)
        action, log_prob = masked_categorical(k_act, logits_fn(obs), legal)
        out = env.step(k_step, state, action, env_params)
        next_state = out.state
        if auto_reset:
            fresh, _ = env.reset(k_reset, env_params)
            next_state = select_tree(out.done, fresh, next_state)
        return next_state, Trajectory(obs, legal, action, log_prob, out.reward, out.done)

    return jax.lax.scan(body, init_state, jax.random.split(key, num_steps))


def max_episode_steps(env: MiniMon) -> int:
    """Upper bound on env steps per episode (turns + forced-switch steps)."""
    return env.config.max_turns + 2 * env.config.team_size


# A state policy sees the full state and is told which player it controls.
# Wrap networks with ``minimon.policies.network_policy`` so they only see obs.
StatePolicy = Callable[[jax.Array, State, Any, int], jax.Array]


def play_episode(env: MiniMon, env_params, key: jax.Array, policy0: StatePolicy,
                 policy1: StatePolicy) -> State:
    """Play one episode to completion and return the terminal state."""
    k_reset, k_run = jax.random.split(key)
    state, _ = env.reset(k_reset, env_params)

    def body(state, key):
        k0, k1, k_step = jax.random.split(key, 3)
        actions = jnp.stack([policy0(k0, state, env_params, 0),
                             policy1(k1, state, env_params, 1)])
        return env.step(k_step, state, actions, env_params).state, None

    state, _ = jax.lax.scan(body, state, jax.random.split(k_run, max_episode_steps(env)))
    return state


def evaluate(env: MiniMon, env_params, key: jax.Array, policy0: StatePolicy,
             policy1: StatePolicy, num_episodes: int = 128) -> dict[str, jax.Array]:
    """Head-to-head results from player 0's perspective."""
    finals = jax.vmap(lambda k: play_episode(env, env_params, k, policy0, policy1))(
        jax.random.split(key, num_episodes))
    win = (finals.winner == 0).astype(jnp.float32)
    loss = (finals.winner == 1).astype(jnp.float32)
    return {
        "win_rate": win.mean(),
        "loss_rate": loss.mean(),
        "draw_rate": 1.0 - win.mean() - loss.mean(),
        "mean_return": (win - loss).mean(),
        "mean_turns": finals.turn.astype(jnp.float32).mean(),
        "all_done": finals.done.all(),
    }
