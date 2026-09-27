"""Baseline state-policies: ``policy(key, state, env_params, player) -> action``.

``random_policy`` and ``greedy_policy`` read the true state (the greedy one
peeks at the opponent's active mon, so it is a *perfect-information* baseline).
``network_policy`` wraps an ``apply_fn`` so it only sees its own observation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from minimon.env import MiniMon
from minimon.mechanics import expected_damage
from minimon.observation import observe
from minimon.rollout import ApplyFn, masked_categorical


def _legal(env: MiniMon, state, env_params, player):
    return env.legal_actions(state, env_params)[player]


def random_policy(env: MiniMon):
    def policy(key, state, env_params, player):
        legal = _legal(env, state, env_params, player)
        return jax.random.categorical(key, jnp.where(legal, 0.0, -jnp.inf))

    return policy


def greedy_policy(env: MiniMon):
    """Use the move with the highest expected damage; switch randomly when forced."""
    M = env.config.num_moves

    def policy(key, state, env_params, player):
        legal = _legal(env, state, env_params, player)
        dmg = jax.vmap(lambda m: expected_damage(env.config, env_params, state, player, m))(
            jnp.arange(M))
        scores = jnp.full((env.num_actions,), -jnp.inf).at[:M].set(dmg)
        scores = jnp.where(legal, scores, -jnp.inf)
        fallback = jax.random.categorical(key, jnp.where(legal, 0.0, -jnp.inf))
        return jnp.where(jnp.any(legal[:M]), jnp.argmax(scores), fallback)

    return policy


def network_policy(env: MiniMon, apply_fn: ApplyFn, params, greedy: bool = False):
    """Wrap ``apply_fn(params, obs) -> logits`` as an observation-only policy."""

    def policy(key, state, env_params, player):
        obs = observe(env.config, env_params, state, player)
        legal = _legal(env, state, env_params, player)
        logits = apply_fn(params, obs)
        if greedy:
            return jnp.argmax(jnp.where(legal, logits, -jnp.inf))
        return masked_categorical(key, logits, legal)[0]

    return policy
