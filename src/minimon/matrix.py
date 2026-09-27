"""Stage-game tools: every state is a zero-sum ``A x A`` matrix game.

* ``expand``           next states for all joint actions (vmapped ``step``)
* ``payoff_matrix``    one-step lookahead payoffs under a value function,
                       averaged over chance samples
* ``solve_matrix_game`` Nash equilibrium via regret-matching+ (jittable)
* ``exploitability``   NashConv of a strategy profile

These make it cheap to build search / CFR-style baselines directly on top of
the environment.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from minimon.constants import HP
from minimon.env import MiniMon, StepOutput
from minimon.state import State

ValueFn = Callable[[State], jax.Array]


def hp_value(state: State) -> jax.Array:
    """Heuristic value for player 0 in [-1, 1]: difference in team HP fraction."""
    frac = jnp.sum(state.hp / state.stats[..., HP], axis=-1) / state.hp.shape[-1]
    return frac[..., 0] - frac[..., 1]


def expand(env: MiniMon, key: jax.Array, state: State, env_params) -> StepOutput:
    """Step every joint action with the same chance key (common random numbers).

    Returns a ``StepOutput`` whose leaves have leading shape ``[A, A]``.
    """
    A = env.num_actions
    grid = jnp.stack(jnp.meshgrid(jnp.arange(A), jnp.arange(A), indexing="ij"), -1)
    step = lambda a: env.step(key, state, a, env_params)  # noqa: E731
    return jax.vmap(jax.vmap(step))(grid)


def payoff_matrix(env: MiniMon, key: jax.Array, state: State, env_params,
                  value_fn: ValueFn = hp_value, num_samples: int = 8):
    """Expected one-step payoff for player 0 over all joint actions.

    Returns ``(payoff f32[A, A], legal bool[2, A])``. Rows/columns of illegal
    actions duplicate the fallback legal action; mask them with ``legal``.
    """

    def sample(k):
        out = expand(env, k, state, env_params)
        return out.reward[..., 0] + jnp.where(out.done, 0.0, value_fn(out.state))

    payoff = jax.vmap(sample)(jax.random.split(key, num_samples)).mean(0)
    return payoff, env.legal_actions(state, env_params)


def _normalise(r, mask):
    pos = jnp.maximum(r, 0.0) * mask
    total = pos.sum()
    uniform = mask / jnp.maximum(mask.sum(), 1.0)
    return jnp.where(total > 0, pos / jnp.maximum(total, 1e-12), uniform)


def solve_matrix_game(payoff: jax.Array, row_mask=None, col_mask=None, num_iters: int = 2000):
    """Approximate Nash equilibrium of a zero-sum game (row player maximises).

    Regret-matching+ with alternating updates and linear averaging.
    Returns ``(x, y, value)``.
    """
    R, C = payoff.shape
    rm = jnp.ones((R,)) if row_mask is None else row_mask.astype(jnp.float32)
    cm = jnp.ones((C,)) if col_mask is None else col_mask.astype(jnp.float32)
    # Neutralise illegal entries so they cannot influence utilities.
    payoff = payoff * rm[:, None] * cm[None, :]

    def body(carry, t):
        rx, ry, sx, sy = carry
        x = _normalise(rx, rm)
        y = _normalise(ry, cm)
        ux = payoff @ y
        rx = jnp.maximum(rx + (ux - x @ ux) * rm, 0.0)
        x = _normalise(rx, rm)
        uy = -(x @ payoff)
        ry = jnp.maximum(ry + (uy - y @ uy) * cm, 0.0)
        y = _normalise(ry, cm)
        w = t + 1.0
        return (rx, ry, sx + w * x, sy + w * y), None

    init = (jnp.zeros((R,)), jnp.zeros((C,)), jnp.zeros((R,)), jnp.zeros((C,)))
    (_, _, sx, sy), _ = jax.lax.scan(body, init, jnp.arange(num_iters, dtype=jnp.float32))
    x, y = sx / sx.sum(), sy / sy.sum()
    return x, y, x @ payoff @ y


def exploitability(payoff: jax.Array, x: jax.Array, y: jax.Array, row_mask=None, col_mask=None):
    """NashConv: total gain available to best-responding players (0 at equilibrium)."""
    br_row = payoff @ y
    br_col = x @ payoff
    if row_mask is not None:
        br_row = jnp.where(row_mask, br_row, -jnp.inf)
    if col_mask is not None:
        br_col = jnp.where(col_mask, br_col, jnp.inf)
    return br_row.max() - br_col.min()
