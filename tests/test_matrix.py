"""Stage-game utilities."""

import jax
import jax.numpy as jnp
import numpy as np

import minimon
from minimon import matrix

RPS = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def test_rock_paper_scissors():
    x, y, v = matrix.solve_matrix_game(RPS, num_iters=5000)
    assert np.allclose(x, 1 / 3, atol=0.02) and np.allclose(y, 1 / 3, atol=0.02)
    assert abs(float(v)) < 0.01
    assert float(matrix.exploitability(RPS, x, y)) < 0.02


def test_dominant_strategy():
    game = jnp.array([[1.0, 2.0], [0.0, 0.5]])
    x, y, v = matrix.solve_matrix_game(game)
    assert float(x[0]) > 0.99 and float(y[0]) > 0.99
    assert abs(float(v) - 1.0) < 0.02


def test_masked_actions_get_zero_probability():
    row_mask = jnp.array([True, True, False])
    x, y, _ = matrix.solve_matrix_game(RPS, row_mask=row_mask)
    assert float(x[2]) == 0.0
    assert abs(float(x.sum()) - 1.0) < 1e-5


def test_payoff_matrix_on_env():
    env = minimon.make("trivial")
    params = env.default_params()
    state, _ = env.reset(jax.random.key(0), params)
    payoff, legal = jax.jit(lambda k, s: matrix.payoff_matrix(env, k, s, params, num_samples=2))(
        jax.random.key(1), state)
    A = env.num_actions
    assert payoff.shape == (A, A) and legal.shape == (2, A)
    assert bool(jnp.isfinite(payoff).all())
    assert float(jnp.abs(payoff).max()) <= 1.0 + 1e-6
    x, y, _ = matrix.solve_matrix_game(payoff, legal[0], legal[1])
    assert float(jnp.where(legal[0], 0.0, x).sum()) == 0.0
    assert float(jnp.where(legal[1], 0.0, y).sum()) == 0.0


def test_expand_shapes():
    env = minimon.make("trivial")
    params = env.default_params()
    state, _ = env.reset(jax.random.key(0), params)
    out = matrix.expand(env, jax.random.key(1), state, params)
    A = env.num_actions
    assert out.reward.shape == (A, A, 2)
    assert out.state.hp.shape == (A, A, 2, env.config.team_size)
