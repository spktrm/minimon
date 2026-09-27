"""Self-play REINFORCE with a small MLP in pure JAX.

Collects a batch of vmapped, fully-jitted self-play trajectories, trains with
the policy gradient using per-player Monte-Carlo returns, and periodically
evaluates against a uniform-random opponent.

    python examples/selfplay_reinforce.py --difficulty easy --iters 200
"""

import argparse
import functools

import jax
import jax.numpy as jnp

import minimon
from minimon import policies


def init_mlp(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (i, o)) * jnp.sqrt(2 / i), jnp.zeros(o))
            for k, i, o in zip(keys, sizes[:-1], sizes[1:], strict=True)]


def apply_mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def returns_to_go(reward, done, gamma):
    """Discounted returns within episodes. reward: [T, 2], done: [T]."""

    def body(g, inp):
        r, d = inp
        g = r + gamma * g * (1.0 - d[..., None])
        return g, g

    _, g = jax.lax.scan(body, jnp.zeros(reward.shape[1:]), (reward, done.astype(jnp.float32)),
                        reverse=True)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--gen", type=int, default=9)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    env = minimon.make(args.difficulty, gen=args.gen)
    env_params = env.default_params(jax.random.key(0))
    key = jax.random.key(1)
    net = init_mlp(key, [env.obs_size, 128, 128, env.num_actions])

    def loss_fn(net, key):
        keys = jax.random.split(key, args.envs)
        collect = functools.partial(minimon.rollout, env, env_params, apply_fn=apply_mlp,
                                    num_steps=args.steps)
        _, traj = jax.vmap(lambda k: collect(k, params=jax.lax.stop_gradient(net)))(keys)
        # Recompute log-probs with gradients: [E, T, 2, A].
        logits = apply_mlp(net, traj.obs)
        logp = jax.nn.log_softmax(jnp.where(traj.legal, logits, -1e9), axis=-1)
        logp_a = jnp.take_along_axis(logp, traj.action[..., None], -1)[..., 0]
        g = jax.vmap(lambda r, d: returns_to_go(r, d, 0.99))(traj.reward, traj.done)
        g = (g - g.mean()) / (g.std() + 1e-6)
        return -(logp_a * g).mean(), traj.done.sum()

    def adam(net, grads, opt, t, b1=0.9, b2=0.999):
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt[0], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt[1], grads)
        step = lambda p, m, v: p - args.lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + 1e-8)  # noqa: E731
        return jax.tree.map(step, net, m, v), (m, v)

    @jax.jit
    def update(net, opt, t, key):
        (loss, episodes), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, key)
        net, opt = adam(net, grads, opt, t)
        return net, opt, loss, episodes

    @jax.jit
    def evaluate(net, key):
        me = policies.network_policy(env, apply_mlp, net)
        return minimon.evaluate(env, env_params, key, me, policies.random_policy(env), 256)

    opt = (jax.tree.map(jnp.zeros_like, net), jax.tree.map(jnp.zeros_like, net))
    for it in range(args.iters + 1):
        key, k_up, k_ev = jax.random.split(key, 3)
        if it % 25 == 0:
            res = evaluate(net, k_ev)
            print(f"iter {it:4d}  win-rate vs random {float(res['win_rate']):.3f}  "
                  f"draw {float(res['draw_rate']):.3f}")
        net, opt, loss, episodes = update(net, opt, it + 1, k_up)


if __name__ == "__main__":
    main()
