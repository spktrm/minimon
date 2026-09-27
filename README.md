# minimon

**A fully-jittable JAX environment that distils competitive Pokémon battling into a
two-player, zero-sum, sequential matrix game** — for fast game-theory and multi-agent RL
research without a Showdown server.

- **Pure JAX, fully jittable.** `reset`, `step`, observation and legal-action masks are
  pure functions. Entire self-play trajectories, including your network's `apply_fn`,
  compile into a single `lax.scan` and `vmap` across thousands of parallel battles.
- **A stage game every turn.** Both players choose simultaneously, so each state is an
  `A × A` zero-sum matrix game. Tools for payoff matrices and Nash solving are included.
- **Stochasticity analogues:** damage rolls, crits, accuracy, secondary effects,
  status rolls and speed ties. Each can be switched off independently.
- **Hidden-information analogues:** opponent moves, held items, abilities, alternate
  types and (without team preview) species stay hidden until they are revealed in battle.
- **Generational analogues, not copies.** Nothing is transcribed 1:1 from the games. Type
  charts, species, moves, statuses, field conditions (weather/terrain), items, abilities
  and the once-per-battle "gimmick" are all *procedurally generated or parameterised*.
  The 9 generation presets are different points in that parameter space.
- **Difficulty presets**, from a 1v1 deterministic perfect-information matrix game up to
  6v6 with hidden information.

## Install

```bash
pip install minimon-jax            # from PyPI (import name: minimon)
pip install -e ".[dev]"            # local development
pip install "jax[cuda12]"          # optional GPU backend
```

## Quickstart

```python
import functools, jax, jax.numpy as jnp
import minimon

env = minimon.make("medium", gen=9)             # 3v3, hidden info, stochastic
params = env.default_params(jax.random.key(0))  # a procedurally generated metagame

state, obs = env.reset(jax.random.key(1), params)          # obs: f32[2, D], one row per player
legal = env.legal_actions(state, params)                    # bool[2, A]
out = jax.jit(env.step)(jax.random.key(2), state, jnp.array([0, 0]), params)
out.state, out.obs, out.reward, out.done                    # reward: f32[2], zero-sum

# Full trajectories with any network: apply_fn(params, obs[D]) -> logits[A]
collect = jax.jit(functools.partial(minimon.rollout, env, apply_fn=my_net.apply, num_steps=128))
final_state, traj = collect(params, jax.random.key(3), params=net_params)   # self-play
# ...or against a fixed opponent: opponent_params=..., opponent_apply_fn=...
# jax.vmap over keys for parallel environments; vmap over env params for domain randomisation.
```

See `examples/quickstart.py` and `examples/selfplay_reinforce.py` for a complete
pure-JAX self-play policy-gradient loop.

## The game

Each `step` is one simultaneous-move stage. The action space has a fixed size
`A = 2M + N + 1` across all configurations:

| range          | action                                               |
|----------------|------------------------------------------------------|
| `[0, M)`       | use move `i`                                         |
| `[M, M+N)`     | switch to team slot `j`                              |
| `[M+N, 2M+N)`  | activate the once-per-battle transform, then move `i` |
| `2M+N`         | no-op (only legal when you have nothing to decide)   |

A turn resolves switches first, then transforms, then moves in priority → speed → tie-break
order, then end-of-turn effects. When an active mon faints, the next step is a
**forced-switch phase**: only the affected player(s) choose. The player who is not
switching gets the no-op. Episodes end when a side has no mons left, or at
`max_turns` (a draw, or an HP-fraction tiebreak with `turn_limit="hp"`). Rewards are
`+1 / −1 / 0` on the terminal transition. Illegal actions are replaced by the first legal
action, but you should mask with `legal_actions`.

## Generation analogues

`minimon.gen_rules(g)` returns a `Rules` preset. Every magnitude is a field you can override.

| gen | adds / changes                                                                        |
|-----|----------------------------------------------------------------------------------------|
| 1   | one shared special stat, category by move type, speed-based crits, harsh statuses; the type chart has a dominant type and a "bugged" immunity |
| 2   | held items, field conditions, special stat split; two new types counter the dominant type and the bug is fixed |
| 3   | abilities (ability-set fields are permanent)                                          |
| 4   | per-move physical/special category                                                    |
| 5   | team preview (opponent species revealed)                                              |
| 6   | **stat transform** (permanent stat multiplier); a new type counters the strongest attacker; weaker crits |
| 7   | **move transform** (one boosted, sure-hit move)                                       |
| 8   | **size transform** (temporary HP multiplier, boosted moves, status → priority guard) |
| 9   | **type transform** (become a hidden alternate type with its own same-type bonus)      |

Transforms are just `minimon.Transform(...)` parameter bundles (`stat_mult`, `hp_mult`,
`duration`, `power_*`, `sure_hit`, `status_guard`, `retype`, …), so you can build new
ones:

```python
import dataclasses
tf = minimon.Transform("custom", True, stat_mult=(1, 1, 1, 1, 2.0), retype=True, duration=2)
cfg = minimon.make_config("hard", gen=9)
env = minimon.MiniMon(cfg.replace(rules=dataclasses.replace(cfg.rules, transform=tf)))
```

## Procedurally generated content (`EnvParams`)

`make_params(config, key)` samples an entire metagame. It is traceable, so
`jax.vmap(lambda k: make_params(cfg, k))(keys)` gives a batch of distinct metagames.

- **Type chart**: `"generational"` (random entries plus the structural generation
  analogues above) or `"random"` (unstructured).
- **Species & moves**: random types, base stats, learnsets, and moves with power,
  accuracy, priority and secondary effects.
- **Statuses**: table rows with chip damage, stat multipliers, skip chance, duration and
  recovery. The first five follow archetypes whose magnitudes come from `Rules`.
- **Field conditions** (weather/terrain analogues): per-type damage multipliers and
  optional chip damage with immune types.
- **Items / abilities**: modifier bundles (stat/damage multipliers, move lock, recoil,
  per-turn heal or boost, entry debuff or field, switch-out heal, endure, low-HP heal,
  consumable). They are revealed to the opponent when a visible effect triggers.

Observations encode each entity by its **properties**, not just its ID, so a policy can
generalise across metagames.

## Difficulty presets

| preset    | team | moves | types | chance | info    | extras                                  |
|-----------|------|-------|-------|--------|---------|-----------------------------------------|
| `trivial` | 1    | 2     | 4     | off    | perfect | no items/abilities/fields/transform     |
| `easy`    | 2    | 2     | 6     | off    | perfect | no items/abilities/fields/transform     |
| `medium`  | 3    | 4     | gen   | on     | hidden  | full generation rules                   |
| `hard`    | 6    | 4     | gen   | on     | hidden  | full generation rules                   |
| `full`    | 6    | 4     | gen   | on     | hidden  | larger dex and content tables           |

Mix and match freely: `minimon.make("hard", gen=4, hidden_info=False, chance=minimon.Chance.deterministic())`.

## Matrix-game tools

```python
from minimon import matrix
payoff, legal = matrix.payoff_matrix(env, key, state, params, value_fn=matrix.hp_value)
x, y, value = matrix.solve_matrix_game(payoff, legal[0], legal[1])   # regret-matching+
matrix.exploitability(payoff, x, y)
```

## Baselines and evaluation

```python
from minimon import policies
res = minimon.evaluate(env, params, key, policies.greedy_policy(env), policies.random_policy(env), 256)
net_policy = policies.network_policy(env, apply_fn, net_params)   # sees only its own observation
```

## Development

```bash
pip install -e ".[dev]"
pytest                # 4 workers by default; `-n 0` for serial, lower -n if RAM-limited
ruff check src tests
python -m build && twine check dist/*
```

The version is `__version__` in `src/minimon/__init__.py` (also `minimon.__version__` at
runtime); see `CHANGELOG.md`. Release steps for PyPI, manual or via GitHub, are in
[`PUBLISHING.md`](PUBLISHING.md).

## License

MIT
