# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/). The single source of truth for the
version is `__version__` in `src/minimon/__init__.py`.

## [Unreleased]

## [0.1.0] - 2026-09-28

### Added
- Fully-jittable two-player zero-sum environment (`MiniMon`) with simultaneous-move
  turns, forced-switch phases and a fixed-size action space.
- Procedurally generated metagames (`make_params`): generational type charts, species,
  moves, status / field / item / ability tables.
- Generation analogues 1-9 (`gen_rules`) and the parameterised `Transform` gimmick.
- Difficulty presets (`trivial` → `full`), stochasticity toggles (`Chance`) and
  hidden-information observations.
- `rollout` with a network `apply_fn`, `evaluate`, baseline policies and matrix-game
  tools (`payoff_matrix`, `solve_matrix_game`, `exploitability`).
