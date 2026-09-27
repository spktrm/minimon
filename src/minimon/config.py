"""Static environment configuration.

Everything here is hashable and Python-level: it determines array shapes and
which code paths are traced, so changing it triggers recompilation. Anything
that should vary per-episode or be vmapped over lives in ``EnvParams`` instead.

* ``Transform`` - the once-per-battle gimmick, as a point in a parameter space
  (stat / HP multipliers, duration, move-power transform, retyping, ...).
* ``Rules``     - generation-analogue mechanics (see ``gen_rules``).
* ``Chance``    - which sources of stochasticity are enabled.
* ``EnvConfig`` - sizes, observability and the above. ``make_config`` builds
  one from a difficulty preset and a generation.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

INF = 1e9


@dataclasses.dataclass(frozen=True)
class Transform:
    """A once-per-battle transformation of the active mon.

    Activated through the gimmick action range. While transformed:

    * ``stat_mult`` scales the five non-HP stats, ``hp_mult`` scales HP.
    * damaging moves have power ``clip(power_base + power_scale * power,
      power_min, power_max)``; with ``single_move`` this applies to the one
      move used on activation and nothing else persists.
    * ``sure_hit`` moves cannot miss; ``suppress_secondary`` removes secondary
      effects; ``status_guard`` turns status moves into a priority guard.
    * ``status_boost_all`` raises all stats when a status move is used.
    * ``retype`` changes the mon's type to its hidden alternate type, with
      ``retype_stab`` same-type bonus when it matches an original type.

    ``duration > 0`` makes it temporary (also ending on switch-out / faint);
    ``0`` means it lasts for the rest of the battle.
    """

    name: str = "none"
    enabled: bool = False
    stat_mult: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    hp_mult: float = 1.0
    duration: int = 0
    single_move: bool = False
    power_base: float = 0.0
    power_scale: float = 1.0
    power_min: float = 0.0
    power_max: float = INF
    sure_hit: bool = False
    suppress_secondary: bool = False
    status_guard: bool = False
    status_boost_all: int = 0
    retype: bool = False
    retype_stab: float = 2.0

    @property
    def timed(self) -> bool:
        return self.enabled and not self.single_move and self.duration > 0

    @property
    def changes_power(self) -> bool:
        return (self.power_base != 0.0 or self.power_scale != 1.0
                or self.power_min > 0.0 or self.power_max < INF)


NO_TRANSFORM = Transform()
# Generation analogues: each later generation explores a different axis.
STAT_TRANSFORM = Transform("stat", True, stat_mult=(1.2, 1.2, 1.2, 1.2, 1.2))
MOVE_TRANSFORM = Transform("move", True, single_move=True, power_scale=1.6, power_min=100.0,
                           power_max=200.0, sure_hit=True, status_boost_all=1)
SIZE_TRANSFORM = Transform("size", True, hp_mult=2.0, duration=3, power_base=70.0,
                           power_scale=0.5, power_max=150.0, sure_hit=True,
                           suppress_secondary=True, status_guard=True)
TYPE_TRANSFORM = Transform("type", True, retype=True, retype_stab=2.0)


@dataclasses.dataclass(frozen=True)
class Rules:
    """Mechanics toggles and magnitudes. ``gen_rules(g)`` gives generation analogues."""

    gen: int = 9
    # --- Stat / damage model --------------------------------------------------------
    special_stat_split: bool = True  # False: one shared special stat
    physical_special_split: bool = True  # False: category decided by move type
    stab_multiplier: float = 1.5
    crit_multiplier: float = 1.5
    crit_rate: float = 1 / 24
    crit_speed_based: bool = False  # rate = base speed * crit_speed_scale
    crit_speed_scale: float = 1 / 512
    # --- Feature toggles ----------------------------------------------------------------
    items: bool = True
    abilities: bool = True
    fields: bool = True
    team_preview: bool = True  # opponent species revealed at the start
    transform: Transform = TYPE_TRANSFORM
    # --- Field conditions ---------------------------------------------------------------
    field_multiplier: float = 1.5  # boosted type x m, weakened type / m
    field_chip: float = 1 / 16
    entry_field_turns: int = 5  # <= 0 means permanent
    move_field_turns: int = 5
    # --- Status archetype magnitudes (rows 1-5 of the status table) --------------------
    weaken_chip: float = 1 / 16  # 1: chip + attack multiplier
    weaken_stat_mult: float = 0.5
    chip_damage: float = 1 / 8  # 2: chip
    slow_speed_mult: float = 0.5  # 3: speed multiplier + skip chance
    slow_skip_chance: float = 0.25
    dormant_min: int = 1  # 4: skip for a sampled number of turns
    dormant_max: int = 3
    stun_recover_chance: float = 0.2  # 5: skip until a recovery roll succeeds
    deterministic_status_turns: int = 2  # blocking-status length when status rolls are off


GEN_TRANSFORMS = {6: STAT_TRANSFORM, 7: MOVE_TRANSFORM, 8: SIZE_TRANSFORM, 9: TYPE_TRANSFORM}


def gen_rules(gen: int) -> Rules:
    """Rule preset for generation analogue ``gen`` (1-9).

    1  one special stat, category-by-type, speed-based crits; no items,
       abilities or fields; harsh statuses (no recovery from stun)
    2  items, field conditions, special stat split
    3  abilities; ability-set fields are permanent
    4  per-move physical/special category
    5  team preview; shorter dormancy
    6  stat transform;  weaker crits; timed ability fields
    7  move transform;  rarer crits, softer weaken / slow
    8  size transform
    9  type transform
    """
    if not 1 <= gen <= 9:
        raise ValueError(f"gen must be in [1, 9], got {gen}")
    return Rules(
        gen=gen,
        special_stat_split=gen >= 2,
        physical_special_split=gen >= 4,
        crit_multiplier=2.0 if gen <= 5 else 1.5,
        crit_rate=1 / 16 if gen <= 6 else 1 / 24,
        crit_speed_based=gen == 1,
        items=gen >= 2,
        abilities=gen >= 3,
        fields=gen >= 2,
        entry_field_turns=0 if gen <= 5 else 5,
        team_preview=gen >= 5,
        transform=GEN_TRANSFORMS.get(gen, NO_TRANSFORM),
        weaken_chip=1 / 8 if gen <= 6 else 1 / 16,
        slow_speed_mult=0.25 if gen <= 6 else 0.5,
        dormant_max=7 if gen <= 4 else 3,
        stun_recover_chance=0.0 if gen == 1 else 0.2,
    )


GENERATIONS: dict[int, Rules] = {g: gen_rules(g) for g in range(1, 10)}


@dataclasses.dataclass(frozen=True)
class Chance:
    """Sources of stochasticity. Disabling one uses a deterministic fallback.

    damage_rolls  uniform 0.85-1.00 multiplier     (off: 0.925)
    crits         critical hits                    (off: never)
    misses        accuracy checks                  (off: always hit)
    secondary     secondary effect chance          (off: applies iff chance >= 0.5)
    status_rolls  skip / duration / recovery rolls (off: skip iff chance >= 0.5,
                                                    blocking statuses last
                                                    ``deterministic_status_turns``)
    speed_ties    coin-flip speed ties             (off: alternate by turn parity)
    """

    damage_rolls: bool = True
    crits: bool = True
    misses: bool = True
    secondary: bool = True
    status_rolls: bool = True
    speed_ties: bool = True

    @classmethod
    def deterministic(cls) -> Chance:
        return cls(False, False, False, False, False, False)

    @property
    def any(self) -> bool:
        return any(dataclasses.astuple(self))


@dataclasses.dataclass(frozen=True)
class EnvConfig:
    team_size: int = 3
    num_moves: int = 4
    max_turns: int = 100
    level: int = 50
    # Procedurally generated content sizes.
    num_species: int = 128
    num_dex_moves: int = 256
    num_statuses: int = 5
    num_fields: int = 4
    num_items: int = 8
    num_abilities: int = 8
    # "generational": random chart with generation-analogue structure, built on
    # ``base_types`` Gen-1 types (+2 from Gen 2, +1 from Gen 6).
    # "random": unstructured random chart with exactly ``base_types`` types.
    type_chart: Literal["generational", "random"] = "generational"
    base_types: int = 15
    # Imperfect information: opponent moves/items/abilities/alternate types
    # (and, without team preview, species) are hidden until revealed.
    hidden_info: bool = True
    # Outcome when ``max_turns`` is hit: "draw" or "hp" (higher total HP% wins).
    turn_limit: Literal["draw", "hp"] = "draw"
    rules: Rules = dataclasses.field(default_factory=Rules)
    chance: Chance = dataclasses.field(default_factory=Chance)

    def __post_init__(self):
        if self.team_size < 1 or self.num_moves < 1:
            raise ValueError("team_size and num_moves must be >= 1")
        if self.num_species < self.team_size:
            raise ValueError("num_species must be >= team_size")
        if self.num_dex_moves < self.num_moves:
            raise ValueError("num_dex_moves must be >= num_moves")
        if min(self.num_statuses, self.num_fields, self.num_items, self.num_abilities) < 1:
            raise ValueError("num_statuses/fields/items/abilities must be >= 1")
        if self.type_chart not in ("generational", "random"):
            raise ValueError(f"unknown type_chart {self.type_chart!r}")
        if self.type_chart == "generational" and self.base_types < 3:
            raise ValueError("generational type charts need base_types >= 3")
        if self.type_chart == "random" and self.base_types < 2:
            raise ValueError("random type charts need base_types >= 2")
        if self.turn_limit not in ("draw", "hp"):
            raise ValueError(f"unknown turn_limit {self.turn_limit!r}")

    @property
    def num_types(self) -> int:
        if self.type_chart == "random":
            return self.base_types
        from minimon.typechart import num_types_for

        return num_types_for(self.rules.gen, self.base_types)

    @property
    def num_actions(self) -> int:
        """``M`` moves + ``N`` switches + ``M`` transform-moves + 1 no-op."""
        return 2 * self.num_moves + self.team_size + 1

    def replace(self, **kw) -> EnvConfig:
        return dataclasses.replace(self, **kw)


DIFFICULTIES = ("trivial", "easy", "medium", "hard", "full")


def make_config(difficulty: str = "medium", gen: int = 9, **overrides) -> EnvConfig:
    """Build a config from a difficulty preset and generation analogue.

    trivial  1v1, 2 moves, 4 random types, deterministic, perfect information.
             A single repeated matrix game - good for solver sanity checks.
    easy     2v2, 2 moves, 6 random types, deterministic, perfect information.
    medium   3v3, 4 moves, generational chart, stochastic, hidden information.
    hard     6v6, 4 moves, generational chart, stochastic, hidden information.
    full     ``hard`` with a larger dex, more content and a longer turn limit.

    ``trivial``/``easy`` strip items, abilities, fields and the transform
    regardless of ``gen``; pass ``rules=...`` in ``overrides`` to change that.
    """
    rules = gen_rules(gen)
    simple = dataclasses.replace(
        rules, items=False, abilities=False, fields=False, transform=NO_TRANSFORM
    )
    presets = {
        "trivial": dict(
            team_size=1, num_moves=2, max_turns=30, num_species=16, num_dex_moves=32,
            type_chart="random", base_types=4, hidden_info=False,
            rules=simple, chance=Chance.deterministic(),
        ),
        "easy": dict(
            team_size=2, num_moves=2, max_turns=50, num_species=32, num_dex_moves=64,
            type_chart="random", base_types=6, hidden_info=False,
            rules=simple, chance=Chance.deterministic(),
        ),
        "medium": dict(team_size=3, num_moves=4, max_turns=100, rules=rules),
        "hard": dict(team_size=6, num_moves=4, max_turns=150, rules=rules),
        "full": dict(
            team_size=6, num_moves=4, max_turns=300, num_species=512, num_dex_moves=1024,
            num_statuses=8, num_fields=6, num_items=16, num_abilities=16, rules=rules,
        ),
    }
    if difficulty not in presets:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
    kwargs = {**presets[difficulty], **overrides}
    return EnvConfig(**kwargs)
