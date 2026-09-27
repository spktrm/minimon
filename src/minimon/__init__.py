"""MiniMon: Pokemon Showdown distilled into a jittable zero-sum matrix game."""

from minimon import constants, matrix, policies, typechart
from minimon.config import (
    DIFFICULTIES,
    GEN_TRANSFORMS,
    GENERATIONS,
    NO_TRANSFORM,
    Chance,
    EnvConfig,
    Rules,
    Transform,
    gen_rules,
    make_config,
)
from minimon.constants import Phase
from minimon.dex import EnvParams, make_params
from minimon.env import MiniMon, StepOutput
from minimon.rollout import Trajectory, evaluate, masked_categorical, play_episode, rollout
from minimon.state import State

__version__ = "0.1.0"


def make(difficulty: str = "medium", gen: int = 9, **overrides) -> MiniMon:
    """Shorthand for ``MiniMon(make_config(difficulty, gen, **overrides))``."""
    return MiniMon(make_config(difficulty, gen, **overrides))


__all__ = [
    "DIFFICULTIES",
    "GENERATIONS",
    "Chance",
    "EnvConfig",
    "EnvParams",
    "GEN_TRANSFORMS",
    "NO_TRANSFORM",
    "Transform",
    "MiniMon",
    "Phase",
    "Rules",
    "State",
    "StepOutput",
    "Trajectory",
    "constants",
    "evaluate",
    "gen_rules",
    "make",
    "make_config",
    "make_params",
    "masked_categorical",
    "matrix",
    "play_episode",
    "policies",
    "rollout",
    "typechart",
]
