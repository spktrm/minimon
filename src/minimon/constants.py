"""Integer constants used inside traced code.

Only the *structural* vocabulary lives here (stats, move categories, effect
kinds, phases). Concrete statuses, field conditions, items and abilities are
rows of procedurally generated tables in ``EnvParams``; index ``0`` of each
table is always "none".
"""

# --- Stats -----------------------------------------------------------------
# Index into ``stats[..., 6]``.
HP, ATK, DEF, SPA, SPD, SPE = range(6)
NUM_STATS = 6

# Index into ``boosts[..., 5]`` and every per-stat multiplier vector (stat - 1).
B_ATK, B_DEF, B_SPA, B_SPD, B_SPE = range(5)
NUM_BOOSTS = 5
MAX_BOOST = 6

NONE = 0  # row 0 of the status / field / item / ability tables


class Category:
    PHYSICAL, SPECIAL, STATUS = range(3)
    NUM = 3


class Effect:
    """Move effect kinds. ``effect_arg`` / ``effect_mag`` parameterise each one.

    NONE        -
    STATUS      arg = status-table row inflicted on the target
    BOOST_SELF  arg = boost index, mag = stages
    DROP_TARGET arg = boost index, mag = stages
    HEAL        mag = fraction of user's max HP restored
    DRAIN       mag = fraction of damage dealt restored
    RECOIL      mag = fraction of damage dealt taken as recoil
    FIELD       arg = field-table row to set
    """

    NONE, STATUS, BOOST_SELF, DROP_TARGET, HEAL, DRAIN, RECOIL, FIELD = range(8)
    NUM = 8


class Phase:
    """TURN: both players act. SWITCH: only players whose active fainted act."""

    TURN, SWITCH = range(2)
    NUM = 2
