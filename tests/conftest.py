import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest

import minimon
from minimon.config import Chance, gen_rules, make_config
from minimon.constants import B_ATK, Category, Effect


def bare_rules(gen: int, **kw):
    """Generation rules with items/abilities/fields off (unless re-enabled)."""
    return dataclasses.replace(
        gen_rules(gen), **{"items": False, "abilities": False, "fields": False, **kw}
    )


def det_env(gen: int = 9, difficulty: str = "medium", rules=None, **kw) -> minimon.MiniMon:
    """Deterministic environment, optionally with custom rules."""
    cfg = make_config(difficulty, gen=gen, chance=Chance.deterministic(), **kw)
    if rules is not None:
        cfg = cfg.replace(rules=rules)
    return minimon.MiniMon(cfg)


@functools.cache
def jitted(env: minimon.MiniMon):
    return jax.jit(env.reset), jax.jit(env.step)


def set_move(params, idx, *, type=0, power=80.0, accuracy=1.0, category=Category.PHYSICAL,
             priority=0, effect=Effect.NONE, arg=0, mag=0.0, chance=0.0):
    """Overwrite dex move ``idx`` with exact properties."""
    upd = dict(move_type=type, move_power=power, move_accuracy=accuracy,
               move_category=category, move_priority=priority, move_effect=effect,
               move_effect_arg=arg, move_effect_mag=mag, move_effect_chance=chance)
    return params.replace(**{
        k: getattr(params, k).at[idx].set(jnp.asarray(v, getattr(params, k).dtype))
        for k, v in upd.items()
    })


def all_moves(state, player, move_idx):
    return state.replace(moves=state.moves.at[player].set(move_idx))


def neutral_chart(params):
    return params.replace(type_chart=jnp.ones_like(params.type_chart))


# Dex slots reserved for hand-built moves in tests.
ATTACK, SPECIAL_ATTACK, BOOST, INFLICT, SET_FIELD = 0, 1, 2, 3, 4


@pytest.fixture
def toy_params():
    """Build params for ``env`` with a neutral chart and known moves in fixed slots."""

    def build(env: minimon.MiniMon, key=0):
        p = neutral_chart(env.default_params(jax.random.key(key)))
        p = set_move(p, ATTACK, type=0, power=80.0)
        p = set_move(p, SPECIAL_ATTACK, type=1, power=80.0, category=Category.SPECIAL)
        p = set_move(p, BOOST, power=0.0, category=Category.STATUS,
                     effect=Effect.BOOST_SELF, arg=B_ATK, mag=2.0, chance=1.0)
        p = set_move(p, INFLICT, power=0.0, category=Category.STATUS,
                     effect=Effect.STATUS, arg=1, chance=1.0)
        p = set_move(p, SET_FIELD, power=0.0, category=Category.STATUS,
                     effect=Effect.FIELD, arg=1, chance=1.0)
        return p

    return build


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Drop compiled XLA programs after each module to bound per-worker memory."""
    yield
    jitted.cache_clear()
    jax.clear_caches()
