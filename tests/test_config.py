import dataclasses

import pytest

import minimon
from minimon.config import (
    GEN_TRANSFORMS,
    NO_TRANSFORM,
    Chance,
    EnvConfig,
    gen_rules,
    make_config,
)


@pytest.mark.parametrize("gen", range(1, 10))
def test_gen_rules_progression(gen):
    r = gen_rules(gen)
    assert r.gen == gen
    assert r.special_stat_split == (gen >= 2)
    assert r.physical_special_split == (gen >= 4)
    assert r.items == (gen >= 2)
    assert r.abilities == (gen >= 3)
    assert r.fields == (gen >= 2)
    assert r.team_preview == (gen >= 5)
    assert r.transform == GEN_TRANSFORMS.get(gen, NO_TRANSFORM)
    assert r.crit_speed_based == (gen == 1)


def test_transform_presets_explore_different_axes():
    by_gen = {g: gen_rules(g).transform for g in (6, 7, 8, 9)}
    assert by_gen[6].stat_mult != (1.0,) * 5 and not by_gen[6].timed
    assert by_gen[7].single_move and by_gen[7].changes_power
    assert by_gen[8].timed and by_gen[8].hp_mult > 1 and by_gen[8].status_guard
    assert by_gen[9].retype
    assert not NO_TRANSFORM.enabled


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gen_rules(0)
    with pytest.raises(ValueError):
        gen_rules(10)
    with pytest.raises(ValueError):
        make_config("impossible")
    with pytest.raises(ValueError):
        EnvConfig(team_size=0)
    with pytest.raises(ValueError):
        EnvConfig(type_chart="canonical")
    with pytest.raises(ValueError):
        EnvConfig(num_species=2, team_size=3)


@pytest.mark.parametrize("difficulty", minimon.DIFFICULTIES)
def test_difficulty_presets(difficulty):
    cfg = make_config(difficulty, gen=9)
    assert cfg.num_actions == 2 * cfg.num_moves + cfg.team_size + 1
    hash(cfg)  # must be usable as a static jit argument


def test_simple_presets_strip_features():
    for d in ("trivial", "easy"):
        cfg = make_config(d, gen=9)
        assert not cfg.hidden_info
        assert not cfg.chance.any
        assert not (cfg.rules.items or cfg.rules.abilities or cfg.rules.fields)
        assert not cfg.rules.transform.enabled


def test_num_types_by_generation():
    assert make_config("medium", gen=1).num_types == 15
    assert make_config("medium", gen=2).num_types == 17
    assert make_config("medium", gen=6).num_types == 18
    assert make_config("medium", gen=6, base_types=8).num_types == 11
    assert make_config("trivial").num_types == 4


def test_chance_deterministic():
    assert not Chance.deterministic().any
    assert Chance().any
    assert dataclasses.replace(Chance.deterministic(), crits=True).any


def test_version_matches_metadata():
    from importlib.metadata import version

    assert minimon.__version__ == version("minimon-jax")
