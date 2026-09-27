"""Procedurally generated content: type charts and mechanic tables."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from minimon import typechart
from minimon.config import gen_rules, make_config
from minimon.dex import (
    ABILITY_ARCHETYPES,
    ITEM_ARCHETYPES,
    make_fields,
    make_modifiers,
    make_params,
    make_statuses,
)

KEY = jax.random.key(0)
VALUES = {0.0, 0.5, 1.0, 2.0}


def test_random_chart_values():
    chart = np.asarray(typechart.random_chart(KEY, 7))
    assert chart.shape == (7, 7)
    assert set(np.unique(chart)) <= VALUES


def test_pad_chart_neutral_row_and_column():
    padded = np.asarray(typechart.pad_chart(typechart.random_chart(KEY, 5)))
    assert padded.shape == (6, 6)
    assert (padded[5] == 1).all() and (padded[:, 5] == 1).all()


@pytest.mark.parametrize("gen", range(1, 10))
def test_generational_chart_structure(gen):
    base = 10
    chart = np.asarray(typechart.generational_chart(KEY, gen, base))
    roles = typechart.chart_roles(gen, base)
    assert chart.shape == (typechart.num_types_for(gen, base),) * 2
    assert set(np.unique(chart)) <= VALUES
    dom, bug, dragon = roles["dominant"], roles["bugged"], roles["dragon"]
    # Dominant type has no weaknesses among the base types.
    assert (np.delete(chart[:base, dom], bug) <= 1.0).all()
    # The Gen-1 "bug": an intended super-effective hit is an immunity, fixed later.
    assert chart[bug, dom] == (0.0 if gen == 1 else 2.0)
    assert chart[dragon, dragon] == 2.0
    if gen >= 2:
        for c in roles["gen2_counters"]:
            assert chart[c, dom] == 2.0 and chart[dom, c] == 0.5
    if gen >= 6:
        f = roles["gen6_counter"]
        assert chart[dragon, f] == 0.0 and chart[f, dragon] == 2.0


def test_generational_chart_consistent_across_gens():
    base = 8
    g1 = np.asarray(typechart.generational_chart(KEY, 1, base))
    g9 = np.asarray(typechart.generational_chart(KEY, 9, base))
    bug, dom = typechart.chart_roles(1, base)["bugged"], typechart.chart_roles(1, base)["dominant"]
    diff = g1 != g9[:base, :base]
    assert diff.sum() == 1 and diff[bug, dom]


def test_status_table_archetypes_follow_rules():
    rules = gen_rules(1)
    t = make_statuses(KEY, rules, 7)
    assert t.chip.shape == (8,)
    assert t.chip[0] == 0 and t.skip[0] == 0 and (t.stat_mult[0] == 1).all()
    assert np.isclose(t.chip[1], rules.weaken_chip)
    assert np.isclose(t.chip[2], rules.chip_damage)
    assert np.isclose(t.stat_mult[3, 4], rules.slow_speed_mult)
    assert t.max_turns[4] == rules.dormant_max and t.skip[4] == 1
    assert t.recover[5] == rules.stun_recover_chance and t.skip[5] == 1


def test_field_table_mirror_pair():
    f = make_fields(KEY, gen_rules(9), 4, 6)
    tm = np.asarray(f.type_mult)
    assert tm.shape == (5, 7)
    assert (tm[0] == 1).all() and f.chip[0] == 0
    # Fields 1 and 2 boost / weaken the same two types in opposite directions.
    assert np.allclose(tm[1] * tm[2], 1.0)
    assert (tm[1] != 1).sum() == 2


@pytest.mark.parametrize("archetypes", [ITEM_ARCHETYPES, ABILITY_ARCHETYPES])
def test_modifier_tables(archetypes):
    n = len(archetypes) + 3
    t = make_modifiers(KEY, n, 4, archetypes)
    assert t.stat_mult.shape == (n + 1, 5)
    # Row 0 is neutral.
    assert (t.stat_mult[0] == 1).all() and t.damage_mult[0] == 1 and not t.lock[0]
    assert t.turn_heal[0] == 0 and (t.entry_foe_boost[0] == 0).all() and not t.endure[0]
    # Every non-empty row does *something*.
    active = ((t.stat_mult != 1).any(-1) | (t.damage_mult != 1) | (t.stab > 0) | t.lock
              | (t.turn_heal > 0) | (t.turn_boost != 0).any(-1)
              | (t.entry_foe_boost != 0).any(-1) | (t.entry_field > 0) | (t.switch_heal > 0)
              | t.endure | (t.low_hp_heal > 0))
    assert bool(active[1:].all())
    assert int(t.entry_field.max()) <= 4


def test_make_params_shapes_and_validity():
    cfg = make_config("medium", gen=9)
    p = make_params(cfg, KEY)
    T = cfg.num_types
    assert p.type_chart.shape == (T + 1, T + 1)
    assert p.species_base.shape == (cfg.num_species, 6)
    assert p.move_power.shape == (cfg.num_dex_moves,)
    assert bool(((p.move_accuracy > 0) & (p.move_accuracy <= 1)).all())
    assert int(p.species_types.max()) <= T
    assert int(p.species_ability.min()) >= 1
    assert int(p.species_ability.max()) <= cfg.num_abilities


def test_make_params_vmaps_to_distinct_metagames():
    cfg = make_config("medium", gen=9)
    batch = jax.vmap(lambda k: make_params(cfg, k))(jax.random.split(KEY, 4))
    assert batch.type_chart.shape[0] == 4
    assert not jnp.allclose(batch.species_base[0], batch.species_base[1])


def test_gen1_single_special_stat():
    p = make_params(make_config("medium", gen=1), KEY)
    assert np.allclose(p.species_base[:, 3], p.species_base[:, 4])
