"""Generation-analogue mechanics: transforms, items, abilities, category rules."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import ATTACK, BOOST, all_moves, bare_rules, det_env, set_move
from test_env import fresh, step

import minimon
from minimon import mechanics as mech
from minimon.config import gen_rules
from minimon.constants import HP, Category


@pytest.mark.parametrize("gen", range(1, 10))
def test_transform_actions_legal_only_from_gen6(gen, toy_params):
    env = det_env(gen=gen, rules=bare_rules(gen))
    params = toy_params(env)
    M, N = env.config.num_moves, env.config.team_size
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    legal = env.legal_actions(state, params)
    assert bool(legal[:, M + N:2 * M + N].any()) == (gen >= 6)
    if gen >= 6:
        s = step(env, state, [M + N, 0], params).state
        assert bool(s.gimmick_used[0]) and not bool(s.gimmick_used[1])
        legal = env.legal_actions(s, params)
        assert not bool(legal[0, M + N:2 * M + N].any())
        assert bool(legal[1, M + N:2 * M + N].all())


def _transform_env(gen, toy_params):
    env = det_env(gen=gen, rules=bare_rules(gen))
    params = toy_params(env)
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    return env, params, state, env.config.num_moves + env.config.team_size


def test_type_transform_changes_defensive_type(toy_params):
    env, params, state, G = _transform_env(9, toy_params)
    s = step(env, state, [G, 0], params).state
    assert bool(s.transformed[0, 0])
    types = mech.defensive_types(env.config, params, s, 0, 0)
    assert types.tolist() == [int(s.alt_type[0, 0]), params.num_types]


def test_stat_transform_is_permanent(toy_params):
    env, params, state, G = _transform_env(6, toy_params)
    M = env.config.num_moves
    s = step(env, state, [G, 0], params).state
    mult = np.asarray(gen_rules(6).transform.stat_mult)
    assert np.allclose(s.stats[0, 0, 1:], state.stats[0, 0, 1:] * mult)
    s = step(env, s, [M + 1, 0], params).state  # switch out and back
    s = step(env, s, [M + 0, 0], params).state
    assert bool(s.transformed[0, 0])


def test_size_transform_guards_and_expires(toy_params):
    env, params, state, G = _transform_env(8, toy_params)
    tf = gen_rules(8).transform
    state = all_moves(state, 1, ATTACK)
    s = step(env, state, [G, 0], params).state  # BOOST becomes a priority guard
    assert float(s.hp[0, 0]) == float(s.stats[0, 0, HP])  # guard blocked the attack
    assert float(s.stats[0, 0, HP]) == np.floor(float(state.stats[0, 0, HP]) * tf.hp_mult)
    assert int(s.transform_turns[0]) == tf.duration - 1
    s = all_moves(s, 1, BOOST)
    for _ in range(tf.duration - 1):
        s = step(env, s, [0, 0], params).state
    assert not bool(s.transformed[0, 0]) and int(s.transform_turns[0]) == 0
    assert float(s.stats[0, 0, HP]) == float(state.stats[0, 0, HP])


def test_size_transform_ends_on_switch(toy_params):
    env, params, state, G = _transform_env(8, toy_params)
    M = env.config.num_moves
    s = step(env, state, [G, 0], params).state
    s = step(env, s, [M + 1, 0], params).state
    assert not bool(s.transformed[0, 0])
    assert float(s.stats[0, 0, HP]) == float(state.stats[0, 0, HP])


def test_move_transform_boosts_one_move(toy_params):
    env, params, state, G = _transform_env(7, toy_params)
    state = all_moves(state, 0, ATTACK)
    normal = step(env, state, [0, 0], params).state
    burst = step(env, state, [G, 0], params).state
    assert float(burst.hp[1, 0]) < float(normal.hp[1, 0])
    assert bool(burst.gimmick_used[0]) and not bool(burst.transformed[0, 0])


@pytest.mark.parametrize("gen", [1, 2, 3])
def test_early_gens_lack_items_and_abilities(gen):
    env = minimon.make("medium", gen=gen)
    state = fresh(env, env.default_params())
    assert bool((state.item > 0).any()) == (gen >= 2)
    assert bool((state.ability > 0).any()) == (gen >= 3)


def test_category_by_type_before_split(toy_params):
    for gen, split in ((3, False), (4, True)):
        env = det_env(gen=gen, rules=bare_rules(gen))
        params = toy_params(env)
        params = params.replace(special_types=params.special_types.at[0].set(True))
        params = set_move(params, ATTACK, type=0, category=Category.PHYSICAL)
        cat = int(mech.move_category(env.config, params, ATTACK))
        assert cat == (Category.PHYSICAL if split else Category.SPECIAL)


def test_team_preview(toy_params):
    for gen, preview in ((4, False), (5, True)):
        env = det_env(gen=gen, rules=bare_rules(gen))
        state = fresh(env, toy_params(env))
        assert bool(state.revealed_species.all()) == preview
        assert bool(state.revealed_species[:, 0].all())


def _row_with(table, predicate):
    rows = np.nonzero(np.asarray(predicate(table)))[0]
    rows = rows[rows > 0]
    assert len(rows), "generated table lacks the archetype"
    return int(rows[0])


def test_entry_field_ability(toy_params):
    env = det_env(gen=6, rules=bare_rules(6, abilities=True, fields=True))
    params = toy_params(env)
    M = env.config.num_moves
    row = _row_with(params.abilities, lambda t: t.entry_field > 0)
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    state = state.replace(ability=state.ability.at[1, 1].set(row),
                          revealed_ability=jnp.zeros_like(state.revealed_ability))
    s = step(env, state, [0, M + 1], params).state
    assert int(s.field) == int(params.abilities.entry_field[row])
    assert bool(s.revealed_ability[1, 1])


def test_regen_item_heals_and_reveals(toy_params):
    env = det_env(gen=2, rules=bare_rules(2, items=True))
    params = toy_params(env)
    row = _row_with(params.items, lambda t: t.turn_heal > 0)
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    state = state.replace(item=state.item.at[0, 0].set(row),
                          hp=state.hp.at[0, 0].multiply(0.5))
    s = step(env, state, [0, 0], params).state
    assert float(s.hp[0, 0]) > float(state.hp[0, 0])
    assert bool(s.revealed_item[0, 0]) and not bool(s.revealed_item[1, 0])


def test_endure_item_is_consumed(toy_params):
    env = det_env(gen=2, rules=bare_rules(2, items=True))
    params = set_move(toy_params(env), ATTACK, power=1000.0)
    row = _row_with(params.items, lambda t: t.endure)
    state = all_moves(all_moves(fresh(env, params), 0, ATTACK), 1, BOOST)
    state = state.replace(item=state.item.at[1, 0].set(row))
    s = step(env, state, [0, 0], params).state
    assert float(s.hp[1, 0]) == 1.0
    assert int(s.item[1, 0]) == 0 and bool(s.revealed_item[1, 0])


def test_status_table_drives_speed(toy_params):
    env = det_env(rules=bare_rules(9))
    params = toy_params(env)
    state = fresh(env, params)
    slow_row = int(np.argmin(np.asarray(params.statuses.stat_mult[:, 4])))
    before = float(mech.effective_speed(env.config, params, state, 0))
    state = state.replace(status=state.status.at[0, 0].set(slow_row))
    after = float(mech.effective_speed(env.config, params, state, 0))
    assert after == pytest.approx(before * float(params.statuses.stat_mult[slow_row, 4]))
    assert after < before


def test_custom_transform(toy_params):
    """Transforms are a parameter space, not a fixed list."""
    import dataclasses

    from minimon.config import Transform

    tf = Transform("custom", True, stat_mult=(1.0, 1.0, 1.0, 1.0, 2.0), retype=True,
                   hp_mult=1.5, duration=2)
    env = det_env(rules=dataclasses.replace(bare_rules(9), transform=tf))
    params = toy_params(env)
    G = env.config.num_moves + env.config.team_size
    state = all_moves(all_moves(fresh(env, params), 0, BOOST), 1, BOOST)
    s = step(env, state, [G, 0], params).state
    assert float(s.stats[0, 0, 5]) == pytest.approx(2 * float(state.stats[0, 0, 5]))
    assert bool(s.transformed[0, 0]) and int(s.transform_turns[0]) == 1
