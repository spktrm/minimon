"""Procedurally generated type-effectiveness charts.

Types are abstract integers; no real Pokemon chart is encoded. Charts are
``[T, T]`` float arrays indexed ``[attacking_type, defending_type]`` with
entries in {0, 0.5, 1, 2}. Environments pad them to ``[T + 1, T + 1]`` with a
neutral "no type" index ``T`` used for the empty second slot of mono-types.

``generational_chart`` layers generation *analogues* on top of a random base
chart, mirroring how the real chart evolved:

* Gen 1  ``base`` types. One type is *dominant* (no weaknesses), and one
         intended super-effective matchup against it is *bugged* into an
         immunity.
* Gen 2+ the bug is fixed and two *counter* types are added that hit the
         dominant type super-effectively and resist it.
* Gen 6+ one *counter* type is added against the strongest remaining
         offensive type (the "dragon" analogue): immune to it and super-effective
         on it.

``chart_roles(gen, base)`` reports which indices play these roles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def num_types_for(gen: int, base: int) -> int:
    """Total types in generation ``gen`` given ``base`` Gen-1 types."""
    return base + (2 if gen >= 2 else 0) + (1 if gen >= 6 else 0)


def chart_roles(gen: int, base: int) -> dict[str, int | tuple[int, ...]]:
    """Indices of the special types in ``generational_chart(key, gen, base)``."""
    if base < 3:
        raise ValueError("generational charts need at least 3 base types")
    roles: dict[str, int | tuple[int, ...]] = {
        "dominant": base - 1,  # e.g. Gen 1 Psychic
        "bugged": base - 2,  # e.g. Gen 1 Ghost -> Psychic
        "dragon": base - 3,  # e.g. Dragon, countered in Gen 6
    }
    if gen >= 2:
        roles["gen2_counters"] = (base, base + 1)  # e.g. Dark / Steel
    if gen >= 6:
        roles["gen6_counter"] = base + 2  # e.g. Fairy
    return roles


def random_chart(
    key: jax.Array,
    num_types: int,
    p_super: float = 0.18,
    p_resist: float = 0.18,
    p_immune: float = 0.03,
) -> jax.Array:
    """Unstructured ``[T, T]`` chart with i.i.d. entries."""
    p_neutral = 1.0 - p_super - p_resist - p_immune
    values = jnp.array([2.0, 0.5, 0.0, 1.0])
    probs = jnp.array([p_super, p_resist, p_immune, p_neutral])
    return jax.random.choice(key, values, shape=(num_types, num_types), p=probs)


def generational_chart(key: jax.Array, gen: int, base: int = 15, **kwargs) -> jax.Array:
    """Random chart with generation-analogue structure (see module docstring).

    The same ``key`` yields consistent charts across generations: the Gen N
    chart restricted to the base types equals the Gen 1 chart apart from the
    Gen-1 bug.
    """
    roles = chart_roles(gen, base)
    T = num_types_for(gen, base)
    full = random_chart(key, base + 3, **kwargs)  # sample all types once for consistency
    keep = jnp.arange(T) if gen >= 6 else jnp.arange(base + (2 if gen >= 2 else 0))
    chart = full[keep][:, keep]

    dom, bug, dragon = roles["dominant"], roles["bugged"], roles["dragon"]
    # Dominant type: no weaknesses among base types, hits the bugged type hard.
    col = chart[:base, dom]
    chart = chart.at[:base, dom].set(jnp.where(col > 1.0, 1.0, col))
    chart = chart.at[dom, bug].set(2.0)
    # The bugged type *should* be super-effective on the dominant type.
    chart = chart.at[bug, dom].set(0.0 if gen == 1 else 2.0)
    # "Dragon" analogue: strong offensively, super-effective on itself.
    chart = chart.at[dragon, dragon].set(2.0)
    if gen >= 2:
        for c in roles["gen2_counters"]:
            chart = chart.at[c, dom].set(2.0).at[dom, c].set(0.5)
    if gen >= 6:
        f = roles["gen6_counter"]
        chart = chart.at[dragon, f].set(0.0).at[f, dragon].set(2.0)
    return chart


def pad_chart(chart) -> jax.Array:
    """Append a neutral "no type" row/column at index ``T``."""
    chart = jnp.asarray(chart, dtype=jnp.float32)
    return jnp.pad(chart, ((0, 1), (0, 1)), constant_values=1.0)
