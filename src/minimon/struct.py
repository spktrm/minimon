"""Minimal frozen-dataclass pytree helper (no flax/chex dependency)."""

from __future__ import annotations

import dataclasses
from typing import TypeVar

import jax

T = TypeVar("T")


def static_field(**kwargs):
    """A dataclass field that is treated as pytree metadata (not traced)."""
    return dataclasses.field(metadata={"static": True}, **kwargs)


def dataclass(cls: type[T]) -> type[T]:
    """Frozen dataclass registered as a JAX pytree, with a ``.replace`` method."""
    cls = dataclasses.dataclass(frozen=True)(cls)
    fields = dataclasses.fields(cls)
    meta = [f.name for f in fields if f.metadata.get("static", False)]
    data = [f.name for f in fields if not f.metadata.get("static", False)]
    jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)

    def replace(self, **updates):
        return dataclasses.replace(self, **updates)

    cls.replace = replace
    return cls
