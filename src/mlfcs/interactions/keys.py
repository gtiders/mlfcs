"""Canonical identities for primitive real-space interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True, slots=True)
class InteractionKey:
    """One anchored primitive interaction with exact lattice translations."""

    sites: tuple[int, ...]
    translations: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.sites) < 2:
            raise ValueError("an interaction must contain at least two indices")
        if len(self.translations) != len(self.sites) - 1:
            raise ValueError("an anchored interaction requires one fewer translations than sites")
        if any(len(value) != 3 for value in self.translations):
            raise ValueError("interaction translations must be integer 3-vectors")

    @property
    def order(self) -> int:
        return len(self.sites)

    @property
    def labels(self) -> tuple[tuple[int, int, int, int], ...]:
        return (
            (self.sites[0], 0, 0, 0),
            *(
                (site, int(translation[0]), int(translation[1]), int(translation[2]))
                for site, translation in zip(self.sites[1:], self.translations, strict=True)
            ),
        )

    @classmethod
    def from_labels(cls, labels) -> InteractionKey:
        values = tuple(tuple(int(value) for value in label) for label in labels)
        if not values:
            raise ValueError("an interaction key cannot be empty")
        origin = np.asarray(values[0][1:], dtype=np.int64)
        translations = tuple(
            tuple(int(value) for value in np.asarray(label[1:], dtype=np.int64) - origin)
            for label in values[1:]
        )
        return cls(tuple(label[0] for label in values), translations)


class InteractionKeyCodec:
    """Fixed-width NumPy representation preserving InteractionKey ordering."""

    def __init__(self, order: int) -> None:
        if order < 2:
            raise ValueError("interaction order must be at least two")
        self.order = int(order)
        self.width = 4 * self.order
        self.canonical_columns = tuple(range(self.order)) + tuple(
            self.order + 3 * axis + component
            for axis in range(self.order - 1)
            for component in range(3)
        )

    def encode(self, key: InteractionKey) -> np.ndarray:
        if key.order != self.order:
            raise ValueError("key order does not match codec")
        return np.asarray(key.labels, dtype=np.int64).reshape(-1)

    def decode(self, row: np.ndarray) -> InteractionKey:
        values = np.asarray(row, dtype=np.int64).reshape(self.order, 4)
        return InteractionKey.from_labels(values)


__all__ = ["InteractionKey", "InteractionKeyCodec"]
