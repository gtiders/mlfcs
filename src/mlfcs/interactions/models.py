"""Public immutable interaction-space data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.interactions.keys import InteractionKey


@dataclass(frozen=True, slots=True)
class PrimitiveOrbitImage:
    key: InteractionKey
    action: object


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionOrbit:
    representative: InteractionKey
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[PrimitiveOrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionSpace:
    primitive: Atoms
    order: int
    cutoff: float
    max_body_order: int | None
    symmetry: object
    orbits: tuple[PrimitiveInteractionOrbit, ...]

    @property
    def n_parameters(self) -> int:
        return sum(orbit.dimension for orbit in self.orbits)


@dataclass(frozen=True, slots=True)
class RealizedOrbitImage:
    cluster: tuple[int, ...]
    action: object


@dataclass(frozen=True, slots=True)
class RealizedInteractionOrbit:
    representative: tuple[int, ...]
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[RealizedOrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class RealizedInteractionSpace:
    order: int
    orbits: tuple[RealizedInteractionOrbit, ...]
    cutoff: float
    max_body_order: int | None = None

    @property
    def n_parameters(self) -> int:
        return sum(orbit.dimension for orbit in self.orbits)

    @property
    def displacement_keys(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for component in orbit.pivots:
                directions = np.unravel_index(int(component), (3,) * self.order)
                keys.add(tuple((orbit.representative[i], int(directions[i])) for i in range(self.order - 1)))
        return tuple(sorted(keys))


__all__ = [
    "PrimitiveInteractionOrbit",
    "PrimitiveInteractionSpace",
    "PrimitiveOrbitImage",
    "RealizedInteractionOrbit",
    "RealizedInteractionSpace",
    "RealizedOrbitImage",
]
