"""Data containers for primitive and finite interaction orbits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.tensors import TensorAction
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


@dataclass(frozen=True, slots=True)
class OrbitImage:
    cluster: tuple[int, ...]
    action: TensorAction


@dataclass(frozen=True, slots=True)
class ClusterOrbit:
    representative: tuple[int, ...]
    basis: np.ndarray
    pivots: np.ndarray
    images: tuple[OrbitImage, ...]

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True, slots=True)
class OrbitSpace:
    order: int
    orbits: tuple[ClusterOrbit, ...]
    cutoff: float
    max_body_order: int | None = None

    @property
    def displacement_keys(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for flat_component in orbit.pivots:
                components = np.unravel_index(int(flat_component), (3,) * self.order)
                key = tuple(
                    (orbit.representative[axis], int(components[axis]))
                    for axis in range(self.order - 1)
                )
                keys.add(key)
        return tuple(sorted(keys))


@dataclass(frozen=True, slots=True)
class PrimitiveOrbitImage:
    key: InteractionKey
    action: TensorAction


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
    symmetry: PrimitiveSymmetryOperations
    orbits: tuple[PrimitiveInteractionOrbit, ...]
