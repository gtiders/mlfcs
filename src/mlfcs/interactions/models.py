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
    """One symmetry-inequivalent primitive interaction cluster.

    ``basis`` is the exact integer basis of the invariant subspace in lattice (scaled)
    coordinates, one column per fitted parameter, and ``pivots`` the component rows whose
    integer block accepts those parameter values.  Lattice coordinates keep the symmetry
    algebra exact for every cell; consumers map to Cartesian components through the cell
    of the space when they need physical tensors.
    """

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
    def cell(self) -> np.ndarray:
        """ndarray : primitive cell matrix, the lattice frame of every orbit basis."""
        return np.asarray(self.primitive.cell.array, dtype=float)

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
    """One primitive orbit space realized in a finite reference.

    Orbits keep their lattice-frame integer basis, and ``cell`` is the primitive cell
    they are expressed in, so consumers render Cartesian components through
    ``scaled_to_cartesian_matrix(cell, order)``.
    """

    order: int
    orbits: tuple[RealizedInteractionOrbit, ...]
    cell: np.ndarray
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
                keys.add(
                    tuple(
                        (orbit.representative[i], int(directions[i])) for i in range(self.order - 1)
                    )
                )
        return tuple(sorted(keys))


__all__ = [
    "PrimitiveInteractionOrbit",
    "PrimitiveInteractionSpace",
    "PrimitiveOrbitImage",
    "RealizedInteractionOrbit",
    "RealizedInteractionSpace",
    "RealizedOrbitImage",
]
