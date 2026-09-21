"""Public immutable interaction-space data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.interactions.keys import InteractionKey
from mlfcs.structure.lattice_frame import LatticeFrame


@dataclass(frozen=True, slots=True)
class PrimitiveOrbitImage:
    key: InteractionKey
    action: object


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionOrbit:
    """One symmetry-inequivalent primitive interaction cluster.

    The orbit carries two bases of the same invariant subspace, and their roles do not
    overlap:

    ``exact_lattice_basis``
        $B_{\\mathbb Z}$: integer columns in the reduced lattice (scaled) frame, one per
        fitted parameter.  The symmetry algebra, the rank certificate and the invariant
        kernel are decided here, exactly, and it is the provenance of a parameter set.
    ``cartesian_basis``
        $Q$: an orthonormal basis of the Cartesian subspace $C = K_n B_{\\mathbb Z}$,
        where $K_n = (A^\\mathsf{T})^{\\otimes n}$ is the fixed-order tensor map of the
        reduced algebra cell $A$.  Fitting, reconstruction, constraints and expansion
        consume this basis, and the fitted parameters are its coefficients.
    ``coefficient_transform``
        $R$ with $C = QR$, the change of coordinates between the fit parameters
        $\\theta$ and the exact lattice coefficients $c = R^{-1}\\theta$.

    ``observation_rows`` are component rows of $Q$ that a finite-difference plan has to
    observe; they are observations, not parameter pivots, and
    :func:`mlfcs.finite_difference.reconstruction.reconstruct_sparse` solves
    ``observation_matrix @ theta = observed`` explicitly instead of assuming that an
    observed component equals a parameter.
    """

    representative: InteractionKey
    exact_lattice_basis: np.ndarray
    cartesian_basis: np.ndarray
    coefficient_transform: np.ndarray
    observation_rows: np.ndarray
    observation_condition: float
    images: tuple[PrimitiveOrbitImage, ...]

    @property
    def dimension(self) -> int:
        return int(self.cartesian_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        """ndarray : the square system mapping parameters to observed components."""
        return self.cartesian_basis[self.observation_rows]


@dataclass(frozen=True, slots=True)
class PrimitiveInteractionSpace:
    primitive: Atoms
    order: int
    cutoff: float
    max_body_order: int | None
    frame: LatticeFrame
    orbits: tuple[PrimitiveInteractionOrbit, ...]

    @property
    def algebra_cell(self) -> np.ndarray:
        """ndarray : the reduced cell every orbit basis is expressed in."""
        return self.frame.algebra_cell

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
    cartesian_basis: np.ndarray
    observation_rows: np.ndarray
    images: tuple[RealizedOrbitImage, ...]

    @property
    def dimension(self) -> int:
        return int(self.cartesian_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        return self.cartesian_basis[self.observation_rows]


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
        """Return the components a finite-difference plan has to observe.

        One key is the representative cluster of an orbit with the observed components,
        so the plan measures exactly the rows ``observation_rows`` reconstructs from.
        """
        keys: set[tuple[tuple[int, int], ...]] = set()
        for orbit in self.orbits:
            for component in orbit.observation_rows:
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
