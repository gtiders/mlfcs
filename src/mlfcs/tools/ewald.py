"""Fixed-supercell dipole Ewald correction for explicit polar-force subtraction."""

from __future__ import annotations

from itertools import product
from math import pi, sqrt
from time import perf_counter
from typing import ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from scipy.special import erfc

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log_error import get_logger
from mlfcs.force_constants.export import translated_atoms
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)
_COULOMB = units.Hartree * units.Bohr  # eV angstrom for charges measured in e


def _lattice_points(
    basis: np.ndarray,
    shift: np.ndarray,
    radius: float,
    inverse: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate every ``shift + n @ basis`` inside a Euclidean ball."""
    if inverse is None:
        inverse = np.linalg.inv(basis)
    center = -shift @ inverse
    span = radius * np.linalg.norm(inverse, axis=0)
    lower = np.ceil(np.nextafter(center - span, -np.inf)).astype(np.int64)
    upper = np.floor(np.nextafter(center + span, np.inf)).astype(np.int64)
    indices = np.asarray(
        tuple(product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper)))),
        dtype=np.int64,
    ).reshape(-1, 3)
    vectors = shift + indices @ basis
    inside = np.einsum("ni,ni->n", vectors, vectors) <= radius * radius
    return indices[inside], vectors[inside]


def _pair_kernel(
    delta: np.ndarray,
    transformed_cell: np.ndarray,
    transformed_inverse: np.ndarray,
    geometry: PeriodicGeometry,
    inverse_sqrt_dielectric: np.ndarray,
    inverse_dielectric: np.ndarray,
    sqrt_dielectric: np.ndarray,
    reciprocal: np.ndarray,
    reciprocal_tensors: np.ndarray,
    eta: float,
    real_radius: float,
    determinant: float,
) -> np.ndarray:
    """Periodic screened dipole tensor before contraction with Born charges."""
    shifted, _ = geometry.minimum_image(delta @ inverse_sqrt_dielectric)
    _, transformed = _lattice_points(transformed_cell, shifted, real_radius, transformed_inverse)
    lengths = np.linalg.norm(transformed, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("different supercell sites occupy the same Cartesian position")
    vectors = transformed @ sqrt_dielectric
    gradient = vectors @ inverse_dielectric
    decaying = np.exp(-((eta * lengths) ** 2))
    complementary = erfc(eta * lengths)
    inverse = 1.0 / lengths
    a = complementary * inverse**3 + (2.0 * eta / sqrt(pi)) * decaying * inverse**2
    b = (
        3.0 * complementary * inverse**5
        + (6.0 * eta / sqrt(pi)) * decaying * inverse**4
        + (4.0 * eta**3 / sqrt(pi)) * decaying * inverse**2
    )
    real = (
        np.sum(a) * inverse_dielectric - np.einsum("n,ni,nj->ij", b, gradient, gradient)
    ) / sqrt(determinant)
    wave = np.einsum("n,nij->ij", np.cos(reciprocal @ delta), reciprocal_tensors)
    return _COULOMB * (real + wave)


class Ewald(Calculator):
    """Harmonic periodic dipole correction for one three-dimensional supercell.

    ``born[a, electric, displacement]`` is in elementary charges and
    ``dielectric`` is the dimensionless electronic dielectric tensor. The
    caller subtracts ``get_forces(atoms)`` from training forces explicitly.
    The ``fc2`` property is primitive-first, like ``ForceConstants.get(2, map)``.
    ``accuracy`` controls the two Ewald sums; ``eta`` changes their numerical
    splitting but not the physical result.
    No non-analytic LO-TO correction is added to exported phonopy files.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        space: ClusterSpace,
        mapping: ClusterMap,
        born: object,
        dielectric: object,
        *,
        accuracy: float = 1e-10,
        eta: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(space, ClusterSpace) or not isinstance(mapping, ClusterMap):
            raise TypeError("Ewald requires a ClusterSpace and ClusterMap")
        if space.fingerprint != mapping.space.fingerprint:
            raise ValueError("cluster space and cluster map use different primitive structures")
        charges = np.array(born, dtype=np.float64, copy=True)
        epsilon = np.array(dielectric, dtype=np.float64, copy=True)
        if charges.shape != (space.primitive.size, 3, 3) or not np.all(np.isfinite(charges)):
            raise ValueError("born must contain one finite 3-by-3 tensor per primitive atom")
        if epsilon.shape != (3, 3) or not np.all(np.isfinite(epsilon)):
            raise ValueError("dielectric must be a finite 3-by-3 tensor")
        dielectric_antisymmetry = float(np.max(np.abs(epsilon - epsilon.T)))
        epsilon = 0.5 * (epsilon + epsilon.T)
        eigvals, eigvecs = np.linalg.eigh(epsilon)
        if np.any(eigvals <= 0.0):
            raise ValueError("dielectric must be positive definite")
        accuracy = float(accuracy)
        if not np.isfinite(accuracy) or not 0.0 < accuracy < 1.0:
            raise ValueError("accuracy must lie strictly between zero and one")
        charges.setflags(write=False)
        epsilon.setflags(write=False)
        self.space = space
        self.mapping = mapping
        self.born = charges
        self.dielectric = epsilon
        self.accuracy = accuracy
        self._cell = np.array(mapping.supercell.cell, copy=True)
        self._numbers = np.array(mapping.supercell.numbers, copy=True)
        self._positions = np.asarray(mapping.supercell.scaled_positions) @ self._cell
        self._geometry = PeriodicGeometry(self._cell)
        self._inverse_dielectric = (eigvecs / eigvals) @ eigvecs.T
        self._sqrt_dielectric = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        self._inverse_sqrt_dielectric = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        self._determinant = float(np.prod(eigvals))
        transformed_cell = self._cell @ self._inverse_sqrt_dielectric
        self._transformed_cell = transformed_cell
        self._transformed_inverse = np.linalg.inv(transformed_cell)
        self._transformed_geometry = PeriodicGeometry(transformed_cell)
        volume = abs(float(np.linalg.det(self._cell)))
        transformed_volume = volume / sqrt(self._determinant)
        if eta is None:
            self._eta = sqrt(pi) / np.cbrt(transformed_volume)
        else:
            self._eta = float(eta)
            if not np.isfinite(self._eta) or self._eta <= 0.0:
                raise ValueError("eta must be a positive finite inverse length")
        decay = sqrt(-np.log(accuracy)) + 1.0
        self._real_radius = decay / self._eta
        reciprocal_basis = 2.0 * pi * np.linalg.inv(self._cell).T
        metric_basis = reciprocal_basis @ self._sqrt_dielectric
        indices, metric_vectors = _lattice_points(
            metric_basis, np.zeros(3), 2.0 * self._eta * decay
        )
        reciprocal = indices @ reciprocal_basis
        metric_squared = np.einsum("ni,ni->n", metric_vectors, metric_vectors)
        nonzero = metric_squared > 0.0
        self._reciprocal = reciprocal[nonzero]
        metric_squared = metric_squared[nonzero]
        weights = (
            (4.0 * pi / volume) * np.exp(-metric_squared / (4.0 * self._eta**2)) / metric_squared
        )
        self._reciprocal_tensors = np.einsum(
            "n,ni,nj->nij", weights, self._reciprocal, self._reciprocal
        )
        started = perf_counter()
        self._fc2 = self._build_fc2()
        self._fc2.setflags(write=False)
        self._full_fc2 = self._expand_fc2()
        self._full_fc2.setflags(write=False)
        logger.info(
            "Ewald ready: %d atoms, %d reciprocal vectors, accuracy %.3g, eta %.6g Å^-1, "
            "real radius %.6g Å, Born-charge sum %.3g e, "
            "dielectric antisymmetry %.3g, "
            "maximum ASR residual %.3g eV/Å², elapsed %.2f s",
            len(self._numbers),
            len(self._reciprocal),
            accuracy,
            self._eta,
            self._real_radius,
            float(np.max(np.abs(np.sum(self.born, axis=0)))),
            dielectric_antisymmetry,
            float(np.max(np.abs(np.sum(self._full_fc2, axis=1)))),
            perf_counter() - started,
        )

    def _build_fc2(self) -> np.ndarray:
        supercell = self.mapping.supercell
        size = len(self._numbers)
        first_atoms = np.asarray(
            [
                np.flatnonzero(supercell.sites == site)[0]
                for site in range(self.space.primitive.size)
            ]
        )
        result = np.empty((len(first_atoms), size, 3, 3), dtype=np.float64)
        for site, first in enumerate(first_atoms):
            for second in range(size):
                if first == second:
                    result[site, second] = 0.0
                    continue
                kernel = _pair_kernel(
                    self._positions[second] - self._positions[first],
                    self._transformed_cell,
                    self._transformed_inverse,
                    self._transformed_geometry,
                    self._inverse_sqrt_dielectric,
                    self._inverse_dielectric,
                    self._sqrt_dielectric,
                    self._reciprocal,
                    self._reciprocal_tensors,
                    self._eta,
                    self._real_radius,
                    self._determinant,
                )
                result[site, second] = (
                    self.born[site].T @ kernel @ self.born[int(supercell.sites[second])]
                )
            result[site, first] = -np.sum(result[site], axis=0)
        return result

    def _expand_fc2(self) -> np.ndarray:
        supercell = self.mapping.supercell
        size = len(self._numbers)
        result = np.empty((size, size, 3, 3), dtype=np.float64)
        for first in range(size):
            result[first] = self._fc2[
                int(supercell.sites[first]), translated_atoms(self.mapping, first)
            ]
        return result

    @property
    def fc2(self) -> np.ndarray:
        """A detached primitive-first FC2 correction in eV/Å²."""
        return self._fc2.copy()

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        if atoms is None or not isinstance(atoms, Atoms):
            raise TypeError("Ewald requires ASE Atoms")
        if not np.array_equal(atoms.numbers, self._numbers):
            raise ValueError("atoms have a different supercell atom sequence")
        cell = np.asarray(atoms.cell)
        if not np.all(np.isfinite(cell)):
            raise ValueError("Ewald requires a finite fixed supercell")
        cell_residual = float(np.max(np.linalg.norm(cell - self._cell, axis=1)))
        if cell_residual >= self.space.primitive.symprec:
            raise ValueError(
                f"Ewald requires the fixed supercell: cell residual {cell_residual:.10g} Å "
                f"is not below symprec {self.space.primitive.symprec:.10g} Å"
            )
        if not np.all(atoms.pbc):
            raise ValueError("Ewald requires three-dimensional periodic atoms")
        super().calculate(atoms, properties, system_changes)
        difference = atoms.positions - self._positions
        if not np.all(np.isfinite(difference)):
            raise ValueError("atomic positions contain NaN or infinite values")
        displacement, _ = self._geometry.minimum_image(difference)
        forces = -np.einsum("ijab,jb->ia", self._full_fc2, displacement)
        energy = -0.5 * float(np.sum(displacement * forces))
        self.results = {"energy": energy, "forces": forces}


__all__ = ["Ewald"]
