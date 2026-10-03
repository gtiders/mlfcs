"""Harmonic frequencies at requested q points in the positional Fourier gauge."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numba import njit
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs.core.log import get_logger
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.lattice import expand
from mlfcs.reciprocal.grid import QStars

_THZ = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (2.0 * np.pi * 1e12)
logger = get_logger(__name__)


@njit(cache=True, nogil=True)
def _dynamical(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    images: np.ndarray,
    tensors: np.ndarray,
    mass_weights: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Accumulate only the requested q points and enforce Hermitian roundoff."""
    matrices = np.zeros((len(points), 3 * n_sites, 3 * n_sites), dtype=np.complex128)
    for iq in range(len(points)):
        for term in range(len(first)):
            angle = 0.0
            for axis in range(3):
                angle += points[iq, axis] * images[term, axis]
            phase = 2.0 * np.pi * angle
            real = np.cos(phase) * mass_weights[term]
            imaginary = np.sin(phase) * mass_weights[term]
            row = 3 * first[term]
            column = 3 * second[term]
            for alpha in range(3):
                for beta in range(3):
                    value = tensors[term, alpha, beta]
                    matrices[iq, row + alpha, column + beta] += complex(
                        value * real, value * imaginary
                    )
        for row in range(3 * n_sites):
            for column in range(row, 3 * n_sites):
                value = 0.5 * (matrices[iq, row, column] + np.conj(matrices[iq, column, row]))
                matrices[iq, row, column] = value
                matrices[iq, column, row] = np.conj(value)
    return matrices


class Harmonic:
    """FC2 Fourier model using the primitive's per-site masses.

    ``frequencies(stars)`` evaluates only the irreducible representatives.
    Passing explicit q coordinates serves a band path and uses the same kernel;
    there is no separate full-grid dynamical-matrix workflow.
    """

    def __init__(self, model: ForceConstants) -> None:
        if not isinstance(model, ForceConstants) or 2 not in model.coefficients:
            raise ValueError("harmonic frequencies require ForceConstants containing FC2")
        primitive = model.cluster_space.primitive_atoms
        masses = primitive.get_masses()
        permutations = model.cluster_space.symmetry.site_permutations
        if not np.array_equal(masses[permutations], np.broadcast_to(masses, permutations.shape)):
            raise ValueError("primitive symmetry maps atoms with different masses")
        lattice = expand(model, 2)
        first = np.asarray([sites[0] for sites in lattice.sites], dtype=np.int32)
        second = np.asarray([sites[1] for sites in lattice.sites], dtype=np.int32)
        images = np.asarray(
            [
                np.asarray(translations[0])
                + primitive.get_scaled_positions()[sites[1]]
                - primitive.get_scaled_positions()[sites[0]]
                for sites, translations in zip(lattice.sites, lattice.translations, strict=True)
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        tensors = np.asarray(lattice.tensors, dtype=np.float64).reshape(-1, 3, 3)
        mass_weights = 1.0 / np.sqrt(masses[first] * masses[second])
        self.model = model
        self.masses = masses
        self._first = first
        self._second = second
        self._images = images
        self._tensors = tensors
        self._mass_weights = mass_weights

    def matrices(self, qpoints: object) -> np.ndarray:
        """Return dynamical matrices at requested points or only star representatives."""
        if isinstance(qpoints, QStars):
            symmetry = self.model.cluster_space.symmetry
            other = qpoints.symmetry
            if symmetry.symprec != other.symprec or any(
                not np.array_equal(getattr(symmetry, field), getattr(other, field))
                for field in (
                    "rotations",
                    "translations",
                    "cartesian_rotations",
                    "site_permutations",
                    "site_shifts",
                )
            ):
                raise ValueError("q stars use a different primitive symmetry from FC2")
            points = qpoints.points
            single = False
        else:
            values = np.asarray(qpoints, dtype=np.float64)
            single = values.shape == (3,)
            points = values.reshape(1, 3) if single else values
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("q points must be finite fractional coordinates of shape (n, 3)")
        matrices = _dynamical(
            np.ascontiguousarray(points),
            self._first,
            self._second,
            self._images,
            self._tensors,
            self._mass_weights,
            len(self.masses),
        )
        return matrices[0] if single else matrices

    def frequencies(self, qpoints: object) -> np.ndarray:
        """Return signed THz frequencies at q points or star representatives.

        Negative values represent imaginary modes. A ``QStars`` argument is
        reduced before any Fourier work; its star weights and expansion remain
        available on that object.
        """
        started = perf_counter()
        matrices = self.matrices(qpoints)
        eigenvalues = np.linalg.eigvalsh(matrices)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _THZ
        n_qpoints = len(frequencies) if frequencies.ndim == 2 else 1
        logger.info(
            "Harmonic frequencies: qpoints=%d modes_per_q=%d imaginary_modes=%d "
            "range=[%.8g, %.8g] THz elapsed=%.2f s",
            n_qpoints,
            frequencies.shape[-1],
            int(np.count_nonzero(frequencies < 0.0)),
            float(np.min(frequencies)),
            float(np.max(frequencies)),
            perf_counter() - started,
        )
        return frequencies


__all__ = ["Harmonic"]
