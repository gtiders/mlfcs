"""Shared FC2 Fourier representation and dynamical-matrix operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs.force_constants.expansion import LatticeForceConstants, expand_lattice_tensors
from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.arrays import readonly
from mlfcs.foundation.tensors import rotate_basis

# Convert sqrt(eV / (angstrom**2 * amu)) to ordinary frequency in THz.
THZ_PER_SQRT_EV_PER_A2_AMU = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (
    2.0 * np.pi * 1e12
)


@dataclass(frozen=True, slots=True)
class DynamicalTerms:
    """Geometry and mass data for the Fourier expansion of primitive FC2 terms.

    Each entry represents an expanded pair interaction between primitive
    motif sites ``i`` and ``j`` with fractional separation

        d = n + s_j - s_i,

    where ``n`` is the lattice translation of the second site relative to the
    first. Its dynamical-matrix contribution carries phase
    ``exp(2*pi*i*q·d)`` and mass factor ``1/sqrt(m_i*m_j)``. The separation
    follows the positional Fourier gauge. ``orbit_indices`` and
    ``image_indices`` identify the symmetry orbit and image that produced each
    term; the term order matches the expanded lattice FC2 tensors.
    """

    first_sites: np.ndarray
    second_sites: np.ndarray
    fractional_separations: np.ndarray
    mass_weights: np.ndarray
    orbit_indices: np.ndarray
    image_indices: np.ndarray


def prepare_dynamical_terms(model: ForceConstants) -> tuple[DynamicalTerms, LatticeForceConstants]:
    """Expand FC2 interactions and prepare their Fourier geometry.

    For each expanded interaction ``(i, j, n)``, record the fractional
    separation ``n + s_j - s_i``, the factor ``1/sqrt(m_i*m_j)``, and its
    originating orbit and image. The geometry aligns one-to-one with the
    returned lattice FC2 tensors, which remain Cartesian in eV/angstrom**2.
    Raise ValueError if FC2 is absent and RuntimeError if expansion provenance
    is inconsistent.
    """
    lattice = expand_lattice_tensors(model, 2)
    block = model.cluster_space.block(2)
    orbit_indices = []
    image_indices = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        for image_index, _cluster in enumerate(model.cluster_space.orbits[orbit_index].clusters):
            orbit_indices.append(orbit_index)
            image_indices.append(image_index)
    if len(orbit_indices) != len(lattice.sites):
        raise RuntimeError("FC2 expansion order differs from orbit/image provenance")
    first = np.asarray([sites[0] for sites in lattice.sites], dtype=np.int32)
    second = np.asarray([sites[1] for sites in lattice.sites], dtype=np.int32)
    separations = np.asarray(
        [
            np.asarray(translations[0])
            + model.cluster_space.scaled_positions[sites[1]]
            - model.cluster_space.scaled_positions[sites[0]]
            for sites, translations in zip(lattice.sites, lattice.translations, strict=True)
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    weights = 1.0 / np.sqrt(model.cluster_space.masses[first] * model.cluster_space.masses[second])
    terms = DynamicalTerms(
        readonly(first, np.int64),
        readonly(second, np.int64),
        readonly(separations, np.float64),
        readonly(weights, np.float64),
        readonly(orbit_indices, np.int64),
        readonly(image_indices, np.int64),
    )
    return terms, lattice


def prepare_fc2_bases(
    cluster_space, terms: DynamicalTerms
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand primitive FC2 orbit bases into Fourier-term image order.

    Each symmetry image receives the Cartesian basis obtained by rotating and
    permuting its primitive orbit basis. For each term, ``offsets`` and
    ``dimensions`` select the corresponding coefficient slice
    ``coefficients[offset:offset + dimension]`` in the global FC2 parameter
    vector. ``terms`` must have been prepared from this cluster space.
    """
    block = cluster_space.block(2)
    maximum_dimension = max(
        cluster_space.orbits[index].dimension
        for index in range(block.orbits.start, block.orbits.stop)
    )
    bases = np.zeros((len(terms.orbit_indices), 9, maximum_dimension), dtype=np.float64)
    offsets = np.empty(len(terms.orbit_indices), dtype=np.int32)
    dimensions = np.empty(len(terms.orbit_indices), dtype=np.int32)
    orbit_starts = {}
    parameter_offset = 0
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit_starts[orbit_index] = parameter_offset
        parameter_offset += cluster_space.orbits[orbit_index].dimension
    for term, (orbit_index, image_index) in enumerate(
        zip(terms.orbit_indices, terms.image_indices, strict=True)
    ):
        orbit = cluster_space.orbits[int(orbit_index)]
        rotation = cluster_space.symmetry.cartesian_rotations[orbit.operations[int(image_index)]].T
        basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[int(image_index)])
        bases[term, :, : orbit.dimension] = basis
        offsets[term] = orbit_starts[int(orbit_index)]
        dimensions[term] = orbit.dimension
    return bases, offsets, dimensions


def translation_complement(masses: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis excluding the three acoustic translations.

    In mass-weighted Cartesian coordinates, rigid translations span vectors
    proportional to ``sqrt(m_i)`` along x, y, and z. Return an orthonormal
    complement with shape ``(3*N, 3*(N-1))`` for positive per-site masses in
    atomic mass units. A single-site primitive cell has an empty complement.
    """
    count = len(masses)
    if count == 1:
        return np.empty((3, 0))
    weights = np.sqrt(masses)
    weights /= np.linalg.norm(weights)
    direction = weights.copy()
    direction[-1] -= 1.0
    direction /= np.linalg.norm(direction)
    householder = np.eye(count) - 2.0 * np.outer(direction, direction)
    basis = np.zeros((3 * count, 3 * (count - 1)))
    for axis in range(3):
        basis[axis::3, axis::3] = householder[:, :-1]
    return basis


@njit(cache=True, nogil=True)
def _accumulate_dynamical_matrices(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    separations: np.ndarray,
    tensors: np.ndarray,
    mass_weights: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Fourier transform expanded FC2 tensors into dynamical matrices.

    For each q point, accumulate

        D[i,alpha,j,beta](q) = sum_R Phi[i,alpha,j,beta](R)
            * exp(2*pi*i*q·(R + s_j - s_i)) / sqrt(m_i*m_j).

    ``points`` are fractional reciprocal coordinates and ``separations``
    stores the corresponding positional-gauge vectors ``R + s_j - s_i``.
    The result is mass weighted and retains the native FC2 units before
    frequency conversion. Hermitian symmetrization removes numerical roundoff;
    it does not impose a physical correction on the force constants.
    """
    matrices = np.zeros((len(points), 3 * n_sites, 3 * n_sites), dtype=np.complex128)
    for iq in range(len(points)):
        for term in range(len(first)):
            angle = 0.0
            for axis in range(3):
                angle += points[iq, axis] * separations[term, axis]
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


def accumulate_dynamical_matrices(points, terms, tensors, n_sites):
    """Evaluate mass-weighted dynamical matrices at fractional q points.

    ``tensors`` must align with ``terms`` and contain Cartesian FC2 tensors in
    eV/angstrom**2. The positional Fourier convention
    ``exp(2*pi*i*q·d)`` uses the pair separations stored in ``terms``. Return
    complex dynamical matrices in the native FC2 unit divided by atomic mass.
    """
    return _accumulate_dynamical_matrices(
        points,
        terms.first_sites,
        terms.second_sites,
        terms.fractional_separations,
        tensors,
        terms.mass_weights,
        n_sites,
    )


__all__ = [
    "THZ_PER_SQRT_EV_PER_A2_AMU",
    "DynamicalTerms",
    "accumulate_dynamical_matrices",
    "prepare_dynamical_terms",
    "prepare_fc2_bases",
    "translation_complement",
]
