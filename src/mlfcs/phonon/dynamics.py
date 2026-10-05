"""Shared FC2 Fourier terms and dynamical-matrix operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs._arrays import readonly
from mlfcs.force_constants.expansion import LatticeForceConstants, expand_lattice_tensors
from mlfcs.force_constants.model import ForceConstants
from mlfcs.tensors import rotate_basis

THZ_PER_SQRT_EV_PER_A2_AMU = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (
    2.0 * np.pi * 1e12
)


@dataclass(frozen=True, slots=True)
class DynamicalTerms:
    """Readonly FC2 Fourier metadata in orbit-major, image-major expansion order.

    ``first_sites`` and ``second_sites`` are int64 arrays of shape (n_terms,)
    containing primitive motif indices. ``fractional_separations`` is float64
    shape (n_terms, 3), equal to the last-slot integer translation plus the
    second minus first wrapped primitive fractional position. ``mass_weights``
    is float64 shape (n_terms,), equal to 1/sqrt(m_i*m_j) in inverse square-root
    atomic mass units. ``orbit_indices`` and ``image_indices`` are int64
    provenance arrays of shape (n_terms,). All arrays are readonly; entries
    align one-for-one with the lattice tensor expansion.
    """

    first_sites: np.ndarray
    second_sites: np.ndarray
    fractional_separations: np.ndarray
    mass_weights: np.ndarray
    orbit_indices: np.ndarray
    image_indices: np.ndarray


def prepare_dynamical_terms(model: ForceConstants) -> tuple[DynamicalTerms, LatticeForceConstants]:
    """Prepare FC2 Fourier metadata with explicit orbit/image provenance.

    Expansion order is block orbit order followed by each orbit's stored image
    order. Separations are fractional rows ``shift + r_j - r_i``; returned
    tensors remain Cartesian in eV/angstrom**2. Raises ValueError if FC2 is
    absent and RuntimeError if provenance and expansion lengths disagree.
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
    """Build term-aligned rotated FC2 bases, parameter offsets, and dimensions.

    ``terms`` must come from ``prepare_dynamical_terms`` for this cluster space.
    Return float64 bases (n_terms, 9, max_dimension) and int32 offset/dimension
    vectors. Every basis is placed by the provenance arrays, preserving expansion
    order and parameter offsets.
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
    """Return a (3*N, 3*(N-1)) orthonormal complement of mass-weighted translations.

    masses is a positive per-site mass vector in atomic mass units. Atom-major
    Cartesian rows exclude the three Gamma translation directions. For one
    site return (3, 0); input masses are not modified.
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
    images: np.ndarray,
    tensors: np.ndarray,
    mass_weights: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Fourier-accumulate requested q points and symmetrize Hermitian roundoff.

    points is fractional reciprocal (nq, 3); first/second index primitive sites,
    images holds fractional positional-gauge separations, tensors is (terms, 3, 3)
    in eV/angstrom**2, and mass_weights is 1/sqrt(m_i*m_j) in inverse atomic mass
    units. Return complex128 (nq, 3*N, 3*N), before conversion to THz.
    Uses exp(+2*pi*i*q.dot(image)); no full reciprocal grid is required.
    """
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


def accumulate_dynamical_matrices(points, terms, tensors, n_sites):
    """Evaluate dynamical matrices from normalized q points and prepared terms.

    ``points`` has shape (nq, 3) in fractional reciprocal coordinates;
    ``tensors`` has shape (n_terms, 3, 3) in eV/angstrom**2. Return complex128
    shape (nq, 3*n_sites, 3*n_sites), Hermitian-symmetrized by the compiled
    accumulation. The calculation uses the positional Fourier gauge.
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
