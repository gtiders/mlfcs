"""Fourier transforms and dynamical matrices for SCPH."""

from __future__ import annotations

import numpy as np

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.phonon.reciprocal import quotient_qpoints
from mlfcs.phonon.statistics import OMEGA_TO_THZ as _OMEGA_TO_THZ


def harmonic_frequencies(
    fc2: ForceConstants, interpolation_multiplier: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic frequencies on a reference-supercell-derived q grid."""
    if 2 not in fc2.orders or fc2.relation is None:
        raise ValueError("fc2 must contain order-2 force constants and a structure relation")
    masses = np.asarray(fc2.relation.primitive.get_masses(), dtype=float)
    lattice = lattice_fc2(fc2)
    terms = _fourier_terms(lattice, fc2.relation.primitive)
    multiplier = _multiplier(interpolation_multiplier, "interpolation_multiplier")
    qpoints = quotient_qpoints(multiplier * fc2.relation.supercell_matrix)
    frequencies = []
    for q in qpoints:
        values = np.linalg.eigvalsh(_dynamical(terms, masses, q))
        frequencies.append(np.sqrt(np.abs(values)) * np.sign(values) * _OMEGA_TO_THZ)
    return qpoints, np.asarray(frequencies)


def _multiplier(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_relation(fc2: ForceConstants, fc4: ForceConstants) -> None:
    r2, r4 = fc2.relation, fc4.relation
    if r2 is None or r4 is None:
        raise ValueError("fc2 and fc4 must contain explicit StructureRelation objects")
    if (
        len(r2.primitive) != len(r4.primitive)
        or not np.array_equal(r2.primitive.numbers, r4.primitive.numbers)
        or not np.allclose(r2.primitive.cell, r4.primitive.cell, atol=1e-8, rtol=1e-10)
        or not np.allclose(r2.primitive.positions, r4.primitive.positions, atol=1e-8, rtol=1e-10)
    ):
        raise ValueError("fc2 and fc4 primitive structures differ")


def _fourier_terms(lattice, primitive):
    """Build Fourier terms directly from exact primitive-lattice FC2 labels."""
    terms = []
    scaled = primitive.get_scaled_positions(wrap=False)
    for (first, site, translation), tensor in lattice.items():
        vector = scaled[site] - scaled[first] + np.asarray(translation, dtype=float)
        terms.append((first, site, vector, tensor))
    return terms


def _dynamical(terms, masses: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = len(masses)
    matrix = np.zeros((3 * n, 3 * n), dtype=complex)
    for a, b, images, tensor in terms:
        phase = np.exp(2j * np.pi * float(images @ q))
        matrix[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += (
            tensor * phase / np.sqrt(masses[a] * masses[b])
        )
    return (matrix + matrix.conj().T) / 2


def _dynamical_batch(terms, masses: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """Build all q-point dynamical matrices in one bounded batch."""
    qpoints = np.asarray(qpoints, dtype=float).reshape((-1, 3))
    n = len(masses)
    matrix = np.zeros((len(qpoints), 3 * n, 3 * n), dtype=complex)
    for first, second, images, tensor in terms:
        phase = np.exp(2j * np.pi * (qpoints @ images))
        matrix[:, 3 * first : 3 * first + 3, 3 * second : 3 * second + 3] += (
            phase[:, None, None] * tensor / np.sqrt(masses[first] * masses[second])
        )
    return (matrix + matrix.conj().swapaxes(-1, -2)) / 2


def _needed_covariances(
    sparse: SparseOrderForceConstants,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    result: set[tuple[int, int, tuple[int, int, int]]] = set()
    for sites, translations in zip(sparse.sites, sparse.translations, strict=True):
        result.add(
            (
                int(sites[2]),
                int(sites[3]),
                tuple((np.asarray(translations[1]) - np.asarray(translations[2])).tolist()),
            )
        )
    return result


__all__ = ["harmonic_frequencies"]
