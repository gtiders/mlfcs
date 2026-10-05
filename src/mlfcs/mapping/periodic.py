"""Periodic quotient arithmetic and primitive-label to supercell lookup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs._arrays import as_int64_array, readonly, require_allocation, require_bound
from mlfcs.algebra.matrix import adjugate_3x3

INT64_MAX = (1 << 63) - 1


@njit(cache=True, inline="always")
def quotient_kernel(translation, adjugate, modulus):
    """Return translation @ adjugate modulo the positive supercell determinant magnitude.

    Translation has shape (3,), adjugate (3, 3), both in the symmetric int64
    domain. Check every multiplication and running addition before executing
    it; raise OverflowError even if later cancellation would make the final
    residue small. Return a new canonical nonnegative int64 quotient label.
    """
    result = np.zeros(3, dtype=np.int64)
    for j in range(3):
        for k in range(3):
            value, factor = translation[k], adjugate[k, j]
            if factor != 0 and abs(value) > INT64_MAX // abs(factor):
                raise OverflowError("periodic quotient intermediate exceeds int64")
            term = value * factor
            if term > 0 and result[j] > INT64_MAX - term:
                raise OverflowError("periodic quotient intermediate exceeds int64")
            if term < 0 and result[j] < -INT64_MAX - term:
                raise OverflowError("periodic quotient intermediate exceeds int64")
            result[j] += term
        result[j] %= modulus
    return result


@njit(cache=True)
def _map_labels_kernel(labels, translations, adjugate, modulus, keys, atoms):
    """Map every lattice label plus translation to its original supercell atom index.

    Labels are (n, 4), translations (m, 3). Sorted keys are (n_atoms, 4)
    (site, quotient), and atoms maps sorted rows back to input atom order.
    Return (n, m) int64 indices. Translation addition and quotient arithmetic
    are checked in this single compiled path. Missing keys raise ValueError;
    nonrepresentable intermediates raise OverflowError. Inputs are preserved.
    """
    result = np.empty((len(labels), len(translations)), dtype=np.int64)
    for i in range(len(labels)):
        for t in range(len(translations)):
            shift = np.empty(3, dtype=np.int64)
            for k in range(3):
                value, offset = labels[i, k + 1], translations[t, k]
                if offset > 0 and value > INT64_MAX - offset:
                    raise OverflowError("translated labels exceed int64")
                if offset < 0 and value < -INT64_MAX - offset:
                    raise OverflowError("translated labels exceed int64")
                shift[k] = value + offset
            q = quotient_kernel(shift, adjugate, modulus)
            key = np.empty(4, dtype=np.int64)
            key[0], key[1:] = labels[i, 0], q
            left, right = 0, len(keys)
            while left < right:
                middle = left + (right - left) // 2
                comparison = 0
                for j in range(4):
                    if keys[middle, j] < key[j]:
                        comparison = -1
                        break
                    if keys[middle, j] > key[j]:
                        comparison = 1
                        break
                if comparison < 0:
                    left = middle + 1
                else:
                    right = middle
            if left == len(keys) or not np.all(keys[left] == key):
                raise ValueError("periodic quotient site is absent from the supercell")
            result[i, t] = atoms[left]
    return result


@dataclass(frozen=True, slots=True)
class PeriodicIndex:
    """Readonly quotient lookup for a validated supercell realization.

    Keys are lexicographically sorted (site, qx, qy, qz) rows; atom_indices
    recovers the external atom ordering. Modulus is abs(det(supercell_matrix)).
    """

    adjugate: np.ndarray
    modulus: int
    keys: np.ndarray
    atom_indices: np.ndarray


def prepare_periodic_index(supercell_matrix, determinant, primitive_site_indices, quotient_labels):
    """Build sorted readonly quotient keys and adjugate data from validated geometry.

    The matrix is an integer (3, 3) row-basis supercell transform; determinant is
    its exact nonzero Python-integer determinant. Site indices have shape (n,),
    quotient labels (n, 3), and outputs are contiguous readonly int64 buffers.
    Python integer matrix preprocessing precedes all fixed-width storage.
    """
    # The 3x3 preprocessing uses Python integers before conversion.
    adjugate = adjugate_3x3(supercell_matrix)
    modulus = require_bound("supercell quotient modulus", abs(determinant))
    require_allocation("periodic index keys", (len(primitive_site_indices), 4))
    keys = np.column_stack((primitive_site_indices, quotient_labels)).astype(np.int64)
    order = np.lexsort(keys[:, ::-1].T)
    return PeriodicIndex(
        adjugate,
        modulus,
        readonly(keys[order], np.int64),
        readonly(order, np.int64),
    )


def map_labels(labels, translations, prepared):
    """Normalize label buffers and map their Cartesian product of translations.

    Return int64 atom indices with shape (n_labels, n_translations). Input
    labels are (site, tx, ty, tz) rows; translations are integer triples.
    Allocation validation precedes the compiled path, which checks actual
    addition, product and accumulation operations instead of a global bound.
    """
    labels, translations = as_int64_array(labels), as_int64_array(translations)
    if (
        labels.ndim != 2
        or labels.shape[1] != 4
        or translations.ndim != 2
        or translations.shape[1] != 3
    ):
        raise ValueError("mapping labels and translations must have shapes (n,4) and (m,3)")
    require_allocation("mapped atoms", (len(labels), len(translations)))
    return _map_labels_kernel(
        labels,
        translations,
        prepared.adjugate,
        prepared.modulus,
        prepared.keys,
        prepared.atom_indices,
    )


__all__ = ["PeriodicIndex", "map_labels", "prepare_periodic_index", "quotient_kernel"]
