"""Periodic quotient arithmetic for primitive-to-supercell atom indexing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import as_int64_array, readonly, require_allocation, require_bound
from mlfcs.foundation.integer import adjugate_3x3

INT64_MAX = (1 << 63) - 1


@njit(cache=True, inline="always")
def quotient_label(translation, adjugate, modulus):
    """Return the periodic quotient label of a primitive lattice translation.

    For supercell matrix ``S``, translations that differ by a supercell
    lattice vector belong to the same class of ``Z^3 / Z^3 S``. Using
    ``adjugate = adj(S)`` and ``modulus = abs(det(S))``, the class is
    represented by ``q(n) = n @ adj(S) mod modulus``.

    Parameters
    ----------
    translation
        Primitive lattice translation ``n`` of shape ``(3,)``.
    adjugate
        Integer adjugate of the supercell matrix, shape ``(3, 3)``.
    modulus
        Positive ``abs(det(S))``.

    Returns
    -------
    ndarray
        Canonical nonnegative integer quotient label of shape ``(3,)``.

    Raises
    ------
    OverflowError
        If an exact fixed-width intermediate cannot be represented safely.
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
    """Map translated primitive lattice labels to supercell atom indices.

    For every primitive label ``(i, n)`` and translation ``t``, form
    ``(i, [n + t])``, where ``[n + t]`` is the corresponding class in the
    supercell translation quotient. The resulting key is looked up in the
    validated periodic index.

    Parameters
    ----------
    labels
        Primitive lattice labels of shape ``(n, 4)`` as
        ``(site, tx, ty, tz)``.
    translations
        Additional primitive lattice translations of shape ``(m, 3)``.
    adjugate, modulus
        Integer data defining the supercell quotient.
    keys
        Lexicographically sorted ``(site, quotient)`` keys.
    atoms
        Supercell atom indices corresponding to ``keys``.

    Returns
    -------
    ndarray
        Supercell atom indices of shape ``(n, m)``.

    Raises
    ------
    ValueError
        If a periodic quotient site is absent from the validated supercell.
    OverflowError
        If translated labels or quotient arithmetic exceed the supported
        integer range.
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
            q = quotient_label(shift, adjugate, modulus)
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
    """Lookup from primitive quotient sites to supercell atom indices.

    A periodic site is identified by a primitive motif index together with a
    translation class in the finite quotient defined by the supercell matrix.

    ``keys`` stores these ``(site, qx, qy, qz)`` identifiers in lexicographic
    order, while ``atom_indices`` maps them back to the original supercell
    atom ordering.
    """

    adjugate: np.ndarray
    modulus: int
    keys: np.ndarray
    atom_indices: np.ndarray


def prepare_periodic_index(supercell_matrix, determinant, primitive_site_indices, quotient_labels):
    """Build the periodic atom lookup for a validated supercell.

    Each supercell atom is identified by its primitive motif site and
    quotient translation label. These identifiers are sorted to support
    deterministic lookup while preserving a map back to the original atom
    ordering.

    Parameters
    ----------
    supercell_matrix
        Integer ``(3, 3)`` supercell matrix ``S``.
    determinant
        Exact nonzero determinant of ``S``.
    primitive_site_indices
        Primitive motif-site index of each supercell atom.
    quotient_labels
        Quotient translation label of each supercell atom, shape ``(n, 3)``.

    Returns
    -------
    PeriodicIndex
        Prepared lookup data for primitive-label to supercell-atom mapping.
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
    """Map primitive lattice labels under additional translations to supercell atoms.

    For every pair of a lattice label and an additional translation, return
    the atom index of the corresponding periodic site in the validated
    supercell.

    Parameters
    ----------
    labels
        Primitive lattice labels of shape ``(n, 4)`` as
        ``(site, tx, ty, tz)``.
    translations
        Primitive lattice translations of shape ``(m, 3)``.
    prepared
        Periodic quotient index of the supercell.

    Returns
    -------
    ndarray
        Integer atom-index array of shape ``(n, m)``.
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


__all__ = ["PeriodicIndex", "map_labels", "prepare_periodic_index", "quotient_label"]
