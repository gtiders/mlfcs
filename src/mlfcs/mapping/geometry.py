"""Supercell validation, periodic quotient indexing and label mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from numba import njit

from mlfcs._arrays import integer_array, readonly, require_allocation, require_bound
from mlfcs.algebra.exact import to_python_rows
from mlfcs.algebra.integer import adjugate_3x3, determinant_3x3
from mlfcs.core.geometry import PeriodicGeometry

INT64_MAX = (1 << 63) - 1


@njit(cache=True, inline="always")
def _quotient_kernel(translation, adjugate, modulus):
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
            q = _quotient_kernel(shift, adjugate, modulus)
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
class _PeriodicIndex:
    """Readonly exact quotient lookup for a validated supercell realization.

    Keys are lexicographically sorted (site, qx, qy, qz) rows; atom_indices
    recovers the external atom ordering. Modulus is abs(det(supercell_matrix)).
    """
    adjugate: np.ndarray
    modulus: int
    keys: np.ndarray
    atom_indices: np.ndarray


def prepare_periodic_index(supercell):
    # The 3x3 preprocessing uses Python integers before conversion, never an
    # unchecked determinant/adjugate multiplication in a fixed-width array.
    """Build sorted readonly quotient keys and exact adjugate data from a validated ClusterMap."""
    adjugate = adjugate_3x3(supercell.supercell_matrix)
    modulus = require_bound("supercell quotient modulus", abs(supercell.determinant))
    require_allocation("periodic index keys", (supercell.n_atoms, 4))
    keys = np.column_stack((supercell.primitive_site_indices, supercell.quotient_labels)).astype(
        np.int64
    )
    order = np.lexsort(keys[:, ::-1].T)
    return _PeriodicIndex(
        adjugate,
        modulus,
        readonly(keys[order], np.int64),
        readonly(order, np.int64),
    )


def mapped_labels(labels, translations, prepared):
    """Normalize label buffers and map their Cartesian product of translations.

    Return int64 atom indices with shape (n_labels, n_translations). Input
    labels are (site, tx, ty, tz) rows; translations are integer triples.
    Allocation validation precedes the compiled path, which checks actual
    addition, product and accumulation operations instead of a global bound.
    """
    labels, translations = integer_array(labels), integer_array(translations)
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


def infer_supercell_matrix(cluster_space, supercell_atoms):
    """Infer the integer supercell matrix from the two lattices.

    Row convention: ``supercell_cell = matrix @ primitive_cell``. Each
    supercell lattice vector must match exactly one primitive lattice image
    within the declared ``symprec``.
    """
    if not isinstance(supercell_atoms, Atoms):
        raise TypeError("supercell_atoms must be an ASE Atoms object")
    primitive_cell = np.asarray(cluster_space.cell, dtype=np.float64)
    supercell_cell = np.asarray(supercell_atoms.cell, dtype=np.float64)
    if not np.all(np.isfinite(supercell_cell)):
        raise ValueError("supercell cell must be finite")

    geometry = PeriodicGeometry(primitive_cell)
    rows = []
    for vector in supercell_cell:
        matches, shifts = geometry.matching_images(vector[None, :], tolerance=cluster_space.symprec)
        if len(matches) != 1:
            raise ValueError(
                "the supercell lattice vector must match exactly one primitive lattice "
                f"image within symprec (found {len(matches)} matches)"
            )
        rows.append(tuple(-int(value) for value in shifts[0]))
    matrix = integer_array(rows, name="inferred supercell matrix")
    return readonly(matrix, np.int64)


def supercell_data(cluster_space, atoms, matrix):
    """Validate a declared supercell and return readonly geometry and lattice addresses.

    matrix uses cell_super = matrix @ cell_primitive, with integer entries and
    nonzero determinant. Require full PBC, matching atom count and positive
    masses. Same-species primitive matching uses Cartesian cluster-space
    symprec; every atom must have exactly one match and a unique periodic key.
    Preserve external atom order. Invalid geometry/matching raises ValueError;
    nonrepresentable quotient arithmetic raises OverflowError.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("supercell_atoms must be an ASE Atoms object")
    require_allocation("supercell labels", (len(atoms), 3))
    if not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(atoms.cell.array)):
        raise ValueError("supercell geometry must be finite")
    if not np.all(np.isfinite(atoms.get_masses())) or np.any(atoms.get_masses() <= 0):
        raise ValueError("atomic masses must be positive and finite")
    if not bool(np.all(atoms.pbc)):
        raise ValueError("supercell must be periodic in all three directions")
    rows = to_python_rows(matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("supercell matrix must have shape (3, 3)")
    determinant = determinant_3x3(rows)
    if determinant == 0:
        raise ValueError("supercell matrix must be nonsingular")
    copies = abs(determinant)
    require_bound("supercell determinant", copies)
    if len(atoms) != cluster_space.n_atoms * copies:
        raise ValueError(
            f"supercell has {len(atoms)} atoms, expected {cluster_space.n_atoms * copies} "
            f"from determinant {determinant}"
        )
    integer_matrix = integer_array(rows, name="supercell matrix")
    expected_cell = np.asarray(integer_matrix, dtype=np.float64) @ cluster_space.cell
    actual_cell = np.asarray(atoms.cell, dtype=np.float64)
    residual = float(np.max(np.linalg.norm(expected_cell - actual_cell, axis=1)))
    if residual >= cluster_space.symprec:
        raise ValueError(
            f"supercell lattice residual {residual:.10g} angstrom is not below "
            f"symprec {cluster_space.symprec:.10g} angstrom"
        )

    cartesian = atoms.get_positions()
    primitive_scaled = cartesian @ np.linalg.inv(cluster_space.cell)
    geometry = PeriodicGeometry(cluster_space.cell)
    sites = np.empty(len(atoms), dtype=np.int64)
    translations = np.empty((len(atoms), 3), dtype=np.int64)
    numbers = integer_array(atoms.numbers, name="supercell atomic numbers")
    for atom, (number, position) in enumerate(zip(numbers, primitive_scaled, strict=True)):
        candidates = np.flatnonzero(cluster_space.atomic_numbers == number)
        differences = (position - cluster_space.scaled_positions[candidates]) @ cluster_space.cell
        matches, matched_shifts = geometry.matching_images(
            differences, tolerance=cluster_space.symprec
        )
        if len(matches) != 1:
            raise ValueError(
                f"supercell atom {atom} has {len(matches)} primitive matches below symprec"
            )
        sites[atom] = int(candidates[matches[0]])
        translations[atom] = -matched_shifts[0]

    adjugate = adjugate_3x3(rows)
    modulus = abs(determinant)
    quotients = np.empty((len(atoms), 3), dtype=np.int64)
    for atom, translation in enumerate(translations):
        quotients[atom] = _quotient_kernel(translation, adjugate, modulus)
    keys = [
        (int(site), *(int(value) for value in quotient))
        for site, quotient in zip(sites, quotients, strict=True)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("supercell does not contain each primitive quotient site exactly once")

    return {
        "supercell_matrix": readonly(integer_matrix, np.int64),
        "cell": readonly(actual_cell, np.float64),
        "scaled_positions": readonly(atoms.get_scaled_positions(wrap=True), np.float64),
        "atomic_numbers": readonly(numbers, np.int64),
        "primitive_site_indices": readonly(sites, np.int64),
        "lattice_translations": readonly(translations, np.int64),
        "quotient_labels": readonly(quotients, np.int64),
        "determinant": determinant,
        "masses": readonly(atoms.get_masses(), np.float64),
    }
