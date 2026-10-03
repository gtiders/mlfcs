"""Supercell validation, periodic quotient indexing and label mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from numba import njit

from mlfcs._arrays import integer_array, readonly, require_allocation, require_bound
from mlfcs.algebra.exact import to_python_rows
from mlfcs.algebra.integer import adjugate_3x3, determinant_3x3, prove_integer_product


@njit(cache=True)
def quotient_kernel(translation, adjugate, modulus):
    result = np.zeros(3, dtype=np.int64)
    for j in range(3):
        for k in range(3):
            result[j] += translation[k] * adjugate[k, j]
        result[j] %= modulus
    return result


@njit(cache=True)
def map_labels(labels, translations, adjugate, modulus, keys, atoms):
    result = np.empty((len(labels), len(translations)), dtype=np.int64)
    for i in range(len(labels)):
        for t in range(len(translations)):
            shift = labels[i, 1:] + translations[t]
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
class _PeriodicIndex:
    adjugate: np.ndarray
    modulus: int
    keys: np.ndarray
    atom_indices: np.ndarray


def prepare_periodic_index(supercell):
    # The 3x3 preprocessing uses Python integers before conversion, never an
    # unchecked determinant/adjugate multiplication in a fixed-width array.
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

    labels, translations = integer_array(labels), integer_array(translations)
    if (
        labels.ndim != 2
        or labels.shape[1] != 4
        or translations.ndim != 2
        or translations.shape[1] != 3
    ):
        raise ValueError("mapping labels and translations must have shapes (n,4) and (m,3)")
    require_allocation("mapped atoms", (len(labels), len(translations)))
    for label in labels:
        for translation in translations:
            shift = tuple(int(label[k + 1]) + int(translation[k]) for k in range(3))
            for value in shift:
                require_bound("translated labels", abs(value))
            for j in range(3):
                require_bound(
                    "periodic quotient dot product",
                    sum(abs(shift[k] * int(prepared.adjugate[k, j])) for k in range(3)),
                )
    return map_labels(
        labels,
        translations,
        prepared.adjugate,
        prepared.modulus,
        prepared.keys,
        prepared.atom_indices,
    )


def infer_supercell_matrix(cluster_space, supercell_atoms):
    """Infer the integer supercell matrix from the two lattices.

    Row convention: ``supercell_cell = matrix @ primitive_cell``. The real
    matrix is rounded to the nearest integers and accepted only when the
    reconstructed lattice reproduces the supercell cell within the declared
    ``symprec``.
    """
    primitive_cell = np.asarray(cluster_space.cell, dtype=np.float64)
    supercell_cell = np.asarray(supercell_atoms.cell, dtype=np.float64)
    real = supercell_cell @ np.linalg.inv(primitive_cell)
    rounded = np.rint(real)
    if not np.all(np.isfinite(rounded)) or np.any(np.abs(rounded) >= float(1 << 63)):
        raise OverflowError("inferred supercell matrix cannot enter int64")
    matrix = rounded.astype(np.int64)
    residual = float(np.max(np.linalg.norm(matrix @ primitive_cell - supercell_cell, axis=1)))
    if residual >= cluster_space.symprec:
        raise ValueError(
            "the supercell lattice is not an integer transform of the primitive cell; "
            f"residual {residual:.10g} angstrom is not below symprec "
            f"{cluster_space.symprec:.10g} angstrom"
        )
    return readonly(matrix, np.int64)


def supercell_data(cluster_space, atoms, matrix):
    """Map an external supercell using the primitive symprec."""
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
    sites = np.empty(len(atoms), dtype=np.int64)
    translations = np.empty((len(atoms), 3), dtype=np.int64)
    numbers = integer_array(atoms.numbers, name="supercell atomic numbers")
    for atom, (number, position) in enumerate(zip(numbers, primitive_scaled, strict=True)):
        candidates = np.flatnonzero(cluster_space.atomic_numbers == number)
        differences = position - cluster_space.scaled_positions[candidates]
        rounded = np.rint(differences)
        if not np.all(np.isfinite(rounded)) or np.any(np.abs(rounded) >= float(1 << 63)):
            raise OverflowError("supercell translations cannot enter int64")
        shifts = rounded.astype(np.int64)
        distances = np.linalg.norm((differences - shifts) @ cluster_space.cell, axis=1)
        matches = np.flatnonzero(distances < cluster_space.symprec)
        if len(matches) != 1:
            nearest = float(np.min(distances)) if distances.size else float("inf")
            raise ValueError(
                f"supercell atom {atom} has {len(matches)} primitive matches below "
                f"symprec; nearest residual is {nearest:.10g} angstrom"
            )
        match = int(matches[0])
        sites[atom] = int(candidates[match])
        translations[atom] = shifts[match]

    adjugate = adjugate_3x3(rows)
    modulus = abs(determinant)
    quotients = np.empty((len(atoms), 3), dtype=np.int64)
    prove_integer_product(translations, adjugate)
    for atom, translation in enumerate(translations):
        quotients[atom] = quotient_kernel(translation, adjugate, modulus)
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
