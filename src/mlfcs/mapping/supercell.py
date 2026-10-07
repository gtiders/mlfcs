"""Construct the periodic lattice mapping between a primitive cell and a supercell."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from mlfcs.foundation.arrays import as_int64_array, readonly, require_allocation, require_bound
from mlfcs.foundation.integer import adjugate_3x3, determinant_3x3, to_python_rows
from mlfcs.geometry.periodic import PeriodicGeometry
from mlfcs.mapping.periodic import quotient_label


def infer_supercell_matrix(cluster_space, supercell_atoms):
    """Infer the integer matrix relating a primitive cell to a supercell.

    The matrix ``S`` is defined by the row-vector convention

        cell_super = S @ cell_primitive.

    Each supercell lattice vector must correspond to a unique integer
    combination of primitive lattice vectors within ``cluster_space.symprec``.

    Returns
    -------
    ndarray
        Readonly ``int64`` matrix of shape ``(3, 3)``.

    Raises
    ------
    TypeError
        If ``supercell_atoms`` is not an ASE ``Atoms`` object.
    ValueError
        If the supercell lattice is nonfinite or cannot be matched uniquely
        to the primitive lattice within the symmetry tolerance.
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
    matrix = as_int64_array(rows, name="inferred supercell matrix")
    return readonly(matrix, np.int64)


def prepare_supercell_data(cluster_space, atoms, matrix):
    """Validate a supercell and map its atoms to primitive lattice sites.

    For supercell matrix ``S`` defined by

        cell_super = S @ cell_primitive,

    each supercell atom is assigned a unique primitive motif site ``i`` and
    integer lattice translation ``n``. Translations are further reduced to
    classes of the finite quotient ``Z^3 / Z^3 S``. The validated atom set
    must contain every combination of primitive motif site and supercell
    translation class exactly once, so the atom count equals
    ``n_primitive * abs(det(S))``. The mapping follows from species, geometry
    and the lattice quotient alone; external atom ordering is preserved but
    need not follow any replication order.

    Returns
    -------
    dict
        Supercell geometry, primitive-site indices, lattice translations,
        quotient labels, determinant and atomic masses.

    Raises
    ------
    TypeError
        If ``atoms`` is not an ASE ``Atoms`` object.
    ValueError
        If the supercell geometry, lattice relation, atom count, masses,
        periodicity, primitive-site matching or quotient uniqueness is
        invalid.
    OverflowError
        If the integer quotient arithmetic exceeds the supported domain.
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
    integer_matrix = as_int64_array(rows, name="supercell matrix")
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
    numbers = as_int64_array(atoms.numbers, name="supercell atomic numbers")
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
        quotients[atom] = quotient_label(translation, adjugate, modulus)
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


__all__ = ["infer_supercell_matrix", "prepare_supercell_data"]
