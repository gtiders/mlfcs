"""Validated structure records for a reference cell and its explicit supercell.

``Cell`` is the input side of a model: a fully periodic reference cell
with wrapped fractional motif coordinates, species, masses and a symmetry
tolerance. ``Supercell`` is the derived side: every supercell atom is assigned a
cell motif site, an integer lattice translation, and its class in the
finite quotient ``Z^3 / Z^3 S`` defined by the integer supercell matrix ``S``
with ``cell_super = S @ cell``.

Records validate their own internal consistency on construction, so downstream
code never re-derives or re-checks the same facts. Conversion from ASE
structures lives in ``cell_data`` and ``supercell_data``; inference of
``S`` lives in ``infer_supercell_matrix``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from numba import njit

from mlfcs.foundation.arrays import (
    INT64_MAX,
    as_int64_array,
    readonly,
    require_allocation,
    require_bound,
)
from mlfcs.foundation.integer import adjugate_3x3, determinant_3x3, to_python_rows
from mlfcs.geometry.lattice import Lattice, match_cell_site


@dataclass(frozen=True, slots=True)
class Cell:
    """Validated reference cell of a force-constant model.

    The cell defines the model lattice, whether or not a smaller primitive cell
    exists. Fractional motif coordinates are wrapped into ``[0, 1)``; no
    minimality check, reduction or standardization is performed, so the
    supplied basis and atom order are preserved.

    ``masses`` are in atomic mass units and ``symprec`` is a positive Cartesian
    matching tolerance in angstrom. The prepared :class:`Lattice` is built once
    and shared by symmetry discovery and supercell assignment.
    """

    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    symprec: float
    _lattice: Lattice = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the reference cell, freeze its arrays and build its lattice."""
        cell = readonly(self.cell, np.float64)
        positions = np.asarray(self.scaled_positions, dtype=np.float64)
        numbers = np.asarray(self.atomic_numbers)
        masses = np.asarray(self.masses, dtype=np.float64)
        symprec = float(self.symprec)
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("cell scaled positions must have shape (n, 3)")
        if numbers.shape != (len(positions),):
            raise ValueError("cell atomic numbers must have shape (n,)")
        if masses.shape != (len(positions),):
            raise ValueError("atomic masses must have shape (n_atoms,)")
        if len(numbers) == 0:
            raise ValueError("cell must contain at least one atom")
        if not np.all(np.isfinite(cell)) or not np.all(np.isfinite(positions)):
            raise ValueError("cell and positions must be finite")
        if not np.isfinite(symprec) or symprec <= 0.0:
            raise ValueError("symprec must be a positive finite length in angstrom")
        if float(np.linalg.det(cell)) == 0.0:
            raise ValueError("cell must be nonsingular")
        if not np.issubdtype(numbers.dtype, np.integer):
            raise TypeError("cell atomic numbers must be integers")
        numbers = as_int64_array(numbers, name="cell atomic numbers")
        if (
            int(numbers.min()) < np.iinfo(np.int32).min
            or int(numbers.max()) > np.iinfo(np.int32).max
        ):
            raise OverflowError("cell atomic numbers exceed the spglib int32 interface")
        if np.any(positions < 0.0) or np.any(positions >= 1.0):
            raise ValueError("cell scaled positions must be wrapped into [0, 1)")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("atomic masses must be positive and finite")
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "scaled_positions", readonly(positions, np.float64))
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "masses", readonly(masses, np.float64))
        object.__setattr__(self, "symprec", symprec)
        object.__setattr__(self, "_lattice", Lattice(cell))

    @property
    def lattice(self) -> Lattice:
        """Periodic-image geometry of this cell, prepared once at construction."""
        return self._lattice

    @property
    def n_atoms(self) -> int:
        """Number of motif atoms in the reference cell."""
        return len(self.atomic_numbers)

    def match_site(self, position, species, *, context: str = ""):
        """Locate the unique motif site that a fractional position belongs to.

        Return ``(site, translation)`` with ``position = scaled_positions[site]
        + translation`` and the integer shift already expressed in the
        ``LatticeSite`` convention. Raise ``ValueError`` when the position
        matches no motif site of ``species``, or more than one, within
        ``symprec``.
        """
        return match_cell_site(
            self.cell,
            self.scaled_positions,
            self.atomic_numbers,
            position,
            species,
            tolerance=self.symprec,
            geometry=self._lattice,
            context=context,
        )


@njit(cache=True)
def quotient_label(translation, adjugate, modulus):
    """Return the periodic quotient label of a cell lattice translation.

    For supercell matrix ``S``, translations that differ by a supercell lattice
    vector belong to the same class of ``Z^3 / Z^3 S``. Using
    ``adjugate = adj(S)`` and ``modulus = abs(det(S))``, the class is
    represented by ``q(n) = n @ adj(S) mod modulus``.

    The canonical representative is a nonnegative integer triple.
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
    """Map translated cell lattice labels to supercell atom indices.

    For every label ``(i, n)`` and offset ``t``, form ``(i, [n + t])``, where
    ``[n + t]`` is the class in the supercell translation quotient, and look the
    resulting key up in the sorted periodic index. Return an ``(n_labels,
    n_translations)`` int64 array; raise ``ValueError`` for an absent key and
    ``OverflowError`` when label arithmetic leaves int64.
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
    """Lookup from cell quotient sites to supercell atom indices.

    A periodic site is identified by a cell motif index together with a
    translation class in the finite quotient defined by the supercell matrix.
    ``keys`` stores these ``(site, qx, qy, qz)`` identifiers in lexicographic
    order, while ``atom_indices`` maps them back to the supercell atom order.
    """

    adjugate: np.ndarray
    modulus: int
    keys: np.ndarray
    atom_indices: np.ndarray

    def __post_init__(self) -> None:
        """Validate and freeze the index arrays."""
        adjugate = as_int64_array(self.adjugate, name="supercell adjugate")
        keys = as_int64_array(self.keys, name="periodic index keys")
        atom_indices = as_int64_array(self.atom_indices, name="periodic index atoms")
        modulus = require_bound("supercell quotient modulus", int(self.modulus))
        if adjugate.shape != (3, 3):
            raise ValueError("supercell adjugate must have shape (3, 3)")
        if keys.ndim != 2 or keys.shape[1:] != (4,):
            raise ValueError("periodic index keys must have shape (n, 4)")
        if atom_indices.shape != (len(keys),):
            raise ValueError("periodic index atoms must have one entry per key")
        if modulus <= 0:
            raise ValueError("supercell quotient modulus must be positive")
        object.__setattr__(self, "adjugate", adjugate)
        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "atom_indices", atom_indices)

    def label(self, translation: object) -> tuple[int, int, int]:
        """Return the quotient class of one cell lattice translation."""
        values = as_int64_array(translation, name="lattice translation")
        if values.shape != (3,):
            raise ValueError("lattice translation must have shape (3,)")
        return tuple(int(v) for v in quotient_label(values, self.adjugate, self.modulus))

    def lookup(self, labels: object, translations: object) -> np.ndarray:
        """Map cell lattice labels under additional translations to supercell atoms.

        For each cell label ``(site, n)`` and translation offset ``t``,
        return the atom index of the periodic site ``(site, [n + t])`` in this
        supercell. Labels have shape ``(n, 4)`` as ``(site, tx, ty, tz)`` and
        translations shape ``(m, 3)``; the result has shape ``(n, m)``.
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
            self.adjugate,
            self.modulus,
            self.keys,
            self.atom_indices,
        )


@dataclass(frozen=True, slots=True)
class Supercell:
    """Realization of the reference cell in an explicit supercell.

    ``cell_super = matrix @ cell`` with an integer ``matrix`` of
    determinant ``determinant``. Every supercell atom carries its reference-cell
    motif site, the integer lattice translation that reaches it, and its class
    in ``Z^3 / Z^3 S``. Atom order follows the input structure and need not be
    any replication order; ``index`` resolves cell labels back to atoms.
    """

    matrix: np.ndarray
    determinant: int
    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    masses: np.ndarray
    cell_site_indices: np.ndarray
    lattice_translations: np.ndarray
    quotient_labels: np.ndarray
    index: PeriodicIndex

    def __post_init__(self) -> None:
        """Validate shapes, freeze arrays and check the declared determinant."""
        matrix = as_int64_array(self.matrix, name="supercell matrix")
        cell = readonly(self.cell, np.float64)
        positions = readonly(self.scaled_positions, np.float64)
        numbers = as_int64_array(self.atomic_numbers, name="supercell atomic numbers")
        masses = readonly(self.masses, np.float64)
        sites = as_int64_array(self.cell_site_indices, name="cell site indices")
        translations = as_int64_array(
            self.lattice_translations, name="supercell lattice translations"
        )
        quotients = as_int64_array(self.quotient_labels, name="supercell quotient labels")
        count = len(numbers)
        if matrix.shape != (3, 3):
            raise ValueError("supercell matrix must have shape (3, 3)")
        determinant = int(self.determinant)
        if determinant == 0 or determinant_3x3(to_python_rows(matrix)) != determinant:
            raise ValueError("supercell determinant is inconsistent with its matrix")
        if cell.shape != (3, 3) or positions.shape != (count, 3):
            raise ValueError("supercell geometry has inconsistent shapes")
        if masses.shape != (count,) or sites.shape != (count,):
            raise ValueError("supercell atom data has inconsistent shapes")
        if translations.shape != (count, 3) or quotients.shape != (count, 3):
            raise ValueError("supercell lattice data has inconsistent shapes")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "determinant", determinant)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "scaled_positions", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "cell_site_indices", sites)
        object.__setattr__(self, "lattice_translations", translations)
        object.__setattr__(self, "quotient_labels", quotients)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in this supercell."""
        return len(self.atomic_numbers)


def cell_data(atoms, symprec) -> Cell:
    """Extract the reference cell of a model from an ASE structure.

    The structure must be fully periodic in three dimensions. Its cell need
    not be cell; fractional coordinates are wrapped into the reference
    cell while the original lattice basis and atom order are preserved. The
    input ``Atoms`` object is neither retained nor modified.

    Raises
    ------
    TypeError
        If ``atoms`` is not an ASE ``Atoms`` object.
    ValueError
        If the structure is not fully periodic or its geometry, masses or
        wrapped coordinates are invalid.
    OverflowError
        If atomic numbers cannot enter the spglib integer interface.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("cell_atoms must be an ASE Atoms object")
    if not np.all(atoms.pbc):
        raise ValueError("cell must be periodic in all three directions")
    return Cell(
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(wrap=True),
        atomic_numbers=atoms.numbers,
        masses=atoms.get_masses(),
        symprec=symprec,
    )


def validate_cell_arrays(
    cell, scaled_positions, atomic_numbers, masses, symprec
) -> Cell:
    """Validate a periodic reference structure without changing its cell basis.

    Convenience constructor for callers that hold the raw arrays rather than an
    ASE structure, such as model files. The record performs every check; see
    :class:`Cell` for the accepted conventions.

    Raises
    ------
    TypeError
        If atomic numbers are not represented by integers.
    ValueError
        If the geometry, masses or wrapped coordinates are invalid.
    OverflowError
        If atomic numbers cannot enter the spglib integer interface.
    """
    return Cell(
        cell=cell,
        scaled_positions=scaled_positions,
        atomic_numbers=atomic_numbers,
        masses=masses,
        symprec=symprec,
    )


def infer_supercell_matrix(cell: Cell, supercell) -> np.ndarray:
    """Infer the integer matrix relating a cell to a supercell.

    The matrix ``S`` satisfies ``cell_super = S @ cell``. Each
    supercell lattice vector must correspond to a unique integer combination of
    cell lattice vectors within ``cell.symprec``.

    Raises
    ------
    TypeError
        If ``supercell`` is not an ASE ``Atoms`` object.
    ValueError
        If the supercell lattice is nonfinite or cannot be matched uniquely.
    """
    if not isinstance(supercell, Atoms):
        raise TypeError("supercell must be an ASE Atoms object")
    supercell_cell = np.asarray(supercell.cell, dtype=np.float64)
    if not np.all(np.isfinite(supercell_cell)):
        raise ValueError("supercell lattice must be finite")
    rows = [
        cell.lattice.lattice_combination(
            vector, tolerance=cell.symprec, context="supercell lattice vector"
        )
        for vector in supercell_cell
    ]
    return readonly(as_int64_array(rows, name="inferred supercell matrix"), np.int64)


def supercell_data(cell: Cell, atoms, matrix) -> Supercell:
    """Validate a supercell and assign every atom to a cell lattice site.

    The atom set must contain each combination of cell motif site and
    supercell translation class exactly once, so the atom count equals
    ``cell.n_atoms * abs(det(S))``. Assignment follows from species,
    geometry and the lattice quotient alone.

    Raises
    ------
    TypeError
        If ``atoms`` is not an ASE ``Atoms`` object.
    ValueError
        If the supercell geometry, lattice relation, atom count, masses,
        periodicity, cell-site matching or quotient uniqueness is invalid.
    OverflowError
        If integer quotient arithmetic exceeds the supported domain.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("supercell must be an ASE Atoms object")
    require_allocation("supercell labels", (len(atoms), 3))
    if not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(atoms.cell.array)):
        raise ValueError("supercell geometry must be finite")
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
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
    if len(atoms) != cell.n_atoms * copies:
        raise ValueError(
            f"supercell has {len(atoms)} atoms, expected {cell.n_atoms * copies} "
            f"from determinant {determinant}"
        )
    integer_matrix = as_int64_array(rows, name="supercell matrix")
    actual_cell = np.asarray(atoms.cell, dtype=np.float64)
    expected_cell = np.asarray(integer_matrix, dtype=np.float64) @ cell.cell
    residual = float(np.max(np.linalg.norm(expected_cell - actual_cell, axis=1)))
    if residual >= cell.symprec:
        raise ValueError(
            f"supercell lattice residual {residual:.10g} angstrom is not below "
            f"symprec {cell.symprec:.10g} angstrom"
        )

    cell_scaled = np.asarray(atoms.get_positions()) @ np.linalg.inv(cell.cell)
    numbers = as_int64_array(atoms.numbers, name="supercell atomic numbers")
    count = len(numbers)
    sites = np.empty(count, dtype=np.int64)
    translations = np.empty((count, 3), dtype=np.int64)
    for atom, (number, position) in enumerate(zip(numbers, cell_scaled, strict=True)):
        sites[atom], translations[atom] = cell.match_site(
            position, number, context=f"supercell atom {atom}"
        )

    adjugate = adjugate_3x3(rows)
    modulus = copies
    quotients = np.empty((count, 3), dtype=np.int64)
    for atom, translation in enumerate(translations):
        quotients[atom] = quotient_label(translation, adjugate, modulus)
    keys = np.column_stack((sites, quotients)).astype(np.int64)
    order = np.lexsort(keys[:, ::-1].T)
    sorted_keys = keys[order]
    if count > 1 and np.any(np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)):
        raise ValueError("supercell does not contain each cell quotient site exactly once")
    return Supercell(
        matrix=integer_matrix,
        determinant=determinant,
        cell=actual_cell,
        scaled_positions=atoms.get_scaled_positions(wrap=True),
        atomic_numbers=numbers,
        masses=masses,
        cell_site_indices=sites,
        lattice_translations=translations,
        quotient_labels=quotients,
        index=PeriodicIndex(
            adjugate=adjugate,
            modulus=modulus,
            keys=sorted_keys,
            atom_indices=order,
        ),
    )


__all__ = [
    "Cell",
    "PeriodicIndex",
    "cell_data",
    "infer_supercell_matrix",
    "supercell_data",
    "validate_cell_arrays",
]
