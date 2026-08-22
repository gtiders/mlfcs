"""Constant-time atom indexing over a finite translation quotient."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlfcs.structure.integer_lattice import (
    IntegerLatticeQuotient,
    determinant_3x3,
    normalize_supercell_matrix,
)


@dataclass(frozen=True, slots=True)
class PeriodicIndex:
    """O(1) atom lookup using primitive site and a lattice-quotient residue."""

    primitive: np.ndarray
    translations: np.ndarray
    supercell_matrix: np.ndarray
    _quotient: IntegerLatticeQuotient = field(init=False, repr=False, compare=False)
    _atom_by_site_cell: np.ndarray = field(init=False, repr=False, compare=False)
    _translation_by_cell: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        primitive = np.asarray(self.primitive, dtype=np.int32)
        translations = np.asarray(self.translations, dtype=np.int32)
        matrix = normalize_supercell_matrix(self.supercell_matrix)
        quotient = IntegerLatticeQuotient(matrix)
        if primitive.ndim != 1 or translations.shape != (len(primitive), 3):
            raise ValueError("primitive labels and translations have incompatible shapes")
        if (
            len(primitive) == 0
            or primitive.min() != 0
            or not np.array_equal(np.unique(primitive), np.arange(int(primitive.max()) + 1))
        ):
            raise ValueError("primitive labels must be contiguous and start at zero")
        expected = self.n_primitive * abs(determinant_3x3(matrix))
        if len(primitive) != expected:
            raise ValueError("reference does not contain exactly one atom per primitive-site coset")
        atom_by_site_cell = np.full((self.n_primitive, quotient.size), -1, dtype=np.int32)
        translation_by_cell = np.empty((quotient.size, 3), dtype=np.int32)
        for atom, (site, translation) in enumerate(zip(primitive, translations, strict=True)):
            cell = quotient.cell_index(translation)
            if atom_by_site_cell[int(site), cell] >= 0:
                raise ValueError("more than one reference atom has the same periodic label")
            atom_by_site_cell[int(site), cell] = atom
            translation_by_cell[cell] = translation
        if np.any(atom_by_site_cell < 0):
            raise ValueError("reference atom lookup is incomplete")
        object.__setattr__(self, "primitive", primitive)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "supercell_matrix", matrix)
        object.__setattr__(self, "_quotient", quotient)
        object.__setattr__(self, "_atom_by_site_cell", atom_by_site_cell)
        object.__setattr__(self, "_translation_by_cell", translation_by_cell)

    @property
    def n_primitive(self) -> int:
        return int(np.max(self.primitive)) + 1

    @property
    def n_cells(self) -> int:
        return abs(determinant_3x3(self.supercell_matrix))

    @property
    def cell_representatives(self) -> np.ndarray:
        """Canonical HNF translations in deterministic cell-index order."""
        return self._quotient.representatives.copy()

    def residue(self, translation: np.ndarray) -> tuple[int, int, int]:
        return tuple(int(value) for value in self._quotient.reduce(np.asarray(translation)))

    def canonical_translation(self, translation: np.ndarray) -> np.ndarray:
        """Reference-frame representative of a primitive-lattice translation residue."""
        return self._translation_by_cell[self._quotient.cell_index(translation)].copy()

    def atom(self, primitive: int, translation: np.ndarray) -> int:
        if primitive < 0 or primitive >= self.n_primitive:
            raise ValueError("unknown primitive site or translation residue")
        return int(self._atom_by_site_cell[int(primitive), self._quotient.cell_index(translation)])

    def atom_many(self, primitive: np.ndarray, translations: np.ndarray) -> np.ndarray:
        """Vectorized atom lookup with NumPy broadcasting over leading axes."""
        sites, _ = np.broadcast_arrays(
            np.asarray(primitive, dtype=np.int64),
            np.asarray(translations, dtype=np.int64)[..., 0],
        )
        full_translations = np.broadcast_to(
            np.asarray(translations, dtype=np.int64), sites.shape + (3,)
        )
        if np.any(sites < 0) or np.any(sites >= self.n_primitive):
            raise ValueError("unknown primitive site or translation residue")
        cells = self._quotient.cell_index_many(full_translations)
        return self._atom_by_site_cell[sites, cells]

    def translate_atom(self, atom: int, shift: np.ndarray) -> int:
        return self.atom(int(self.primitive[atom]), self.translations[atom] + np.asarray(shift))

    def translate_atoms(self, atoms: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """Translate an atom array by every supplied primitive-lattice shift."""
        atom_values = np.asarray(atoms, dtype=np.int64)
        shift_values = np.asarray(shifts, dtype=np.int64)
        if shift_values.ndim == 1:
            if shift_values.shape != (3,):
                raise ValueError("shift must have shape (3,) or (n, 3)")
            shift_values = shift_values[None, :]
        if shift_values.ndim != 2 or shift_values.shape[1] != 3:
            raise ValueError("shifts must have shape (n, 3)")
        sites = np.broadcast_to(
            self.primitive[atom_values], (len(shift_values),) + atom_values.shape
        )
        translations = self.translations[atom_values][None, ...] + shift_values.reshape(
            (len(shift_values),) + (1,) * atom_values.ndim + (3,)
        )
        return self.atom_many(sites, translations)

    def anchor(self, cluster: tuple[int, ...]) -> tuple[int, ...]:
        shift = -self.translations[cluster[0]]
        return tuple(self.translate_atom(atom, shift) for atom in cluster)

    def representative(self, primitive: int) -> int:
        return self.atom(primitive, np.zeros(3, dtype=np.int32))
