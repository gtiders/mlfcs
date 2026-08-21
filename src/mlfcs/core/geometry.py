"""Structure relations and periodic indexing independent of atom ordering.

The calculation frame is deliberately the user supplied reference supercell.
Integer translations are labels in the primitive lattice; their residues live
in ``Z^3 / Z^3 S`` for a general row-vector supercell matrix ``S``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from ase import Atoms
from ase.geometry import find_mic, minkowski_reduce
from scipy.optimize import linear_sum_assignment

from mlfcs.core.integer_lattice import (
    IntegerLatticeQuotient,
    determinant_3x3,
    normalize_supercell_matrix,
    residue_key,
)


def _translation_label(translation: np.ndarray, matrix: np.ndarray) -> tuple[int, int, int]:
    """Canonical exact label of a primitive-lattice translation modulo ``S``."""
    return residue_key(translation, matrix)


def _coset_translations(matrix: np.ndarray) -> np.ndarray:
    """Enumerate the canonical row-HNF fundamental domain."""
    return IntegerLatticeQuotient(matrix).representatives.copy()


@dataclass(frozen=True, slots=True)
class PeriodicGeometry:
    """One reduced-lattice implementation of periodic distance operations.

    The main calculation never enumerates a fixed image box in the supplied
    cell basis.  ASE selects a general minimum image; the reduced lattice is
    then used only to recover all images tied at that minimum.  This keeps
    skewed and unimodularly transformed frames on the same geometry rule.
    """

    cell: np.ndarray
    pbc: np.ndarray | bool = True
    atol: float = 1e-8
    rtol: float = 1e-10
    _reduction: np.ndarray = field(init=False, repr=False, compare=False)
    _closest_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, compare=False
    )
    _minimum_length_cache: dict[tuple[float, float, float], float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError("periodic geometry requires a nonsingular 3x3 cell")
        pbc = np.broadcast_to(np.asarray(self.pbc, dtype=bool), (3,)).copy()
        if not np.all(pbc):
            raise ValueError("MLFCS periodic geometry requires three periodic directions")
        _, reduction = minkowski_reduce(cell, pbc=pbc)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "_reduction", np.rint(reduction).astype(np.int32))
        object.__setattr__(self, "_closest_cache", {})
        object.__setattr__(self, "_minimum_length_cache", {})

    def mic(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ASE's general minimum-image vectors and their lengths."""
        values = np.asarray(vectors, dtype=float)
        return find_mic(values, self.cell, pbc=self.pbc)

    def pair_distances(self, positions: np.ndarray) -> np.ndarray:
        """Return the complete general-MIC distance matrix for ``positions``."""
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("pair_distances expects an (n, 3) Cartesian array")
        _, lengths = self.mic((values[None, :, :] - values[:, None, :]).reshape((-1, 3)))
        return np.asarray(lengths).reshape((len(values), len(values)))

    def closest_images(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return all degenerate nearest images and their supercell shifts.

        Shifts are row-vector integer coefficients of ``cell``.  The first
        returned array contains Cartesian image vectors; the second contains
        the matching shifts.  The search is local in a Minkowski-reduced
        basis around ASE's MIC result, rather than in a fixed source-cell box.
        """
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,):
            raise ValueError("closest_images expects one Cartesian 3-vector")
        key = tuple(float(component) for component in value)
        cached = self._closest_cache.get(key)
        if cached is not None:
            return cached
        mic, length = find_mic(value[None, :], self.cell, pbc=self.pbc)
        minimum = float(np.asarray(length).reshape(-1)[0])
        centre = np.rint((mic.reshape(3) - value) @ np.linalg.inv(self.cell)).astype(np.int32)
        local = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=np.int32)
        shifts = centre + local @ self._reduction
        images = value + shifts @ self.cell
        lengths = np.linalg.norm(images, axis=1)
        tied = np.abs(lengths - minimum) <= self.atol + self.rtol * max(minimum, 1.0)
        # A numerical tie may appear multiple times after a reduced-basis
        # transform.  Keep a deterministic unique representative of each
        # actual lattice shift.
        unique, locations = np.unique(shifts[tied], axis=0, return_index=True)
        order = np.argsort(locations)
        result = images[tied][locations[order]], unique[order]
        self._closest_cache[key] = result
        self._minimum_length_cache[key] = minimum
        return result

    def minimum_length(self, vector: np.ndarray) -> float:
        """Return a cached general-MIC length for one Cartesian vector."""
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,):
            raise ValueError("minimum_length expects one Cartesian 3-vector")
        key = tuple(float(component) for component in value)
        cached = self._minimum_length_cache.get(key)
        if cached is not None:
            return cached
        _, length = self.mic(value[None, :])
        result = float(np.asarray(length).reshape(-1)[0])
        self._minimum_length_cache[key] = result
        return result

    def joint_closest_image_shifts(self, vectors: np.ndarray) -> np.ndarray:
        """Return mutually compatible nearest-image shifts for an anchored cluster.

        ``vectors`` contains the Cartesian vectors from the anchor to every
        tail atom before a supercell image is selected.  Every returned row
        chooses a nearest image for each tail and also requires every
        tail-to-tail vector to be a minimum image.  This is the joint image
        convention required by lattice-labelled cluster exports.
        """
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("joint closest images expect an (n, 3) Cartesian array")
        if len(values) == 0:
            return np.zeros((1, 0, 3), dtype=np.int32)

        candidates: list[np.ndarray] = []
        for vector in values:
            _, shifts = self.closest_images(vector)
            candidates.append(shifts)

        compatible: list[np.ndarray] = []
        for selection in product(*candidates):
            shifts = np.asarray(selection, dtype=np.int32)
            images = values + shifts @ self.cell
            valid = True
            for left in range(len(images)):
                for right in range(left + 1, len(images)):
                    difference = images[right] - images[left]
                    length = float(np.linalg.norm(difference))
                    target = self.minimum_length(difference)
                    if abs(length - target) > self.atol + self.rtol * max(target, 1.0):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                compatible.append(shifts)
        if not compatible:
            return np.empty((0, len(values), 3), dtype=np.int32)
        return np.asarray(compatible, dtype=np.int32)


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
        sites = np.broadcast_to(self.primitive[atom_values], (len(shift_values),) + atom_values.shape)
        translations = self.translations[atom_values][None, ...] + shift_values.reshape(
            (len(shift_values),) + (1,) * atom_values.ndim + (3,)
        )
        return self.atom_many(sites, translations)

    def anchor(self, cluster: tuple[int, ...]) -> tuple[int, ...]:
        shift = -self.translations[cluster[0]]
        return tuple(self.translate_atom(atom, shift) for atom in cluster)

    def representative(self, primitive: int) -> int:
        return self.atom(primitive, np.zeros(3, dtype=np.int32))


@dataclass(frozen=True, slots=True)
class StructureRelation:
    """Verified relationship between an explicit primitive and reference frame."""

    primitive: Atoms
    reference: Atoms
    supercell_matrix: np.ndarray
    primitive_index: np.ndarray
    cell_translation: np.ndarray
    position_residual: float
    _index: PeriodicIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_index",
            PeriodicIndex(self.primitive_index, self.cell_translation, self.supercell_matrix),
        )

    @classmethod
    def from_atoms(
        cls, primitive: Atoms, reference: Atoms, *, tolerance: float = 1e-5
    ) -> StructureRelation:
        if not np.all(primitive.pbc) or not np.all(reference.pbc):
            raise ValueError("force constants require periodic primitive and reference structures")
        source_reference = reference
        primitive = primitive.copy()
        reference = reference.copy()
        reference.calc = source_reference.calc
        primitive.wrap()
        reference.wrap()
        transform = np.asarray(reference.cell) @ np.linalg.inv(np.asarray(primitive.cell))
        matrix = normalize_supercell_matrix(transform)
        if not np.allclose(transform, matrix, atol=tolerance, rtol=0.0):
            raise ValueError("reference is not an integer supercell of primitive")
        if abs(determinant_3x3(matrix)) * len(primitive) != len(reference):
            raise ValueError("supercell determinant and atom counts are inconsistent")
        labels = np.empty(len(reference), dtype=np.int32)
        translations = np.empty((len(reference), 3), dtype=np.int32)
        residuals = np.empty(len(reference), dtype=float)
        cell_translations = _coset_translations(matrix)
        geometry = PeriodicGeometry(reference.cell, reference.pbc)
        for number in np.unique(reference.numbers):
            reference_atoms = np.flatnonzero(reference.numbers == number)
            primitive_atoms = np.flatnonzero(primitive.numbers == number)
            if len(reference_atoms) != len(primitive_atoms) * len(cell_translations):
                raise ValueError(
                    "reference chemical composition is inconsistent with primitive images"
                )
            slot_sites = np.repeat(primitive_atoms, len(cell_translations))
            slot_translations = np.tile(cell_translations, (len(primitive_atoms), 1))
            slot_positions = primitive.positions[slot_sites] + slot_translations @ np.asarray(
                primitive.cell
            )
            delta = reference.positions[reference_atoms, None, :] - slot_positions[None, :, :]
            _, lengths = geometry.mic(delta.reshape(-1, 3))
            cost = lengths.reshape(len(reference_atoms), len(slot_sites))
            rows, columns = linear_sum_assignment(cost)
            if np.max(cost[rows, columns], initial=0.0) >= tolerance:
                failing = int(reference_atoms[rows[np.argmax(cost[rows, columns])]])
                raise ValueError(f"reference atom {failing} cannot be mapped to primitive")
            labels[reference_atoms[rows]] = slot_sites[columns]
            translations[reference_atoms[rows]] = slot_translations[columns]
            residuals[reference_atoms[rows]] = cost[rows, columns]
        # Constructing the index performs the global one-per-site-per-coset
        # validation and preserves the incoming reference order.
        PeriodicIndex(labels, translations, matrix)
        # Carry the verified frame mapping with every reference structure so
        # format writers and downstream FC2 materialization never reconstruct
        # identity from array position or floating-point coordinates.
        reference.arrays["primitive_index"] = labels.copy()
        reference.arrays["cell_translation"] = translations.copy()
        reference.arrays["primitive_scaled_position"] = primitive.get_scaled_positions()[labels]
        reference.info["mlfcs_supercell_matrix"] = matrix.tolist()
        return cls(primitive, reference, matrix, labels, translations, float(np.max(residuals)))

    @property
    def index(self) -> PeriodicIndex:
        return self._index

    def displacement(self, atoms: Atoms) -> np.ndarray:
        """Return MIC displacements without ever reordering a training frame."""
        if len(atoms) != len(self.reference):
            raise ValueError("training structure atom count differs from reference")
        if not np.array_equal(atoms.numbers, self.reference.numbers):
            raise ValueError("training structure atom order differs from reference")
        if not np.allclose(atoms.cell, self.reference.cell, atol=1e-7, rtol=0.0):
            raise ValueError("training structure cell differs from reference")
        vectors, _ = PeriodicGeometry(self.reference.cell, self.reference.pbc).mic(
            atoms.positions - self.reference.positions
        )
        return np.asarray(vectors)


def align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float = 1e-5,
) -> tuple[Atoms, float]:
    """Explicitly reorder ``atoms`` to ``reference`` and report the residual.

    This utility is intentionally separate from fitting and finite-difference
    APIs. It can be useful for independently produced snapshots, but never
    silently changes the labels supplied to a calculation.
    """
    if len(atoms) != len(reference):
        raise ValueError("structure atom count differs from reference")
    if not np.allclose(atoms.cell, reference.cell, atol=tolerance, rtol=0.0):
        raise ValueError("structure cell differs from reference")
    permutation = np.empty(len(reference), dtype=np.int32)
    maximum = 0.0
    geometry = PeriodicGeometry(reference.cell, reference.pbc)
    for number in np.unique(reference.numbers):
        target = np.flatnonzero(reference.numbers == number)
        source = np.flatnonzero(atoms.numbers == number)
        if len(target) != len(source):
            raise ValueError("structure chemical composition differs from reference")
        delta = atoms.positions[source][None, :, :] - reference.positions[target][:, None, :]
        _, lengths = geometry.mic(delta.reshape(-1, 3))
        cost = lengths.reshape(len(target), len(source))
        rows, columns = linear_sum_assignment(cost)
        maximum = max(maximum, float(np.max(cost[rows, columns], initial=0.0)))
        permutation[target[rows]] = source[columns]
    if maximum > tolerance:
        raise ValueError(
            f"structure cannot be aligned to reference within tolerance; maximum residual {maximum:.3e} Å"
        )
    aligned = atoms[permutation]
    aligned.info.update(atoms.info)
    return aligned, maximum


def _unique_distances(values: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> list[float]:
    result: list[float] = []
    for value in np.sort(values):
        if value > atol and (not result or not np.isclose(value, result[-1], atol=atol, rtol=rtol)):
            result.append(float(value))
    if not result:
        raise ValueError("no periodic neighbors found; supercell is too small")
    return result


__all__ = [
    "PeriodicGeometry",
    "PeriodicIndex",
    "StructureRelation",
    "_unique_distances",
    "align_structures",
    "normalize_supercell_matrix",
]
