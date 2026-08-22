"""Structure relations and periodic indexing independent of atom ordering.

The calculation frame is deliberately the user supplied reference supercell.
Integer translations are labels in the primitive lattice; their residues live
in ``Z^3 / Z^3 S`` for a general row-vector supercell matrix ``S``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from ase import Atoms
from ase.geometry import find_mic, minkowski_reduce
from scipy.optimize import linear_sum_assignment


def normalize_supercell_matrix(matrix: object) -> np.ndarray:
    """Return a validated integer 3x3 supercell matrix.

    A length-three input remains a convenient spelling for a diagonal matrix;
    it is not a separate representation in the core.
    """
    values = np.asarray(matrix)
    if values.shape == (3,):
        values = np.diag(values)
    if values.shape != (3, 3):
        raise ValueError("supercell_matrix must be three repeats or an integer 3x3 matrix")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=1e-10, rtol=0.0):
        raise ValueError("supercell_matrix must contain integers")
    determinant = round(float(np.linalg.det(rounded)))
    if determinant == 0:
        raise ValueError("supercell_matrix must be nonsingular")
    return rounded.astype(np.int32)


def _translation_label(translation: np.ndarray, matrix: np.ndarray) -> tuple[int, int, int]:
    """Canonical exact label of a primitive-lattice translation modulo ``S``."""
    determinant = abs(round(float(np.linalg.det(matrix))))
    adjugate = np.rint(np.linalg.det(matrix) * np.linalg.inv(matrix)).astype(np.int64)
    residue = np.mod(np.asarray(translation, dtype=np.int64) @ adjugate, determinant)
    return tuple(int(value) for value in residue)


def _coset_translations(matrix: np.ndarray) -> np.ndarray:
    """Enumerate one deterministic primitive-lattice translation per coset."""
    count = abs(round(float(np.linalg.det(matrix))))
    zero = np.zeros(3, dtype=np.int32)
    found: dict[tuple[int, int, int], np.ndarray] = {_translation_label(zero, matrix): zero}
    pending = deque([zero])
    generators = np.eye(3, dtype=np.int32)
    # The three primitive unit translations generate the finite quotient.
    # A breadth-first traversal visits every residue exactly once, avoiding
    # the former cubic bounding box for large determinant supercells.
    while pending and len(found) < count:
        current = pending.popleft()
        for generator in generators:
            candidate = current + generator
            residue = _translation_label(candidate, matrix)
            if residue not in found:
                found[residue] = candidate
                pending.append(candidate)
    if len(found) != count:  # pragma: no cover - defensive guard for malformed arithmetic
        raise RuntimeError("could not enumerate supercell translation cosets")
    return np.asarray(
        sorted(found.values(), key=lambda value: (value[2], value[1], value[0])), dtype=np.int32
    )


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
        return images[tied][locations[order]], unique[order]


@dataclass(frozen=True, slots=True)
class PeriodicIndex:
    """O(1) atom lookup using primitive site and a lattice-quotient residue."""

    primitive: np.ndarray
    translations: np.ndarray
    supercell_matrix: np.ndarray
    _atoms: dict[tuple[int, tuple[int, int, int]], int] = field(
        init=False, repr=False, compare=False
    )
    _translations_by_residue: dict[tuple[int, int, int], np.ndarray] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        primitive = np.asarray(self.primitive, dtype=np.int32)
        translations = np.asarray(self.translations, dtype=np.int32)
        matrix = normalize_supercell_matrix(self.supercell_matrix)
        if primitive.ndim != 1 or translations.shape != (len(primitive), 3):
            raise ValueError("primitive labels and translations have incompatible shapes")
        if (
            len(primitive) == 0
            or primitive.min() != 0
            or not np.array_equal(np.unique(primitive), np.arange(int(primitive.max()) + 1))
        ):
            raise ValueError("primitive labels must be contiguous and start at zero")
        atoms: dict[tuple[int, tuple[int, int, int]], int] = {}
        translations_by_residue: dict[tuple[int, int, int], np.ndarray] = {}
        for atom, (site, translation) in enumerate(zip(primitive, translations, strict=True)):
            residue = _translation_label(translation, matrix)
            key = (int(site), residue)
            if key in atoms:
                raise ValueError("more than one reference atom has the same periodic label")
            atoms[key] = atom
            translations_by_residue.setdefault(residue, translation.copy())
        expected = self.n_primitive * abs(round(float(np.linalg.det(matrix))))
        if len(atoms) != expected:
            raise ValueError("reference does not contain exactly one atom per primitive-site coset")
        object.__setattr__(self, "primitive", primitive)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "supercell_matrix", matrix)
        object.__setattr__(self, "_atoms", atoms)
        object.__setattr__(self, "_translations_by_residue", translations_by_residue)

    @property
    def n_primitive(self) -> int:
        return int(np.max(self.primitive)) + 1

    @property
    def n_cells(self) -> int:
        return abs(round(float(np.linalg.det(self.supercell_matrix))))

    def residue(self, translation: np.ndarray) -> tuple[int, int, int]:
        return _translation_label(np.asarray(translation, dtype=np.int64), self.supercell_matrix)

    def canonical_translation(self, translation: np.ndarray) -> np.ndarray:
        """Reference-frame representative of a primitive-lattice translation residue."""
        return self._translations_by_residue[self.residue(translation)].copy()

    def atom(self, primitive: int, translation: np.ndarray) -> int:
        try:
            return self._atoms[(int(primitive), self.residue(translation))]
        except KeyError as error:
            raise ValueError("unknown primitive site or translation residue") from error

    def translate_atom(self, atom: int, shift: np.ndarray) -> int:
        return self.atom(int(self.primitive[atom]), self.translations[atom] + np.asarray(shift))

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
        if abs(round(float(np.linalg.det(matrix)))) * len(primitive) != len(reference):
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
        return PeriodicIndex(self.primitive_index, self.cell_translation, self.supercell_matrix)

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


def make_supercell(atoms: Atoms, supercell_matrix: object) -> tuple[Atoms, PeriodicIndex]:
    """Build a general integer supercell, with deterministic reference ordering."""
    if not np.all(atoms.pbc):
        raise ValueError("force constants require periodic boundary conditions")
    primitive = atoms.copy()
    primitive.wrap()
    matrix = normalize_supercell_matrix(supercell_matrix)
    translations = _coset_translations(matrix)
    positions = np.concatenate(
        [primitive.positions + shift @ primitive.cell for shift in translations]
    )
    supercell = Atoms(
        numbers=np.tile(primitive.numbers, len(translations)),
        positions=positions,
        cell=matrix @ primitive.cell,
        pbc=True,
    )
    primitive_indices = np.tile(np.arange(len(primitive), dtype=np.int32), len(translations))
    atom_translations = np.repeat(translations, len(primitive), axis=0)
    supercell.arrays["primitive_index"] = primitive_indices
    supercell.arrays["cell_translation"] = atom_translations
    supercell.arrays["primitive_scaled_position"] = np.tile(
        primitive.get_scaled_positions(), (len(translations), 1)
    )
    supercell.info["mlfcs_supercell_matrix"] = matrix.tolist()
    return supercell, PeriodicIndex(primitive_indices, atom_translations, matrix)


def build_supercell(atoms: Atoms, supercell_matrix: object) -> Atoms:
    """Build an ASE supercell in MLFCS's deterministic reference frame.

    Diagonal three-tuples preserve the historic cell-major expansion order;
    arbitrary integral 3x3 matrices use the same general construction.  The
    returned structure carries MLFCS mapping metadata and can be supplied as
    a reference frame to finite differences, fitting, SSCHA, or export.
    """
    return make_supercell(atoms, supercell_matrix)[0]


def neighbor_shell_cutoff(
    supercell: Atoms, index: PeriodicIndex, shell: int, *, report: bool = True
) -> float:
    if shell < 1:
        raise ValueError("neighbor shell must be positive")
    distances = PeriodicGeometry(supercell.cell, supercell.pbc).pair_distances(supercell.positions)
    maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index, distances=distances)
    if report:
        print(
            f"Supercell neighbor limit: maximum shell = {maximum_shell}, maximum cutoff radius = {maximum_radius:.10f} Å"
        )
    if shell > maximum_shell:
        raise ValueError(
            f"neighbor shell {shell} exceeds this supercell's enumerable maximum of {maximum_shell} (cutoff radius {maximum_radius:.10f} Å)"
        )
    candidates = []
    for site in range(index.n_primitive):
        unique = _unique_distances(distances[index.representative(site)])
        candidates.append(
            unique[-1] * 1.1 if len(unique) <= shell else (unique[shell - 1] + unique[shell]) / 2.0
        )
    selected = float(max(candidates))
    if report:
        print(f"Selected neighbor cutoff: shell = {shell}, cutoff radius = {selected:.10f} Å")
    return selected


def neighbor_shell_limit(
    supercell: Atoms, index: PeriodicIndex, *, distances: np.ndarray | None = None
) -> tuple[int, float]:
    if distances is None:
        distances = PeriodicGeometry(supercell.cell, supercell.pbc).pair_distances(
            supercell.positions
        )
    shells = [
        _unique_distances(distances[index.representative(site)])
        for site in range(index.n_primitive)
    ]
    maximum_shell = min(map(len, shells))
    if maximum_shell < 1:
        raise ValueError("supercell is too small to contain a reliable neighbor shell")
    boundaries = [
        values[-1] * 1.1
        if len(values) <= maximum_shell
        else (values[maximum_shell - 1] + values[maximum_shell]) / 2.0
        for values in shells
    ]
    return maximum_shell, float(max(boundaries))


def resolve_cutoff(
    supercell: Atoms, index: PeriodicIndex, cutoff: float | None, *, report: bool = True
) -> float:
    if cutoff is None:
        maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index)
        if report:
            print(
                f"Supercell neighbor limit: maximum shell = {maximum_shell}, maximum cutoff radius = {maximum_radius:.10f} Å"
            )
            print(f"Selected maximum cutoff radius: {maximum_radius:.10f} Å")
        return maximum_radius
    value = float(cutoff)
    if value < 0 and value.is_integer():
        return neighbor_shell_cutoff(supercell, index, -int(value), report=report)
    if value <= 0:
        raise ValueError("cutoff must be a positive distance or negative integer shell")
    return value


def _unique_distances(values: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> list[float]:
    result: list[float] = []
    for value in np.sort(values):
        if value > atol and (not result or not np.isclose(value, result[-1], atol=atol, rtol=rtol)):
            result.append(float(value))
    if not result:
        raise ValueError("no periodic neighbors found; supercell is too small")
    return result
