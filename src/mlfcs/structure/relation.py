"""Verified relationships between primitive and reference structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from mlfcs.structure.integer_lattice import (
    IntegerLatticeQuotient,
    determinant_3x3,
    normalize_supercell_matrix,
)
from mlfcs.structure.periodic_geometry import PeriodicGeometry
from mlfcs.structure.supercell_mapping import PeriodicIndex


def _coset_translations(matrix: np.ndarray) -> np.ndarray:
    return IntegerLatticeQuotient(matrix).representatives.copy()


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


__all__ = [
    "StructureRelation",
    "align_structures",
    "normalize_supercell_matrix",
]
