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


def _validate_symprec(symprec: object) -> float:
    """Return a usable length precision, rejecting non-finite and non-positive values."""
    value = float(symprec)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"symprec must be a finite positive length in angstrom, got {symprec!r}")
    return value


def _cell_residual(reference_cell: np.ndarray, matrix: np.ndarray, primitive_cell: np.ndarray) -> float:
    """Return the largest lattice mismatch per primitive lattice coefficient, in angstrom.

    Dividing by the row sum of ``|S|`` expresses the error as angstrom per primitive lattice
    coefficient, so one ``symprec`` means the same thing for a 1x1x1 cell and for a large repeat.
    This normalization is not a second threshold.
    """
    rebuilt = matrix @ primitive_cell
    difference = reference_cell - rebuilt
    lengths = np.linalg.norm(difference, axis=1)
    scale = np.maximum(1.0, np.sum(np.abs(matrix), axis=1))
    return float(np.max(lengths / scale))


def _attach_frame_metadata(
    reference: Atoms, labels: np.ndarray, translations: np.ndarray, matrix: np.ndarray
) -> None:
    """Record the verified frame mapping on the reference structure."""
    reference.arrays["primitive_index"] = labels.copy()
    reference.arrays["cell_translation"] = translations.copy()
    reference.arrays["primitive_scaled_position"] = reference.get_scaled_positions()[labels]
    reference.info["mlfcs_supercell_matrix"] = np.asarray(matrix).tolist()


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
    #: The single length precision of this relation, in angstrom.  It is the same value the
    #: symmetry identification and the lattice/atom mapping used, and the only tolerance the
    #: fixed-cell training-frame check reads.
    symprec: float
    #: Largest per-primitive-lattice-vector lattice mismatch, in angstrom.
    cell_residual: float
    _index: PeriodicIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_index",
            PeriodicIndex(self.primitive_index, self.cell_translation, self.supercell_matrix),
        )

    @classmethod
    def identity(cls, primitive: Atoms, *, symprec: float) -> StructureRelation:
        """Return the relation of a primitive cell used as its own reference.

        This is the canonical exact-R case: no floating-point mapping is needed, so the matrix is
        the identity and every atom maps onto itself.  ``symprec`` is still recorded, because the
        fixed-cell training-frame check uses it.
        """
        symprec = _validate_symprec(symprec)
        if not np.all(primitive.pbc):
            raise ValueError("force constants require a periodic primitive structure")
        cell = primitive.copy()
        cell.wrap()
        labels = np.arange(len(cell), dtype=np.int32)
        translations = np.zeros((len(cell), 3), dtype=np.int32)
        matrix = np.eye(3, dtype=np.int64)
        _attach_frame_metadata(cell, labels, translations, matrix)
        return cls(
            cell,
            cell.copy(),
            matrix,
            labels,
            translations,
            0.0,
            symprec,
            0.0,
        )

    @classmethod
    def from_atoms(
        cls, primitive: Atoms, reference: Atoms, *, symprec: float
    ) -> StructureRelation:
        """Verify that an explicit reference is an integer supercell of the primitive.

        ``symprec`` is the only length precision here, in angstrom: it accepts the lattice
        relation, the atom mapping and, later, fixed-cell training frames.  A dimensionless
        matrix difference is never compared with it.
        """
        symprec = _validate_symprec(symprec)
        if not np.all(primitive.pbc) or not np.all(reference.pbc):
            raise ValueError("force constants require periodic primitive and reference structures")
        source_reference = reference
        primitive = primitive.copy()
        reference = reference.copy()
        reference.calc = source_reference.calc
        primitive.wrap()
        reference.wrap()
        transform = np.asarray(reference.cell) @ np.linalg.inv(np.asarray(primitive.cell))
        # This is the only place allowed to turn two floating-point cells into a candidate
        # integer matrix; the discrete tools take over from here.
        candidate = np.rint(transform).astype(np.int64)
        matrix = normalize_supercell_matrix(candidate)
        cell_residual = _cell_residual(
            np.asarray(reference.cell, dtype=float), matrix, np.asarray(primitive.cell, dtype=float)
        )
        if cell_residual >= symprec:
            raise ValueError(
                f"reference is not an integer supercell of primitive within symprec: lattice "
                f"residual {cell_residual:.6e} angstrom per primitive lattice coefficient against "
                f"symprec {symprec:.6e} angstrom, candidate matrix {matrix.tolist()}"
            )
        if abs(determinant_3x3(matrix)) * len(primitive) != len(reference):
            raise ValueError(
                f"supercell determinant and atom counts are inconsistent: |det S| = "
                f"{abs(determinant_3x3(matrix))} times {len(primitive)} primitive atoms is not "
                f"the {len(reference)} reference atoms of matrix {matrix.tolist()}"
            )
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
            if np.max(cost[rows, columns], initial=0.0) >= symprec:
                worst = int(np.argmax(cost[rows, columns]))
                failing = int(reference_atoms[rows[worst]])
                raise ValueError(
                    f"reference atom {failing} cannot be mapped onto a primitive site plus an "
                    f"integer lattice vector: largest mapping residual "
                    f"{float(cost[rows[worst], columns[worst]]):.6e} angstrom against symprec "
                    f"{symprec:.6e} angstrom"
                )
            labels[reference_atoms[rows]] = slot_sites[columns]
            translations[reference_atoms[rows]] = slot_translations[columns]
            residuals[reference_atoms[rows]] = cost[rows, columns]
        # Constructing the index performs the global one-per-site-per-coset
        # validation and preserves the incoming reference order.
        PeriodicIndex(labels, translations, matrix)
        # Carry the verified frame mapping with every reference structure so
        # format writers and downstream FC2 materialization never reconstruct
        # identity from array position or floating-point coordinates.
        _attach_frame_metadata(reference, labels, translations, matrix)
        return cls(
            primitive,
            reference,
            matrix,
            labels,
            translations,
            float(np.max(residuals)),
            symprec,
            cell_residual,
        )

    @property
    def index(self) -> PeriodicIndex:
        return self._index

    def displacement(self, atoms: Atoms) -> np.ndarray:
        """Return MIC displacements without ever reordering a training frame."""
        if len(atoms) != len(self.reference):
            raise ValueError("training structure atom count differs from reference")
        if not np.array_equal(atoms.numbers, self.reference.numbers):
            raise ValueError("training structure atom order differs from reference")
        frame_residual = float(
            np.max(np.linalg.norm(np.asarray(atoms.cell) - np.asarray(self.reference.cell), axis=1))
        )
        if frame_residual >= self.symprec:
            raise ValueError(
                f"training structure cell differs from reference: lattice residual "
                f"{frame_residual:.6e} angstrom against symprec {self.symprec:.6e} angstrom; a "
                "varying-cell training frame is a different physical model"
            )
        vectors, _ = PeriodicGeometry(self.reference.cell, self.reference.pbc).mic(
            atoms.positions - self.reference.positions
        )
        return np.asarray(vectors)


__all__ = [
    "StructureRelation",
    "normalize_supercell_matrix",
]
