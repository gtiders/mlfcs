from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic


@dataclass(frozen=True, slots=True)
class ReferenceSupercell:
    """User-ordered reference supercell mapped onto an explicit primitive cell."""

    primitive: Atoms
    reference: Atoms
    supercell_matrix: np.ndarray
    primitive_index: np.ndarray
    cell_translation: np.ndarray

    @classmethod
    def from_atoms(
        cls,
        primitive: Atoms,
        reference: Atoms,
        *,
        tolerance: float = 1e-5,
    ) -> ReferenceSupercell:
        source_reference = reference
        primitive = primitive.copy()
        reference = reference.copy()
        reference.calc = source_reference.calc
        primitive.wrap()
        reference.wrap()
        transform = np.asarray(reference.cell) @ np.linalg.inv(np.asarray(primitive.cell))
        matrix = np.rint(transform).astype(np.int32)
        if not np.allclose(transform, matrix, atol=tolerance, rtol=0.0):
            raise ValueError("reference is not an integer supercell of primitive")
        if abs(round(np.linalg.det(matrix))) * len(primitive) != len(reference):
            raise ValueError("supercell determinant and atom counts are inconsistent")

        primitive_fractional = primitive.get_scaled_positions()
        reference_fractional = reference.positions @ np.linalg.inv(np.asarray(primitive.cell))
        primitive_index = np.empty(len(reference), dtype=np.int32)
        translations = np.empty((len(reference), 3), dtype=np.int32)
        for atom, (number, position) in enumerate(
            zip(reference.numbers, reference_fractional, strict=True)
        ):
            matches = []
            for candidate in np.flatnonzero(primitive.numbers == number):
                difference = position - primitive_fractional[candidate]
                translation = np.rint(difference).astype(np.int32)
                matches.append(
                    (float(np.linalg.norm(difference - translation)), candidate, translation)
                )
            matches.sort(key=lambda item: item[0])
            if not matches or matches[0][0] >= tolerance:
                raise ValueError(f"reference atom {atom} cannot be mapped to primitive")
            _, primitive_index[atom], translations[atom] = matches[0]
        keys = {
            (int(index), *map(int, translation))
            for index, translation in zip(primitive_index, translations, strict=True)
        }
        if len(keys) != len(reference):
            raise ValueError("reference-to-primitive atom mapping is not one-to-one")
        return cls(primitive, reference, matrix, primitive_index, translations)

    @property
    def internal_permutation(self) -> np.ndarray:
        """Select reference atoms in MLFCS cell-major order without changing input data."""
        return np.lexsort(
            (
                self.primitive_index,
                self.cell_translation[:, 0],
                self.cell_translation[:, 1],
                self.cell_translation[:, 2],
            )
        )

    def displacement(self, atoms: Atoms) -> np.ndarray:
        if len(atoms) != len(self.reference):
            raise ValueError("training structure atom count differs from reference")
        if not np.array_equal(atoms.numbers, self.reference.numbers):
            raise ValueError("training structure atom order differs from reference")
        if not np.allclose(atoms.cell, self.reference.cell, atol=1e-7, rtol=0.0):
            raise ValueError("training structure cell differs from reference")
        vectors, _ = find_mic(
            atoms.positions - self.reference.positions,
            self.reference.cell,
            pbc=self.reference.pbc,
        )
        return np.asarray(vectors)


@dataclass(frozen=True, slots=True)
class FitDataset:
    displacements: np.ndarray
    forces: np.ndarray
    reference_forces: np.ndarray

    @classmethod
    def from_atoms(
        cls,
        geometry: ReferenceSupercell,
        structures: Sequence[Atoms],
    ) -> FitDataset:
        structures = tuple(structures)
        if not structures:
            raise ValueError("at least one training structure is required")
        reference_forces = _forces(geometry.reference, required=False)
        displacement = np.asarray([geometry.displacement(atoms) for atoms in structures])
        displacement -= displacement.mean(axis=1, keepdims=True)
        force = np.asarray([_forces(atoms, required=True) for atoms in structures])
        force -= force.mean(axis=1, keepdims=True)
        reference_forces -= reference_forces.mean(axis=0, keepdims=True)
        return cls(
            displacement,
            force - reference_forces[None, ...],
            reference_forces,
        )


def _forces(atoms: Atoms, *, required: bool) -> np.ndarray:
    if atoms.calc is not None and "forces" in atoms.calc.results:
        return np.asarray(atoms.calc.results["forces"], dtype=float).copy()
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float).copy()
    if required:
        raise ValueError("every training structure must provide forces")
    return np.zeros((len(atoms), 3), dtype=float)
