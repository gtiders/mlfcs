from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms

from mlfcs.structure.integer_lattice import adjugate_3x3, determinant_3x3


@dataclass(frozen=True, slots=True)
class SymmetryOperations:
    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    atom_permutations: np.ndarray
    symbol: str

    @classmethod
    def from_primitive_operations(cls, operations, index) -> SymmetryOperations:
        """Realize exact primitive affine symmetry operations on one reference."""
        matrix = index.supercell_matrix.astype(np.int64)
        determinant = determinant_3x3(matrix)
        adjugate = adjugate_3x3(matrix)
        compatible = []
        for operation, rotation in enumerate(operations.rotations):
            numerator = matrix @ rotation.T @ adjugate
            if np.all(np.mod(numerator, determinant) == 0):
                compatible.append(operation)
        selected = np.asarray(compatible, dtype=np.int32)
        rotations = operations.rotations[selected]
        translations = operations.translations[selected]
        cartesian_rotations = operations.cartesian_rotations[selected]
        site_permutations = operations.site_permutations[selected]
        site_shifts = operations.site_shifts[selected]
        atom_sites = index.primitive
        atom_translations = index.translations.astype(np.int64)
        sites = site_permutations[:, atom_sites]
        translated = np.einsum("aj,okj->oak", atom_translations, rotations, optimize=True)
        translated += site_shifts[:, atom_sites]
        permutations = index.atom_many(sites, translated).astype(np.int32)
        return cls(
            rotations,
            translations,
            cartesian_rotations,
            permutations,
            operations.symbol,
        )

    @property
    def size(self) -> int:
        return len(self.rotations)


@dataclass(frozen=True, slots=True)
class PrimitiveSymmetryOperations:
    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    symbol: str

    @classmethod
    def from_atoms(cls, primitive: Atoms, *, symprec: float) -> PrimitiveSymmetryOperations:
        cell = (
            np.asarray(primitive.cell),
            primitive.get_scaled_positions(),
            primitive.numbers,
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise ValueError("spglib could not determine the primitive crystal symmetry")
        rotations = np.asarray(dataset.rotations, dtype=np.int32)
        translations = np.asarray(dataset.translations, dtype=float)
        lattice = np.asarray(primitive.cell)
        inverse = np.linalg.inv(lattice)
        cartesian = np.asarray([inverse @ rotation.T @ lattice for rotation in rotations])
        scaled = primitive.get_scaled_positions(wrap=False)
        permutations_array = np.empty((len(rotations), len(primitive)), dtype=np.int32)
        shifts = np.empty((len(rotations), len(primitive), 3), dtype=np.int32)
        for operation, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        ):
            transformed = scaled @ rotation.T + translation
            for site, position in enumerate(transformed):
                candidates = np.flatnonzero(primitive.numbers == primitive.numbers[site])
                differences = position - scaled[candidates]
                integers = np.rint(differences).astype(np.int32)
                residuals = np.linalg.norm((differences - integers) @ lattice, axis=1)
                selected = np.flatnonzero(residuals < symprec * 10.0)
                if len(selected) != 1:
                    raise ValueError(
                        f"symmetry operation {operation} maps primitive site {site} "
                        f"to {len(selected)} sites"
                    )
                location = int(selected[0])
                permutations_array[operation, site] = int(candidates[location])
                shifts[operation, site] = integers[location]
        return cls(
            rotations,
            translations,
            cartesian,
            permutations_array,
            shifts,
            dataset.international.strip(),
        )

    @property
    def size(self) -> int:
        return len(self.rotations)

    def transform_label(
        self, operation: int, label: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        site = int(label[0])
        translation = np.asarray(label[1:], dtype=np.int64)
        transformed = translation @ self.rotations[operation].T + self.site_shifts[operation, site]
        return (
            int(self.site_permutations[operation, site]),
            *(int(value) for value in transformed),
        )
