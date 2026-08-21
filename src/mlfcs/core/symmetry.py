from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.core.integer_lattice import adjugate_3x3, determinant_3x3


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
        translated = np.einsum(
            "aj,okj->oak", atom_translations, rotations, optimize=True
        )
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
