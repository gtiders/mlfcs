from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import find_mic


@dataclass(frozen=True, slots=True)
class SymmetryOperations:
    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    atom_permutations: np.ndarray
    symbol: str

    @classmethod
    def from_atoms(
        cls,
        primitive: Atoms,
        supercell: Atoms,
        *,
        symprec: float = 1e-5,
    ) -> SymmetryOperations:
        cell = (np.asarray(primitive.cell), primitive.get_scaled_positions(), primitive.numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise ValueError("spglib could not determine the crystal symmetry")
        rotations = np.asarray(dataset.rotations, dtype=np.int32)
        translations = np.asarray(dataset.translations, dtype=float)
        lattice = np.asarray(primitive.cell)
        inverse = np.linalg.inv(lattice)
        cartesian = np.asarray([inverse @ rotation.T @ lattice for rotation in rotations])
        permutations = _map_supercell(
            primitive,
            supercell,
            rotations,
            translations,
            symprec=symprec,
        )
        return cls(rotations, translations, cartesian, permutations, dataset.international.strip())

    @property
    def size(self) -> int:
        return len(self.rotations)


def _map_supercell(
    primitive: Atoms,
    supercell: Atoms,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    symprec: float,
) -> np.ndarray:
    primitive_inverse = np.linalg.inv(np.asarray(primitive.cell))
    target = supercell.get_positions()
    result = np.empty((len(rotations), len(supercell)), dtype=np.int32)
    for operation, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        fractional = supercell.positions @ primitive_inverse
        transformed = (fractional @ rotation.T + translation) @ primitive.cell
        for atom, position in enumerate(transformed):
            delta = target - position
            # ASE's MIC implementation handles skewed cells reliably.
            _, lengths = find_mic(delta, supercell.cell, pbc=True)
            same_species = supercell.numbers == supercell.numbers[atom]
            valid = np.flatnonzero(same_species & (lengths < symprec * 10.0))
            if len(valid) != 1:
                raise ValueError(
                    f"symmetry operation {operation} maps atom {atom} to {len(valid)} atoms"
                )
            result[operation, atom] = valid[0]
    return result
