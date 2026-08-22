from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.structure.geometry import StructureRelation


@dataclass(frozen=True, slots=True)
class FitDataset:
    displacements: np.ndarray
    forces: np.ndarray
    reference_forces: np.ndarray
    center_of_mass_displacements: np.ndarray
    net_forces: np.ndarray

    @classmethod
    def from_atoms(
        cls,
        geometry: StructureRelation,
        structures: Sequence[Atoms],
    ) -> FitDataset:
        structures = tuple(structures)
        if not structures:
            raise ValueError("at least one training structure is required")
        reference_forces = _forces(geometry.reference, required=False)
        displacement = np.asarray([geometry.displacement(atoms) for atoms in structures])
        force = np.asarray([_forces(atoms, required=True) for atoms in structures])
        return cls(
            displacement,
            force - reference_forces[None, ...],
            reference_forces,
            displacement.mean(axis=1),
            force.sum(axis=1),
        )


def _forces(atoms: Atoms, *, required: bool) -> np.ndarray:
    if atoms.calc is not None and "forces" in atoms.calc.results:
        values = np.asarray(atoms.calc.results["forces"], dtype=float)
    elif "forces" in atoms.arrays:
        values = np.asarray(atoms.arrays["forces"], dtype=float)
    elif required:
        raise ValueError("every training structure must provide forces")
    else:
        values = np.zeros((len(atoms), 3), dtype=float)
    expected = (len(atoms), 3)
    if values.shape != expected:
        raise ValueError(f"forces must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("forces contain NaN or infinite values")
    return values.copy()
