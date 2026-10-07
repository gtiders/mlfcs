"""Ordered supercell displacements and physical target forces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.foundation.arrays import require_allocation
from mlfcs.geometry.periodic import PeriodicGeometry
from mlfcs.mapping import ClusterMap


@dataclass(frozen=True, slots=True, init=False)
class ForceDataset:
    """Displacement-force samples for one mapped periodic supercell.

    Each frame stores Cartesian atomic displacements relative to
    ``cluster_map.supercell_atoms`` and the corresponding target forces.
    Displacements are minimum-image vectors in angstrom; forces are in
    eV/angstrom. Frame and atom ordering are preserved as supplied. The dataset
    does not interpret how structures were generated and does not evaluate
    calculators.
    """

    cluster_map: ClusterMap
    displacements: np.ndarray
    forces: np.ndarray

    def __init__(self, cluster_map: ClusterMap, structures: Iterable[Atoms]):
        """Construct a dataset from ordered ASE snapshots carrying stored forces."""
        if not isinstance(cluster_map, ClusterMap):
            raise TypeError("cluster_map must be a ClusterMap")
        if isinstance(structures, (Atoms, np.ndarray)):
            raise TypeError("structures must be an iterable of ASE Atoms")
        geometry = PeriodicGeometry(cluster_map.cell)
        positions = cluster_map.supercell_atoms.positions
        displacements, forces = [], []
        for index, atoms in enumerate(structures):
            displacement, force = _sample(cluster_map, atoms, index, positions, geometry)
            require_allocation("force dataset", (index + 1, cluster_map.n_atoms, 3))
            displacements.append(displacement.copy())
            forces.append(force.copy())
        if not forces:
            raise ValueError("at least one force-bearing structure is required")
        self._initialize(cluster_map, np.stack(displacements), np.stack(forces))

    def _initialize(self, cluster_map, displacements, forces):
        """Attach the displacement and force arrays to this dataset."""
        displacements.setflags(write=False)
        forces.setflags(write=False)
        object.__setattr__(self, "cluster_map", cluster_map)
        object.__setattr__(self, "displacements", displacements)
        object.__setattr__(self, "forces", forces)

    @property
    def supercell_atoms(self) -> Atoms:
        """Reference supercell defining the displacement origin and atom order."""
        return self.cluster_map.supercell_atoms

    def subtract_forces(self, forces: object) -> ForceDataset:
        """Return a dataset with a force contribution removed.

        ``forces`` may contain one supercell force array or one array per frame.
        The returned dataset keeps the same displacements and replaces its
        target forces by

            ``F_target <- F_target - F_subtracted``.

        A single supercell array is broadcast across frames. The original
        dataset is unchanged.
        """
        values = np.asarray(forces, dtype=np.float64)
        if values.shape not in (self.forces.shape, self.forces.shape[1:]):
            raise ValueError("forces must have shape (frames, atoms, 3) or (atoms, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("forces must be finite")
        target = self.forces - values
        if not np.all(np.isfinite(target)):
            raise ValueError("force subtraction produced nonfinite values")
        result = object.__new__(type(self))
        result._initialize(self.cluster_map, self.displacements, target)
        return result


def _sample(cluster_map, atoms, index, supercell_positions, geometry):
    """Extract one displacement-force sample compatible with the mapped supercell."""
    if not isinstance(atoms, Atoms):
        raise TypeError(f"structure {index} is not an ASE Atoms object")
    if not np.array_equal(atoms.numbers, cluster_map.atomic_numbers):
        raise ValueError(f"structure {index} has a different atom sequence")
    if not np.all(atoms.pbc):
        raise ValueError(f"structure {index} must be periodic in all directions")
    cell_residual = float(np.max(np.linalg.norm(np.asarray(atoms.cell) - cluster_map.cell, axis=1)))
    if not np.isfinite(cell_residual) or cell_residual >= cluster_map.cluster_space.symprec:
        raise ValueError(f"structure {index} has a different supercell")
    if not np.all(np.isfinite(atoms.positions)):
        raise ValueError(f"structure {index} contains invalid positions")
    displacement, _ = geometry.minimum_image(atoms.positions - supercell_positions)
    if atoms.calc is None:
        raise ValueError(f"structure {index} has no stored ASE forces")
    forces = atoms.calc.get_property("forces", atoms, allow_calculation=False)
    if forces is None:
        raise ValueError(f"structure {index} has no stored ASE forces")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(f"structure {index} contains invalid forces")
    return displacement, forces
