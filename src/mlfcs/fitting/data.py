"""Retained force-design matrices for solvers that use the original equations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.supercell import ClusterMap


@dataclass(frozen=True, slots=True)
class FitData:
    """Per-structure equations ``designs[i] @ parameters = forces[i]``.

    Retaining these arrays is opt-in because their storage grows with the
    number of training structures. No normal matrix is formed at construction.
    """

    space: ClusterSpace
    designs: tuple[np.ndarray, ...]
    forces: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        if len(self.designs) == 0 or len(self.designs) != len(self.forces):
            raise ValueError(
                "designs and forces must contain the same nonzero number of structures"
            )
        rows = np.asarray(self.designs[0]).shape[0]
        designs = []
        forces = []
        for index, (design, force) in enumerate(zip(self.designs, self.forces, strict=True)):
            matrix = np.array(design, dtype=np.float64, copy=True, order="C")
            vector = np.array(force, dtype=np.float64, copy=True, order="C")
            if matrix.shape != (rows, self.space.n_parameters) or vector.shape != (rows,):
                raise ValueError(f"structure {index} has inconsistent design or force shape")
            if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
                raise ValueError(f"structure {index} has nonfinite design or forces")
            matrix.setflags(write=False)
            vector.setflags(write=False)
            designs.append(matrix)
            forces.append(vector)
        object.__setattr__(self, "designs", tuple(designs))
        object.__setattr__(self, "forces", tuple(forces))

    def __reduce__(self):
        return type(self), (self.space, self.designs, self.forces)

    @classmethod
    def from_atoms(cls, mapping: ClusterMap, structures: Iterable[Atoms]) -> FitData:
        """Retain the same force equations used by the streamed fit path."""
        if not isinstance(mapping, ClusterMap):
            raise TypeError("mapping must be a ClusterMap")
        from mlfcs.fitting.build import _design_samples
        from mlfcs.fitting.design import ForceDesign

        samples = tuple(_design_samples(mapping, structures, ForceDesign(mapping)))
        if not samples:
            raise ValueError("at least one training structure is required")
        designs, forces = zip(*samples, strict=True)
        return cls(mapping.space, designs, forces)

    @property
    def n_structures(self) -> int:
        return len(self.designs)

    @property
    def n_equations(self) -> int:
        return sum(len(force) for force in self.forces)

    @property
    def n_parameters(self) -> int:
        return self.space.n_parameters

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stacked matrix and force vector for direct least squares."""
        return np.concatenate(self.designs), np.concatenate(self.forces)

    def normal_system(self):
        """Compress the retained equations into the usual streamed FitSystem."""
        from mlfcs.fitting.system import FitSystem

        count = self.n_parameters
        matrix = np.zeros((count, count), dtype=np.float64, order="F")
        rhs = np.zeros(count, dtype=np.float64)
        force_norm = 0.0
        for design, force in zip(self.designs, self.forces, strict=True):
            dsyrk(1.0, a=design, c=matrix, beta=1.0, trans=1, lower=0, overwrite_c=1)
            rhs += design.T @ force
            force_norm += float(force @ force)
        upper = np.triu(np.asarray(matrix))
        matrix = upper + np.triu(upper, 1).T
        return FitSystem(self.space, matrix, rhs, force_norm, self.n_equations, self.n_structures)

    def force_constants(self, parameters: object) -> ForceConstants:
        """Bind parameters from any solver without constructing a normal system."""
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.n_parameters,):
            raise ValueError(
                f"parameters must have shape {(self.n_parameters,)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters contain NaN or infinite values")
        return ForceConstants(
            self.space,
            {block.order: values[block.parameters] for block in self.space.blocks},
        )


__all__ = ["FitData"]
