"""Per-structure fitting equations retained alongside the normal system."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.log import get_logger
from mlfcs.force_constants import ForceConstants
from mlfcs.mapping import ClusterMap
from mlfcs.fitting.system import FitSystem, _readonly, _sample

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FitData:
    """Per-structure equations ``designs[i] @ parameters = forces[i]``.

    Retaining the raw equations grows memory with the number of training
    structures; the accumulated normal equations of
    :class:`~mlfcs.fitting.FitSystem` remain the default streaming path and
    are recovered from here through :meth:`normal_system`.
    """

    cluster_space: ClusterSpace
    designs: tuple[np.ndarray, ...]
    forces: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        if not self.designs or len(self.designs) != len(self.forces):
            raise ValueError("designs and forces must be nonempty and equally long")
        rows = len(self.designs[0])
        if rows == 0:
            raise ValueError("per-structure equations must have at least one row")
        blocks = []
        for design, force in zip(self.designs, self.forces, strict=True):
            if not np.all(np.isfinite(design)) or not np.all(np.isfinite(force)):
                raise ValueError("fit equations must be finite")
            blocks.append(
                (
                    _readonly(design, (rows, self.cluster_space.n_parameters)),
                    _readonly(force, (rows,)),
                )
            )
        object.__setattr__(self, "designs", tuple(block[0] for block in blocks))
        object.__setattr__(self, "forces", tuple(block[1] for block in blocks))

    def __reduce__(self):
        return type(self), (self.cluster_space, self.designs, self.forces)

    @classmethod
    def from_atoms(cls, cluster_map: ClusterMap, structures: Iterable[Atoms]) -> FitData:
        """Retain the per-structure equations of one streaming fit."""
        if isinstance(structures, (Atoms, np.ndarray)):
            raise TypeError("FitData.from_atoms() requires an iterable of ASE Atoms")
        from mlfcs.fitting.design import ForceDesign

        design = ForceDesign(cluster_map)
        workspace = design.allocate_workspace()
        supercell_positions = cluster_map.scaled_positions @ cluster_map.cell
        inverse_cell = np.linalg.inv(cluster_map.cell)
        designs: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        started = perf_counter()
        for count, atoms in enumerate(structures, start=1):
            displacement, values = _sample(
                cluster_map, atoms, count - 1, supercell_positions, inverse_cell
            )
            designs.append(design.matrix(displacement, workspace=workspace))
            forces.append(values.reshape(-1))
        if not designs:
            raise ValueError("at least one training structure is required")
        logger.info(
            "Retained %d per-structure equations in %.2f s", len(designs), perf_counter() - started
        )
        return cls(cluster_map.cluster_space, tuple(designs), tuple(forces))

    @property
    def n_structures(self) -> int:
        return len(self.designs)

    @property
    def n_equations(self) -> int:
        return sum(len(force) for force in self.forces)

    @property
    def n_parameters(self) -> int:
        return self.cluster_space.n_parameters

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stacked design matrix and force vector."""
        return np.concatenate(self.designs), np.concatenate(self.forces)

    def normal_system(self) -> FitSystem:
        """Compress the retained equations into one :class:`FitSystem`."""
        parameters = self.n_parameters
        matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
        rhs = np.zeros(parameters, dtype=np.float64)
        force_norm = 0.0
        for design, force in zip(self.designs, self.forces, strict=True):
            dsyrk(1.0, a=design, c=matrix, beta=1.0, trans=1, lower=0, overwrite_c=1)
            rhs += design.T @ force
            flattened = force.reshape(-1)
            force_norm += float(flattened @ flattened)
        upper = np.triu(np.asarray(matrix))
        matrix = upper + np.triu(upper, 1).T
        return FitSystem(
            cluster_space=self.cluster_space,
            matrix=matrix,
            rhs=rhs,
            force_norm=force_norm,
            n_equations=self.n_equations,
            n_structures=self.n_structures,
        )

    def force_constants(self, parameters: object) -> ForceConstants:
        """Bind a complete packed parameter vector to this cluster space."""
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.n_parameters,):
            raise ValueError(
                f"parameters must have shape {(self.n_parameters,)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters contain NaN or infinite values")
        return ForceConstants(
            self.cluster_space,
            {block.order: values[block.parameters] for block in self.cluster_space.blocks},
        )


__all__ = ["FitData"]
