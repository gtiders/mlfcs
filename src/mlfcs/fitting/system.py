"""Force-fitting sufficient systems and training data ingestion."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs._arrays import require_allocation
from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.log import get_logger
from mlfcs.errors import UnobservedParameterError
from mlfcs.fitting.design import ForceDesign
from mlfcs.force_constants import ForceConstants
from mlfcs.mapping import ClusterMap


def _readonly(values: object, shape: tuple[int, ...]) -> np.ndarray:
    require_allocation("fit buffer", shape)
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {result.shape}")
    result.setflags(write=False)
    return result


def _bytes(values: np.ndarray) -> bytes:
    array = np.ascontiguousarray(values, dtype=np.float64)
    return array.astype(array.dtype.newbyteorder(">"), copy=False).tobytes()


@dataclass(frozen=True, slots=True)
class FitSystem:
    """The reusable normal system of one force-only least-squares problem."""

    cluster_space: ClusterSpace
    matrix: np.ndarray
    rhs: np.ndarray
    force_norm: float
    n_equations: int
    n_structures: int

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        count = self.cluster_space.n_parameters
        matrix = _readonly(self.matrix, (count, count))
        rhs = _readonly(self.rhs, (count,))
        force_norm = float(self.force_norm)
        n_equations = int(self.n_equations)
        n_structures = int(self.n_structures)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
            raise ValueError("fit matrix and right-hand side must be finite")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("fit matrix must be exactly symmetric")
        if np.any(np.diag(matrix) < 0.0):
            raise ValueError("fit matrix diagonal must be nonnegative")
        zero = np.diag(matrix) == 0.0
        if np.any(rhs[zero] != 0.0):
            raise ValueError("a zero fit-matrix column has a nonzero right-hand side")
        if not np.isfinite(force_norm) or force_norm < 0.0:
            raise ValueError("force_norm must be finite and nonnegative")
        if n_equations < 0 or n_structures < 0:
            raise ValueError("fit-system counts must be nonnegative")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "force_norm", force_norm)
        object.__setattr__(self, "n_equations", n_equations)
        object.__setattr__(self, "n_structures", n_structures)

    def __reduce__(self):
        return type(self), (
            self.cluster_space,
            self.matrix,
            self.rhs,
            self.force_norm,
            self.n_equations,
            self.n_structures,
        )

    @classmethod
    def from_atoms(cls, cluster_map: ClusterMap, structures: Iterable[Atoms]) -> FitSystem:
        """Stream evaluated ASE structures into the optimized normal system."""
        if not isinstance(cluster_map, ClusterMap):
            raise TypeError("mapping must be a ClusterMap")

        return _build_system(cluster_map, structures)

    @property
    def n_parameters(self) -> int:
        return self.cluster_space.n_parameters

    @property
    def unobserved_parameters(self) -> tuple[int, ...]:
        """Return columns that are exactly zero in the training design."""
        return tuple(int(index) for index in np.flatnonzero(np.diag(self.matrix) == 0.0))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.cluster_space.fingerprint.encode("ascii"))
        digest.update(struct.pack(">qqd", self.n_equations, self.n_structures, self.force_norm))
        digest.update(_bytes(self.matrix))
        digest.update(_bytes(self.rhs))
        return digest.hexdigest()

    def parameter_name(self, parameter: int) -> str:
        """Return the physical layout address of one packed parameter."""
        parameter = int(parameter)
        if not 0 <= parameter < self.n_parameters:
            raise IndexError("parameter is outside the cluster space")
        offsets = self.cluster_space.parameter_offsets
        orbit_index = int(np.searchsorted(offsets, parameter, side="right") - 1)
        component = parameter - int(offsets[orbit_index])
        for block in self.cluster_space.blocks:
            if block.parameters.start <= parameter < block.parameters.stop:
                return (
                    f"FC{block.order} orbit {orbit_index - block.orbits.start} "
                    f"component {component}"
                )
        raise RuntimeError("cluster-space parameter layout is inconsistent")

    def _require_observed(self) -> None:
        missing = self.unobserved_parameters
        if not missing:
            return
        shown = ", ".join(self.parameter_name(index) for index in missing[:12])
        remainder = "" if len(missing) <= 12 else f", and {len(missing) - 12} more"
        raise UnobservedParameterError(
            f"{len(missing)} parameters are absent from the training design: "
            f"{shown}{remainder}. Add structurally different displacements or merge another "
            "FitSystem. Regularization cannot recover a parameter absent from the design."
        )

    @property
    def column_scale(self) -> np.ndarray:
        """Return exact inverse column norms after rejecting zero columns."""
        self._require_observed()
        return 1.0 / np.sqrt(np.diag(self.matrix))

    def solve(self, *, rtol: float = 1e-8, max_steps: int = 1000) -> np.ndarray:
        """Return physical parameters from the scaled MINRES default solver."""
        self._require_observed()
        from mlfcs.fitting.solver import solve

        return solve(self.matrix, self.rhs, rtol=rtol, max_steps=max_steps)

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

    def residual(self, parameters: object) -> float:
        """Return the force residual norm represented by this sufficient system."""
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.n_parameters,):
            raise ValueError(
                f"parameters must have shape {(self.n_parameters,)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        model = float(values @ self.matrix @ values)
        cross = float(values @ self.rhs)
        squared = model - 2.0 * cross + self.force_norm
        cancellation = (
            32.0 * np.finfo(float).eps * (abs(model) + 2.0 * abs(cross) + abs(self.force_norm))
        )
        if not np.isfinite(squared) or not np.isfinite(cancellation):
            raise ValueError("fit-system residual is not finite")
        if abs(squared) <= cancellation:
            squared = 0.0
        if squared < 0.0:
            raise ValueError(
                "fit-system residual squared is negative beyond roundoff; "
                "matrix, rhs, and force_norm are inconsistent"
            )
        return float(np.sqrt(squared))

    def rmse(self, parameters: object) -> float:
        residual = self.residual(parameters)
        if self.n_equations:
            return residual / np.sqrt(self.n_equations)
        if residual != 0.0:
            raise ValueError("fit system has a nonzero residual but no equations")
        return 0.0

    def relative_error(self, parameters: object) -> float:
        residual = self.residual(parameters)
        if self.force_norm > 0.0:
            return residual / np.sqrt(self.force_norm)
        return 0.0 if residual == 0.0 else float("inf")

    def __add__(self, other: object) -> FitSystem:
        if not isinstance(other, FitSystem):
            return NotImplemented
        if self.cluster_space.fingerprint != other.cluster_space.fingerprint:
            raise ValueError("fit systems use different cluster spaces")
        return FitSystem(
            cluster_space=self.cluster_space,
            matrix=self.matrix + other.matrix,
            rhs=self.rhs + other.rhs,
            force_norm=self.force_norm + other.force_norm,
            n_equations=self.n_equations + other.n_equations,
            n_structures=self.n_structures + other.n_structures,
        )


logger = get_logger(__name__)


def _sample(
    cluster_map: ClusterMap,
    atoms: Atoms,
    index: int,
    supercell_positions: np.ndarray,
    inverse_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(atoms, Atoms):
        raise TypeError(f"training structure {index} is not an ASE Atoms object")
    supercell = cluster_map
    if not np.array_equal(atoms.numbers, supercell.atomic_numbers):
        raise ValueError(f"training structure {index} has a different atom sequence")
    if not np.array_equal(atoms.pbc, np.ones(3, dtype=bool)):
        raise ValueError(f"training structure {index} must be periodic in all directions")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    cell_residual = float(np.max(np.linalg.norm(cell - supercell.cell, axis=1)))
    symprec = cluster_map.cluster_space.symprec
    if cell_residual >= symprec:
        raise ValueError(
            f"training structure {index} has cell residual {cell_residual:.10g} Å, "
            f"not below symprec {symprec:.10g} Å"
        )
    scaled = (atoms.positions - supercell_positions) @ inverse_cell
    scaled -= np.rint(scaled)
    displacement = scaled @ supercell.cell
    if not np.all(np.isfinite(displacement)):
        raise ValueError(f"training structure {index} contains invalid positions")
    if atoms.calc is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = atoms.calc.get_property("forces", atoms, allow_calculation=False)
    if forces is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(f"training structure {index} contains invalid forces")
    return displacement, forces


def _build_system(cluster_map: ClusterMap, structures: Iterable[Atoms]):
    """Build a :class:`FitSystem` without retaining the training structures."""

    if isinstance(structures, (Atoms, np.ndarray)):
        raise TypeError("FitSystem.from_atoms() requires an iterable of ASE Atoms")
    design = ForceDesign(cluster_map)
    workspace = design.allocate_workspace()
    parameters = design.n_parameters
    require_allocation("fit normal matrix", (parameters, parameters))
    require_allocation("fit right hand side", (parameters,))
    matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
    rhs = np.zeros(parameters, dtype=np.float64)
    force_norm = 0.0
    count = 0
    started = perf_counter()
    supercell_positions = cluster_map.scaled_positions @ cluster_map.cell
    inverse_cell = np.linalg.inv(cluster_map.cell)
    for count, atoms in enumerate(structures, start=1):
        displacement, forces = _sample(
            cluster_map, atoms, count - 1, supercell_positions, inverse_cell
        )
        values = design.matrix(displacement, workspace=workspace)
        dsyrk(
            1.0,
            a=values,
            c=matrix,
            beta=1.0,
            trans=1,
            lower=0,
            overwrite_c=1,
        )
        flattened = forces.reshape(-1)
        rhs += values.T @ flattened
        force_norm += float(flattened @ flattened)
    if count == 0:
        raise ValueError("at least one training structure is required")
    upper = np.triu(np.asarray(matrix))
    matrix = upper + np.triu(upper, 1).T
    logger.info(
        "Built %d-parameter fit system from %d structures in %.2f s",
        parameters,
        count,
        perf_counter() - started,
    )
    return FitSystem(
        cluster_space=cluster_map.cluster_space,
        matrix=matrix,
        rhs=rhs,
        force_norm=force_norm,
        n_equations=count * design.rows,
        n_structures=count,
    )


__all__ = ["FitSystem"]
