"""Public sufficient system for a force-constant fit."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.errors import UnobservedParameterError
from mlfcs.force_constants import ForceConstants
from mlfcs.supercell import ClusterMap


def _readonly(values: object, shape: tuple[int, ...]) -> np.ndarray:
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

    space: ClusterSpace
    matrix: np.ndarray
    rhs: np.ndarray
    force_norm: float
    n_equations: int
    n_structures: int

    def __post_init__(self) -> None:
        if not isinstance(self.space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        count = self.space.n_parameters
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
            self.space,
            self.matrix,
            self.rhs,
            self.force_norm,
            self.n_equations,
            self.n_structures,
        )

    @classmethod
    def from_atoms(cls, mapping: ClusterMap, structures: Iterable[Atoms]) -> FitSystem:
        """Stream evaluated ASE structures into the optimized normal system."""
        if not isinstance(mapping, ClusterMap):
            raise TypeError("mapping must be a ClusterMap")
        from mlfcs.fitting.build import build_system

        return build_system(mapping, structures)

    @property
    def n_parameters(self) -> int:
        return self.space.n_parameters

    @property
    def unobserved_parameters(self) -> tuple[int, ...]:
        """Return columns that are exactly zero in the training design."""
        return tuple(int(index) for index in np.flatnonzero(np.diag(self.matrix) == 0.0))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.space.fingerprint.encode("ascii"))
        digest.update(struct.pack(">qqd", self.n_equations, self.n_structures, self.force_norm))
        digest.update(_bytes(self.matrix))
        digest.update(_bytes(self.rhs))
        return digest.hexdigest()

    def parameter_name(self, parameter: int) -> str:
        """Return the physical layout address of one packed parameter."""
        parameter = int(parameter)
        if not 0 <= parameter < self.n_parameters:
            raise IndexError("parameter is outside the cluster space")
        offsets = self.space.parameter_offsets
        orbit_index = int(np.searchsorted(offsets, parameter, side="right") - 1)
        component = parameter - int(offsets[orbit_index])
        for block in self.space.blocks:
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
            self.space,
            {block.order: values[block.parameters] for block in self.space.blocks},
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
        if self.space.fingerprint != other.space.fingerprint:
            raise ValueError("fit systems use different cluster spaces")
        return FitSystem(
            space=self.space,
            matrix=self.matrix + other.matrix,
            rhs=self.rhs + other.rhs,
            force_norm=self.force_norm + other.force_norm,
            n_equations=self.n_equations + other.n_equations,
            n_structures=self.n_structures + other.n_structures,
        )


__all__ = ["FitSystem"]
