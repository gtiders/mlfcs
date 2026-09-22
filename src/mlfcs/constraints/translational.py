"""Shared post-processing projection for the translational acoustic sum rule."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

COEFFICIENT_ZERO_RTOL = 64.0 * np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class ASRProjectionResult:
    """Measured outcome of one Euclidean projection onto ASR-feasible parameters."""

    parameters: np.ndarray
    initial_residual: float
    final_residual: float
    correction_norm: float
    relative_correction: float
    optimality_residual: float
    iterations: int


@dataclass(frozen=True, slots=True)
class TranslationalASRProjector:
    """One order-local ASR projector in Cartesian orbit-coefficient coordinates."""

    order: int
    physical_dimension: int
    constraints: sparse.csr_matrix

    @classmethod
    def from_orbit_space(cls, orbit_space) -> TranslationalASRProjector:
        """Build the projector consumed by fitting and finite differences."""
        constraints = build_translational_constraints(orbit_space)
        physical_dimension = sum(int(orbit.dimension) for orbit in orbit_space.orbits)
        return cls(int(orbit_space.order), physical_dimension, constraints)

    def maximum_residual(self, parameters: np.ndarray) -> float:
        """Return the largest Cartesian acoustic-sum drift."""
        values = self._parameters(parameters)
        if self.constraints.shape[0] == 0 or values.size == 0:
            return 0.0
        residual = self.constraints @ values
        return float(np.max(np.abs(residual))) if residual.size else 0.0

    def relative_residual(self, parameters: np.ndarray) -> float:
        """Return the ASR drift normalized by the equation and parameter scales."""
        values = self._parameters(parameters)
        if self.constraints.shape[0] == 0 or values.size == 0:
            return 0.0
        residual = self.maximum_residual(values)
        row_scale = np.asarray(np.abs(self.constraints).sum(axis=1)).reshape(-1)
        equation_scale = float(np.max(row_scale)) if row_scale.size else 0.0
        parameter_scale = float(np.max(np.abs(values))) if values.size else 0.0
        denominator = max(equation_scale * parameter_scale, np.finfo(float).tiny)
        return residual / denominator

    def project(
        self,
        parameters: np.ndarray,
        *,
        tolerance: float,
    ) -> ASRProjectionResult:
        """Return the closest coefficients satisfying the acoustic sum rule."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("ASR projection tolerance must be finite and positive")
        values = self._parameters(parameters)
        if not np.all(np.isfinite(values)):
            raise ValueError("ASR projection parameters contain NaN or infinite values")
        initial = self.maximum_residual(values)
        if self.constraints.shape[0] == 0 or values.size == 0:
            return ASRProjectionResult(values.copy(), initial, initial, 0.0, 0.0, 0.0, 0)

        projected = values.copy()
        correction = np.zeros_like(values)
        iterations = 0
        optimality = 0.0
        for _ in range(3):
            residual = np.asarray(self.constraints @ projected)
            if self.relative_residual(projected) <= tolerance:
                break
            solver_tolerance = max(tolerance * 0.1, np.finfo(float).eps)
            solution = lsmr(
                self.constraints,
                residual,
                atol=solver_tolerance,
                btol=solver_tolerance,
                maxiter=max(1000, 4 * self.physical_dimension),
            )
            step = np.asarray(solution[0])
            if not np.all(np.isfinite(step)):
                raise RuntimeError(f"order-{self.order} ASR projection produced non-finite values")
            projected -= step
            correction += step
            iterations += int(solution[2])
            equation_error = np.asarray(self.constraints @ step) - residual
            optimality = max(
                optimality,
                float(np.linalg.norm(self.constraints.T @ equation_error)),
            )

        final = self.maximum_residual(projected)
        relative_final = self.relative_residual(projected)
        if relative_final > tolerance:
            raise RuntimeError(
                f"order-{self.order} ASR projection did not converge for "
                f"{self.physical_dimension} parameters: residual {initial:.6e} -> "
                f"{final:.6e}, relative residual {relative_final:.6e} exceeds "
                f"{tolerance:.6e}"
            )
        correction_norm = float(np.linalg.norm(correction))
        relative_correction = correction_norm / max(
            float(np.linalg.norm(values)), np.finfo(float).tiny
        )
        return ASRProjectionResult(
            parameters=projected,
            initial_residual=initial,
            final_residual=final,
            correction_norm=correction_norm,
            relative_correction=relative_correction,
            optimality_residual=optimality,
            iterations=iterations,
        )

    def _parameters(self, parameters: np.ndarray) -> np.ndarray:
        values = np.asarray(parameters, dtype=float)
        if values.ndim != 1 or values.shape[0] != self.physical_dimension:
            raise ValueError(
                f"order-{self.order} ASR projection expects "
                f"{self.physical_dimension} parameters, got shape {values.shape}"
            )
        return values


def build_translational_constraints(orbit_space) -> sparse.csr_matrix:
    """Build the order-local ASR equations in Cartesian orbit coefficients."""
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for orbit_index, orbit in enumerate(orbit_space.orbits):
        cartesian_basis = np.asarray(orbit.cartesian_basis, dtype=float)
        for image in orbit.images:
            transformed = np.asarray(image.action.apply_columns(cartesian_basis), dtype=float)
            scale = max(float(np.max(np.abs(transformed))) if transformed.size else 0.0, 1.0)
            nonzero_bound = COEFFICIENT_ZERO_RTOL * scale
            for component in range(3**orbit_space.order):
                directions = np.unravel_index(component, (3,) * orbit_space.order)
                labels = image.key.labels if hasattr(image, "key") else image.cluster
                key = tuple(labels[:-1]) + tuple(int(value) for value in directions)
                equation = equations.setdefault(key, len(equations))
                nonzero = np.flatnonzero(np.abs(transformed[component]) > nonzero_bound)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(transformed[component, value]) for value in nonzero)
    matrix = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def maximum_acoustic_sum_rule_drift(orbit_space, coefficients: list[np.ndarray]) -> float:
    """Return the largest atomic-sum residual of one orbit parameter vector."""
    values = np.concatenate(coefficients) if coefficients else np.empty(0, dtype=float)
    return TranslationalASRProjector.from_orbit_space(orbit_space).maximum_residual(values)


def project_acoustic_sum_rule(
    orbit_space,
    coefficients: list[np.ndarray],
    *,
    tolerance: float = 1e-10,
    return_result: bool = False,
):
    """Project orbit coefficients through the canonical shared ASR projector."""
    offsets = np.cumsum([0] + [len(values) for values in coefficients])
    values = np.concatenate(coefficients) if coefficients else np.empty(0, dtype=float)
    result = TranslationalASRProjector.from_orbit_space(orbit_space).project(
        values,
        tolerance=tolerance,
    )
    projected = [result.parameters[begin:end] for begin, end in pairwise(offsets)]
    return (projected, result) if return_result else projected


__all__ = [
    "COEFFICIENT_ZERO_RTOL",
    "ASRProjectionResult",
    "TranslationalASRProjector",
    "build_translational_constraints",
    "maximum_acoustic_sum_rule_drift",
    "project_acoustic_sum_rule",
]
