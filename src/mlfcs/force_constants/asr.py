"""Independent acoustic sum-rule projection for primitive force constants."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.core.errors import ConstraintProjectionError
from mlfcs.force_constants.acoustic import constraint_matrix, relative_residual
from mlfcs.force_constants.model import ForceConstants


@dataclass(frozen=True, slots=True)
class ASRReport:
    """Diagnostics for one tensor order projected onto the acoustic sum rule."""

    order: int
    equations: int
    parameters: int
    residual_before: float
    residual_after: float
    relative_before: float
    relative_after: float
    correction_norm: float
    relative_correction: float
    iterations: int


@dataclass(frozen=True, slots=True)
class ASRResult:
    """A projected force-constant model and its order-local diagnostics."""

    force_constants: ForceConstants
    reports: tuple[ASRReport, ...]

    def report(self, order: int) -> ASRReport:
        """Return diagnostics for one projected order."""
        for value in self.reports:
            if value.order == order:
                return value
        raise KeyError(f"order {order} was not projected")


def _project(
    order: int,
    matrix: sparse.csr_matrix,
    parameters: np.ndarray,
    *,
    rtol: float,
    name: str = "ASR",
) -> tuple[np.ndarray, ASRReport]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (matrix.shape[1],):
        raise ValueError(
            f"order-{order} {name} expects {matrix.shape[1]} parameters, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"order-{order} parameters contain NaN or infinite values")
    before, relative_before = relative_residual(matrix, values)
    if matrix.shape[0] == 0 or values.size == 0 or relative_before <= rtol:
        return values.copy(), ASRReport(
            order,
            matrix.shape[0],
            matrix.shape[1],
            before,
            before,
            relative_before,
            relative_before,
            0.0,
            0.0,
            0,
        )

    projected = values.copy()
    correction = np.zeros_like(values)
    iterations = 0
    solver_rtol = max(rtol * 0.1, np.finfo(float).eps)
    maximum_steps = max(1, 4 * min(matrix.shape))
    for _ in range(3):
        residual = np.asarray(matrix @ projected)
        solution = lsmr(
            matrix,
            residual,
            atol=solver_rtol,
            btol=solver_rtol,
            conlim=0.0,
            maxiter=maximum_steps,
        )
        step = np.asarray(solution[0])
        if not np.all(np.isfinite(step)):
            raise ConstraintProjectionError(
                f"order-{order} {name} projection produced a non-finite correction"
            )
        projected -= step
        correction += step
        iterations += int(solution[2])
        after, relative_after = relative_residual(matrix, projected)
        if relative_after <= rtol:
            break
    else:
        raise ConstraintProjectionError(
            f"order-{order} {name} projection did not converge: relative residual "
            f"{relative_before:.6e} -> {relative_after:.6e}, requested {rtol:.6e}"
        )

    correction_norm = float(np.linalg.norm(correction))
    parameter_norm = float(np.linalg.norm(values))
    relative_correction = correction_norm / parameter_norm if parameter_norm else 0.0
    return projected, ASRReport(
        order,
        matrix.shape[0],
        matrix.shape[1],
        before,
        after,
        relative_before,
        relative_after,
        correction_norm,
        relative_correction,
        iterations,
    )


def enforce_asr(
    model: ForceConstants,
    *,
    orders: Iterable[int] | None = None,
    rtol: float = 1e-10,
) -> ASRResult:
    """Project selected primitive force-constant orders onto translational invariance."""
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive")
    selected = model.orders if orders is None else tuple(int(order) for order in orders)
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("orders must be unique and ascending")
    missing = tuple(order for order in selected if order not in model.coefficients)
    if missing:
        raise KeyError(f"force constants do not contain orders {missing}")

    coefficients = dict(model.coefficients)
    reports = []
    for order in selected:
        matrix = constraint_matrix(model, order)
        projected, report = _project(
            order,
            matrix,
            model.coefficients[order],
            rtol=float(rtol),
        )
        coefficients[order] = projected
        reports.append(report)
    return ASRResult(ForceConstants(model.space, coefficients), tuple(reports))


__all__ = ["ASRReport", "ASRResult", "enforce_asr"]
