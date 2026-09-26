"""Default solution of a scaled force-fitting normal system."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import minres

from mlfcs.core.log_error import get_logger

logger = get_logger(__name__)


def solve(matrix: np.ndarray, rhs: np.ndarray, *, rtol: float, max_steps: int) -> np.ndarray:
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be positive and finite")
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError("the default solver requires every matrix diagonal to be positive")
    scale = 1.0 / np.sqrt(diagonal)
    normal = matrix * scale[:, None] * scale[None, :]
    scaled_rhs = scale * rhs
    logger.info(
        "MINRES system: dimension=%d rhs_norm=%.10e column_scale=[%.6g, %.6g] "
        "rtol=%.3g max_steps=%d",
        len(rhs),
        float(np.linalg.norm(scaled_rhs)),
        float(np.min(scale)),
        float(np.max(scale)),
        rtol,
        max_steps,
    )
    iterations = 0
    started = perf_counter()

    def callback(_value):
        nonlocal iterations
        iterations += 1

    scaled, info = minres(
        normal,
        scaled_rhs,
        x0=np.zeros(len(rhs)),
        rtol=rtol,
        maxiter=max_steps,
        callback=callback,
        check=True,
    )
    parameters = scale * scaled
    residual = float(np.linalg.norm(matrix @ parameters - rhs))
    logger.info(
        "MINRES finished: steps=%d stop_code=%d normal_residual=%.10e "
        "relative_normal_residual=%.10e elapsed=%.2f s",
        iterations,
        info,
        residual,
        residual / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny),
        perf_counter() - started,
    )
    if info != 0:
        raise RuntimeError(
            f"fit solve did not converge in {iterations} steps: stop code {info}, "
            f"normal residual {residual:.10e}"
        )
    return parameters


__all__ = ["solve"]
