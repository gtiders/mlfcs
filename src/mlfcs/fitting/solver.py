"""Default solution of a scaled force-fitting normal system."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import minres

from mlfcs.core.log import get_logger

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
    iterations = 0

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
        "MINRES finished after %d steps: stop code %d, normal residual %.10e",
        iterations,
        info,
        residual,
    )
    if info != 0:
        raise RuntimeError(
            f"fit solve did not converge in {iterations} steps: stop code {info}, "
            f"normal residual {residual:.10e}"
        )
    return parameters


__all__ = ["solve"]
