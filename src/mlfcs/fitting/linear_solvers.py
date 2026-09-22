"""Unconstrained solution of force-fitting Gram systems."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import minres


def solve_gram_system(
    gram,
    rhs,
    target_norm,
    scale,
    *,
    tolerance,
    max_iterations,
):
    """Solve one preconditioned Gram system in the physical parameter coordinates."""
    scale = np.asarray(scale)
    scaled_rhs = scale * rhs
    n_parameters = len(scaled_rhs)
    if n_parameters == 0:
        residual = float(np.sqrt(max(float(target_norm), 0.0)))
        return np.zeros(0), 0, 0, residual, 0.0
    normal = gram * scale[:, None] * scale[None, :]
    iterations = 0

    def callback(_values):
        nonlocal iterations
        iterations += 1

    parameters, info = minres(
        normal,
        scaled_rhs,
        x0=np.zeros(n_parameters),
        rtol=tolerance,
        maxiter=max_iterations,
        callback=callback,
        check=True,
    )
    stationarity = normal @ parameters - scaled_rhs
    model_norm = float(parameters @ normal @ parameters)
    cross = float(parameters @ scaled_rhs)
    residual_squared = model_norm - 2 * cross + target_norm
    cancellation_bound = (
        32 * np.finfo(float).eps * (abs(model_norm) + 2 * abs(cross) + abs(target_norm))
    )
    if abs(residual_squared) <= cancellation_bound:
        residual_squared = 0.0
    residual_squared = max(float(residual_squared), 0.0)
    return (
        parameters,
        int(info),
        iterations,
        float(np.sqrt(residual_squared)),
        float(np.linalg.norm(stationarity)),
    )


__all__ = ["solve_gram_system"]
