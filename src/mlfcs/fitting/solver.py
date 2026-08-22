"""Strictly constrained solution of force-fitting Gram systems."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import pinvh
from scipy.sparse.linalg import LinearOperator, cg


class ConstraintNullSpace:
    """Implicit orthogonal projector onto null(C), including redundant rows."""

    def __init__(self, constraints, reporter=None):
        self.constraints = sparse.csr_matrix(constraints)
        row_gram = (self.constraints @ self.constraints.T).toarray()
        row_gram = (row_gram + row_gram.T) * 0.5
        self.row_gram_inverse, self.rank = pinvh(row_gram, return_rank=True, check_finite=False)
        if reporter is not None:
            reporter(
                f"Implicit constraint null space: numerical rank={self.rank}/"
                f"{self.constraints.shape[0]}, redundant rows="
                f"{self.constraints.shape[0] - self.rank}"
            )

    def project(self, values):
        values = np.asarray(values)
        residual = self.constraints @ values
        multipliers = self.row_gram_inverse @ residual
        return values - self.constraints.T @ multipliers


def solve_gram_system(
    gram,
    rhs,
    target_norm,
    scale,
    constraints,
    *,
    tolerance,
    max_iterations,
    damping,
    verbose,
    reporter=None,
):
    """Solve a preconditioned Gram system in the equality-constraint null space."""
    scale = np.asarray(scale)
    normal = gram * scale[:, None] * scale[None, :]
    if damping:
        normal.flat[:: len(normal) + 1] += damping**2
    scaled_rhs = scale * rhs
    n_parameters = len(scaled_rhs)
    started = perf_counter()
    previous = started
    iterations = 0
    if constraints.shape[0]:
        projector = ConstraintNullSpace(constraints, reporter=reporter)
        projected_rhs = projector.project(scaled_rhs)

        def multiply(values):
            return projector.project(normal @ projector.project(values))

        system = LinearOperator(
            (n_parameters,) * 2,
            matvec=multiply,
            rmatvec=multiply,
            dtype=np.float64,
        )

        def callback(values):
            nonlocal iterations, previous
            iterations += 1
            now = perf_counter()
            if verbose and (iterations <= 5 or iterations % 100 == 0):
                drift = np.linalg.norm(constraints @ values[:n_parameters], ord=np.inf)
                gradient = multiply(values) - projected_rhs
                relative_gradient = np.linalg.norm(gradient) / max(
                    np.linalg.norm(projected_rhs), np.finfo(float).tiny
                )
                print(
                    f"Projected CG iteration {iterations}: relative gradient="
                    f"{relative_gradient:.6e}, max constraint residual={drift:.6e}, "
                    f"step={now - previous:.3f} s, elapsed={now - started:.2f} s",
                    flush=True,
                )
            previous = now

        parameters, info = cg(
            system,
            projected_rhs,
            x0=np.zeros(n_parameters),
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            callback=callback,
        )
        parameters = projector.project(parameters)
        stationarity = projector.project(normal @ parameters - scaled_rhs)
    else:

        def callback(_values):
            nonlocal iterations
            iterations += 1

        parameters, info = cg(
            normal,
            scaled_rhs,
            x0=np.zeros(n_parameters),
            rtol=tolerance,
            atol=0.0,
            maxiter=max_iterations,
            callback=callback,
        )
        stationarity = normal @ parameters - scaled_rhs
    residual_squared = max(
        float(parameters @ normal @ parameters - 2 * parameters @ scaled_rhs + target_norm),
        0.0,
    )
    return (
        parameters,
        int(info),
        iterations,
        float(np.sqrt(residual_squared)),
        float(np.linalg.norm(stationarity)),
    )
