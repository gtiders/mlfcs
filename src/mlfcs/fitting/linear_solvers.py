"""Strictly constrained solution of force-fitting Gram systems."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import pinvh, qr, solve_triangular
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg

logger = logging.getLogger(__name__)


def explicit_constraint_null_space(constraints, *, tolerance=1e-11):
    """Construct a block-sparse map from free to exactly constrained parameters.

    Constraint-connected parameter components are factorized independently.
    This avoids both a global dense null-space matrix and the unreduced Gram
    matrix.  Within each component, pivoted QR selects dependent columns and a
    triangular solve expresses them in terms of the free columns.
    """
    matrix = sparse.csr_matrix(constraints)
    n_parameters = matrix.shape[1]
    if matrix.shape[0] == 0:
        return sparse.eye(n_parameters, format="csc")
    adjacency = matrix.T @ matrix
    adjacency.data[:] = 1.0
    n_components, labels = connected_components(adjacency, directed=False)
    row_indices = []
    column_indices = []
    data = []
    reduced_offset = 0
    ranks = []
    for component in range(n_components):
        columns = np.flatnonzero(labels == component)
        rows = np.unique(matrix[:, columns].nonzero()[0])
        block = matrix[rows][:, columns].toarray()
        if not len(rows) or not np.any(block):
            rank = 0
            permutation = np.arange(len(columns))
            local = np.eye(len(columns))
        else:
            _q, triangular, permutation = qr(
                block, mode="economic", pivoting=True, check_finite=False
            )
            diagonal = np.abs(np.diag(triangular))
            threshold = (
                tolerance * max(block.shape) * float(diagonal.max()) if len(diagonal) else 0.0
            )
            rank = int(np.count_nonzero(diagonal > threshold))
            free = len(columns) - rank
            local = np.zeros((len(columns), free))
            if free:
                local[permutation[rank:], np.arange(free)] = 1.0
                if rank:
                    local[permutation[:rank]] = -solve_triangular(
                        triangular[:rank, :rank],
                        triangular[:rank, rank:],
                        check_finite=False,
                    )
        ranks.append(rank)
        nonzero_row, nonzero_column = np.nonzero(np.abs(local) > tolerance)
        row_indices.extend(columns[nonzero_row])
        column_indices.extend(reduced_offset + nonzero_column)
        data.extend(local[nonzero_row, nonzero_column])
        reduced_offset += local.shape[1]
    result = sparse.coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(n_parameters, reduced_offset),
    ).tocsc()
    residual = matrix @ result
    maximum = float(np.max(np.abs(residual.data))) if residual.nnz else 0.0
    if maximum > max(tolerance * 100, 1e-9):
        raise RuntimeError(f"explicit constraint null space has residual {maximum:.6e}")
    logger.info(
        f"Explicit constraint parameterization: {n_parameters} -> "
        f"{reduced_offset} parameters in {n_components} blocks, "
        f"rank={sum(ranks)}, nnz={result.nnz}"
    )
    return result


class ConstraintNullSpace:
    """Implicit orthogonal projector onto null(C), including redundant rows."""

    def __init__(self, constraints):
        self.constraints = sparse.csr_matrix(constraints)
        row_gram = (self.constraints @ self.constraints.T).toarray()
        row_gram = (row_gram + row_gram.T) * 0.5
        self.row_gram_inverse, self.rank = pinvh(row_gram, return_rank=True, check_finite=False)
        logger.info(
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
):
    """Solve a preconditioned Gram system in the equality-constraint null space."""
    scale = np.asarray(scale)
    normal = gram * scale[:, None] * scale[None, :]
    scaled_rhs = scale * rhs
    n_parameters = len(scaled_rhs)
    started = perf_counter()
    previous = started
    iterations = 0
    if constraints.shape[0]:
        projector = ConstraintNullSpace(constraints)
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
            if logger.isEnabledFor(logging.DEBUG) and (iterations <= 5 or iterations % 100 == 0):
                drift = np.linalg.norm(constraints @ values[:n_parameters], ord=np.inf)
                gradient = multiply(values) - projected_rhs
                relative_gradient = np.linalg.norm(gradient) / max(
                    np.linalg.norm(projected_rhs), np.finfo(float).tiny
                )
                logger.debug(
                    f"Projected CG iteration {iterations}: relative gradient="
                    f"{relative_gradient:.6e}, max constraint residual={drift:.6e}, "
                    f"step={now - previous:.3f} s, elapsed={now - started:.2f} s",
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
