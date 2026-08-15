"""Strictly constrained solution of force-fitting Gram systems."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import pinvh, qr, solve_triangular
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg


def explicit_constraint_null_space(constraints, *, tolerance=1e-11, reporter=None):
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
    if reporter is not None:
        reporter(
            f"Explicit constraint parameterization: {n_parameters} -> "
            f"{reduced_offset} parameters in {n_components} blocks, "
            f"rank={sum(ranks)}, nnz={result.nnz}"
        )
    return result


def solve_scaled_group_lasso(
    gram,
    rhs,
    target_norm,
    scale,
    constraints,
    groups,
    *,
    n_equations,
    tolerance,
    max_iterations,
    verbose,
    reporter=None,
):
    """Solve equality-constrained scaled group LASSO from Gram statistics.

    Groups are symmetry-irreducible cluster-orbit slices in column-normalized
    coordinates.  The concomitant residual scale removes an absolute penalty
    parameter, while the group threshold accounts for group dimension.
    """
    scale = np.asarray(scale)
    normal = np.asarray(gram) * scale[:, None] * scale[None, :]
    scaled_rhs = scale * np.asarray(rhs)
    n_parameters = len(scaled_rhs)
    groups = tuple(groups)
    if not groups:
        raise ValueError("scaled group LASSO requires at least one parameter group")
    projector = (
        ConstraintNullSpace(constraints, reporter=reporter) if constraints.shape[0] else None
    )

    def project(values):
        return projector.project(values) if projector is not None else np.asarray(values)

    target_rms = np.sqrt(max(float(target_norm), 0.0) / n_equations)
    sigma_floor = max(target_rms * 1e-10, np.finfo(float).tiny)
    sigma = max(target_rms, sigma_floor)
    common = np.sqrt(2.0 * np.log(max(len(groups), 2)))
    penalties = np.asarray(
        [(np.sqrt(group.stop - group.start) + common) / n_equations for group in groups]
    )
    diagonal = np.diag(normal) / (n_equations * sigma)
    positive = diagonal[diagonal > np.finfo(float).tiny]
    rho = float(np.median(positive)) if len(positive) else 1.0
    rho = max(rho, np.finfo(float).eps)
    parameters = np.zeros(n_parameters)
    sparse_parameters = np.zeros(n_parameters)
    dual = np.zeros(n_parameters)
    started = perf_counter()
    cg_iterations = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        inverse_loss_scale = 1.0 / (n_equations * sigma)
        quadratic_rhs = inverse_loss_scale * scaled_rhs + rho * (sparse_parameters - dual)

        def multiply(values, loss_scale=inverse_loss_scale, penalty_scale=rho):
            projected = project(values)
            return project(loss_scale * (normal @ projected) + penalty_scale * projected)

        system = LinearOperator(
            (n_parameters,) * 2,
            matvec=multiply,
            rmatvec=multiply,
            dtype=np.float64,
        )
        projected_rhs = project(quadratic_rhs)

        def count_cg(_values):
            nonlocal cg_iterations
            cg_iterations += 1

        parameters, cg_info = cg(
            system,
            projected_rhs,
            x0=parameters,
            rtol=min(max(tolerance * 0.1, 1e-12), 1e-8),
            atol=0.0,
            maxiter=min(n_parameters, 1000),
            callback=count_cg,
        )
        if cg_info < 0:
            raise RuntimeError("scaled group-LASSO quadratic subproblem failed")
        parameters = project(parameters)
        previous_sparse = sparse_parameters.copy()
        candidate = parameters + dual
        sparse_parameters.fill(0.0)
        for group, penalty in zip(groups, penalties, strict=True):
            values = candidate[group]
            norm = float(np.linalg.norm(values))
            shrinkage = max(1.0 - penalty / (rho * max(norm, np.finfo(float).tiny)), 0.0)
            sparse_parameters[group] = shrinkage * values
        primal_vector = parameters - sparse_parameters
        dual += primal_vector
        dual_vector = rho * (sparse_parameters - previous_sparse)

        residual_squared = max(
            float(parameters @ normal @ parameters - 2.0 * parameters @ scaled_rhs + target_norm),
            0.0,
        )
        updated_sigma = max(np.sqrt(residual_squared / n_equations), sigma_floor)
        sigma_change = abs(updated_sigma - sigma) / max(sigma, sigma_floor)
        sigma = 0.5 * sigma + 0.5 * updated_sigma
        primal = float(np.linalg.norm(primal_vector))
        dual_residual = float(np.linalg.norm(dual_vector))
        primal_limit = tolerance * max(
            np.sqrt(n_parameters), np.linalg.norm(parameters), np.linalg.norm(sparse_parameters)
        )
        dual_limit = tolerance * max(np.sqrt(n_parameters), rho * np.linalg.norm(dual))
        if verbose and (iteration <= 5 or iteration % 25 == 0):
            active = sum(np.linalg.norm(sparse_parameters[group]) > 1e-12 for group in groups)
            print(
                f"Scaled group-LASSO iteration {iteration}: sigma={sigma:.6e} eV/A, "
                f"active_orbits={active}/{len(groups)}, primal={primal:.3e}, "
                f"dual={dual_residual:.3e}, elapsed={perf_counter() - started:.2f} s",
                flush=True,
            )
        sigma_tolerance = max(np.sqrt(tolerance), 1e-6)
        if (
            primal <= primal_limit
            and dual_residual <= dual_limit
            and sigma_change <= sigma_tolerance
        ):
            converged = True
            break
        if iteration % 25 == 0:
            if primal > 10.0 * dual_residual:
                rho *= 2.0
                dual /= 2.0
            elif dual_residual > 10.0 * primal:
                rho *= 0.5
                dual *= 2.0

    parameters = project(parameters)
    residual_squared = max(
        float(parameters @ normal @ parameters - 2.0 * parameters @ scaled_rhs + target_norm),
        0.0,
    )
    active_groups = sum(np.linalg.norm(sparse_parameters[group]) > 1e-10 for group in groups)
    return (
        parameters,
        0 if converged else max_iterations,
        iteration,
        float(np.sqrt(residual_squared)),
        float(max(primal, dual_residual)),
        float(sigma),
        int(active_groups),
        int(cg_iterations),
        float(primal),
        float(dual_residual),
    )


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
