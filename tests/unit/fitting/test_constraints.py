import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from mlfcs.fitting.model import (
    _ConstraintNullSpace,
    _solve_constrained_lsmr,
    _StreamingGramSystem,
)


def test_kkt_solver_enforces_constraint_during_least_squares():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    operator = LinearOperator(
        matrix.shape,
        matvec=matrix.__matmul__,
        rmatvec=matrix.T.__matmul__,
        dtype=float,
    )
    constraints = sparse.csr_matrix([[1.0, -1.0]])
    target = np.array([1.0, 3.0, 5.0])

    solution = _solve_constrained_lsmr(
        operator,
        constraints,
        target,
        tolerance=1e-12,
        max_iterations=100,
        damping=0.0,
        verbose=False,
    )[0]

    np.testing.assert_allclose(constraints @ solution, 0.0, atol=1e-11)
    np.testing.assert_allclose(solution, [7 / 3, 7 / 3], atol=1e-11)


def test_implicit_null_space_is_idempotent_and_satisfies_constraints():
    constraints = sparse.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    projector = _ConstraintNullSpace(constraints)
    values = np.array([2.0, -3.0, 7.0])
    projected = projector.project(values)

    np.testing.assert_allclose(constraints @ projected, 0.0, atol=1e-13)
    np.testing.assert_allclose(projector.project(projected), projected, atol=1e-13)


def test_projected_gram_cg_matches_explicit_constrained_solution():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = np.array([1.0, 3.0, 5.0])
    gram = _StreamingGramSystem(
        matrix.T @ matrix,
        matrix.T @ target,
        float(target @ target),
    )
    constraints = sparse.csr_matrix([[1.0, -1.0]])
    result = gram.solve(
        np.ones(2),
        constraints,
        tolerance=1e-12,
        max_iterations=100,
        damping=0.0,
        verbose=False,
    )

    np.testing.assert_allclose(result[0], [7 / 3, 7 / 3], atol=1e-11)
    assert result[1] == 0
    assert result[2] < 10
    assert result[4] < 1e-11
