import numpy as np
from ase import Atoms
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from mlfcs.api import ForceConstantCalculation
from mlfcs.fitting.constraints import build_joint_constraints, build_wick_to_taylor_transform
from mlfcs.fitting.model import (
    _ConstraintNullSpace,
    _solve_constrained_lsmr,
    _StreamingGramSystem,
    _symmetrized_covariance,
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


def test_wick_rotational_constraints_equal_taylor_constraints_after_transform():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculations = tuple(
        ForceConstantCalculation(
            primitive,
            order=order,
            supercell=(2, 1, 1),
            cutoff=4.1,
            verbose=False,
        )
        for order in (2, 3, 4)
    )
    dimensions = [sum(orbit.dimension for orbit in item.orbit_space.orbits) for item in calculations]
    rng = np.random.default_rng(31)
    displacement = rng.normal(size=(20, 2, 3))
    displacement -= displacement.mean(axis=1, keepdims=True)
    covariance = _symmetrized_covariance(displacement, calculations[0])
    transform = build_wick_to_taylor_transform(calculations, covariance)
    # Build the same Taylor rotational rows explicitly, then map them into
    # Wick coordinates and compare with the public constraint builder.
    from mlfcs.fitting.constraints import (
        _adjacent_rotational_constraints,
        _compress_rows,
    )

    rotational = sparse.vstack(
        [
            _adjacent_rotational_constraints(
                calculations[index], calculations[index + 1], dimensions, index
            )
            for index in range(len(calculations) - 1)
        ],
        format="csr",
    )
    expected = _compress_rows(rotational @ transform)
    wick = build_joint_constraints(
        calculations,
        acoustic=False,
        rotational_mode=2,
        covariance=covariance,
    ).matrix
    np.testing.assert_allclose(wick.toarray(), expected.toarray(), atol=1e-12, rtol=1e-12)
