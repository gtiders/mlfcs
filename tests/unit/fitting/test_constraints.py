import numpy as np
from ase import Atoms
from ase.build import bulk
from scipy import sparse

from mlfcs.api import ForceConstantCalculation
from mlfcs.core.constraints import build_harmonic_rotational_constraints
from mlfcs.fitting.basis import symmetrized_covariance as _symmetrized_covariance
from mlfcs.fitting.constraints import (
    build_joint_constraints,
    build_wick_to_taylor_fc1_transform,
    build_wick_to_taylor_transform,
    omitted_taylor_fc1,
)
from mlfcs.fitting.model import _StreamingGramSystem
from mlfcs.fitting.solver import ConstraintNullSpace as _ConstraintNullSpace


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
    dimensions = [
        sum(orbit.dimension for orbit in item.orbit_space.orbits) for item in calculations
    ]
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
        _fc1_rotation_matrix,
    )

    harmonic = build_harmonic_rotational_constraints(
        calculations[0].orbit_space, calculations[0].supercell
    )
    harmonic = sparse.hstack(
        [harmonic, sparse.csr_matrix((harmonic.shape[0], sum(dimensions[1:])))],
        format="csr",
    )
    adjacent = sparse.vstack(
        [
            _adjacent_rotational_constraints(
                calculations[index], calculations[index + 1], dimensions, index
            )
            for index in range(len(calculations) - 1)
        ],
        format="csr",
    )
    lower = harmonic @ transform + _fc1_rotation_matrix(
        calculations[0].index.n_primitive
    ) @ build_wick_to_taylor_fc1_transform(calculations, covariance)
    expected = _compress_rows(sparse.vstack([adjacent @ transform, lower], format="csr"))
    wick = build_joint_constraints(
        calculations,
        acoustic=False,
        rotational_mode=2,
        covariance=covariance,
    ).matrix
    np.testing.assert_allclose(wick.toarray(), expected.toarray(), atol=1e-12, rtol=1e-12)


def test_fc2_only_fit_includes_shared_fc1_zero_rotational_boundary():
    primitive = bulk("Si", "diamond", a=5.43)
    calculation = ForceConstantCalculation(
        primitive,
        order=2,
        supercell=(2, 2, 2),
        cutoff=-2,
        verbose=False,
    )
    covariance = np.eye(len(calculation.supercell) * 3)
    actual = build_joint_constraints(
        (calculation,),
        acoustic=False,
        rotational_mode=2,
        covariance=covariance,
    ).matrix
    expected = build_harmonic_rotational_constraints(calculation.orbit_space, calculation.supercell)

    assert actual.shape[0] > 0
    # Row compression can change normalization/order, so compare null-space
    # action through the ranks of the two equivalent row spaces.
    assert np.linalg.matrix_rank(actual.toarray()) == np.linalg.matrix_rank(expected.toarray())
    stacked = sparse.vstack([actual, expected]).toarray()
    assert np.linalg.matrix_rank(stacked) == np.linalg.matrix_rank(expected.toarray())


def test_explicit_fc1_transform_matches_reported_wick_contraction():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculations = tuple(
        ForceConstantCalculation(
            primitive,
            order=order,
            supercell=(2, 1, 1),
            cutoff=4.1,
            verbose=False,
        )
        for order in (2, 3)
    )
    covariance = np.eye(len(calculations[0].supercell) * 3)
    rng = np.random.default_rng(19)
    parameters = rng.normal(
        size=sum(sum(orbit.dimension for orbit in item.orbit_space.orbits) for item in calculations)
    )
    transform = build_wick_to_taylor_fc1_transform(calculations, covariance)
    np.testing.assert_allclose(
        (transform @ parameters).reshape(-1, 3),
        omitted_taylor_fc1(calculations, parameters, covariance),
        atol=1e-13,
    )
