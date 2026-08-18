import numpy as np
import pytest
from ase import Atoms
from scipy import sparse

from mlfcs.api import ForceConstantCalculation
from mlfcs.clusters.orbits import cluster_invariant_dimension
from mlfcs.fitting.constraints import (
    _validate_missing_contractions,
    append_zero_taylor_order_constraints,
    build_joint_constraints,
    build_wick_to_taylor_fc1_transform,
    build_wick_to_taylor_transform,
    omitted_taylor_fc1,
)
from mlfcs.fitting.solver import ConstraintNullSpace as _ConstraintNullSpace
from mlfcs.fitting.solver import explicit_constraint_null_space, solve_scaled_group_lasso


def test_implicit_null_space_is_idempotent_and_satisfies_constraints():
    constraints = sparse.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    projector = _ConstraintNullSpace(constraints)
    values = np.array([2.0, -3.0, 7.0])
    projected = projector.project(values)
    np.testing.assert_allclose(constraints @ projected, 0.0, atol=1e-13)
    np.testing.assert_allclose(projector.project(projected), projected, atol=1e-13)


def test_scaled_group_lasso_selects_orbits_and_preserves_hard_constraint():
    result = solve_scaled_group_lasso(
        np.eye(3),
        np.array([3.0, 3.0, 0.01]),
        20.0,
        np.ones(3),
        sparse.csr_matrix([[1.0, -1.0, 0.0]]),
        (slice(0, 2), slice(2, 3)),
        n_equations=10,
        tolerance=1e-6,
        max_iterations=500,
        verbose=False,
    )
    parameters, stop_code = result[:2]
    assert stop_code == 0
    np.testing.assert_allclose(parameters[0], parameters[1], atol=1e-12)
    assert abs(parameters[2]) < 1e-6


def test_explicit_fc1_transform_matches_reported_wick_contraction():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculations = tuple(
        ForceConstantCalculation(
            primitive, order=order, supercell=(2, 1, 1), cutoff=4.1, verbose=False
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


def test_fc1_transform_maps_supercell_anchor_to_primitive_site():
    primitive = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]),
        pbc=True,
    )
    calculations = tuple(
        ForceConstantCalculation(
            primitive, order=order, supercell=(2, 2, 2), cutoff=-1, verbose=False
        )
        for order in (2, 3)
    )
    covariance = np.eye(len(calculations[0].supercell) * 3)
    transform = build_wick_to_taylor_fc1_transform(calculations, covariance)
    assert transform.shape[0] == 3 * len(primitive)


def test_frozen_fc2_constraint_cancels_fc4_to_fc2_taylor_contraction():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculations = tuple(
        ForceConstantCalculation(
            primitive, order=order, supercell=(2, 1, 1), cutoff=4.1, verbose=False
        )
        for order in (2, 3, 4)
    )
    covariance = np.eye(len(calculations[0].supercell) * 3) * 0.03
    constraints = append_zero_taylor_order_constraints(
        build_joint_constraints(calculations, acoustic=False), calculations, covariance, (2,)
    )
    null_space = explicit_constraint_null_space(constraints.matrix)
    transform = build_wick_to_taylor_transform(calculations, covariance)
    fc2_count = sum(orbit.dimension for orbit in calculations[0].orbit_space.orbits)
    rng = np.random.default_rng(73)
    parameters = np.asarray(null_space @ rng.normal(size=null_space.shape[1]))

    np.testing.assert_allclose(transform[:fc2_count] @ parameters, 0.0, atol=1e-12)
    assert np.linalg.norm(parameters[fc2_count:]) > 0


def test_missing_symmetry_forbidden_zero_wick_contraction_is_accepted(monkeypatch):
    monkeypatch.setattr(
        "mlfcs.fitting.constraints.cluster_invariant_dimension", lambda *args, **kwargs: 0
    )
    calculation = type("Calculation", (), {"index": object(), "symmetry": object()})()
    _validate_missing_contractions(
        {(0, 0, 0): {4: np.array([[2e-13]])}},
        {(0, 0, 0): {4: np.array([[3.0]])}},
        {},
        calculation,
        source_order=5,
        target_order=3,
    )


def test_centrosymmetric_onsite_odd_tensor_has_zero_allowed_dimension():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculation = ForceConstantCalculation(
        primitive, order=3, supercell=(2, 1, 1), cutoff=4.1, verbose=False
    )
    assert cluster_invariant_dimension((0, 0, 0), calculation.index, calculation.symmetry) == 0


def test_missing_symmetry_forbidden_nonzero_wick_contraction_is_an_error(monkeypatch):
    monkeypatch.setattr(
        "mlfcs.fitting.constraints.cluster_invariant_dimension", lambda *args, **kwargs: 0
    )
    calculation = type("Calculation", (), {"index": object(), "symmetry": object()})()
    with pytest.raises(RuntimeError, match="symmetry-forbidden"):
        _validate_missing_contractions(
            {(0, 0, 0): {4: np.array([[1e-4]])}},
            {(0, 0, 0): {4: np.array([[1.0]])}},
            {},
            calculation,
            source_order=5,
            target_order=3,
        )


def test_missing_symmetry_allowed_wick_contraction_is_a_support_error(monkeypatch):
    monkeypatch.setattr(
        "mlfcs.fitting.constraints.cluster_invariant_dimension", lambda *args, **kwargs: 2
    )
    calculation = type("Calculation", (), {"index": object(), "symmetry": object()})()
    with pytest.raises(ValueError, match="outside its configured support"):
        _validate_missing_contractions(
            {(0, 1, 2): {4: np.array([[0.0]])}},
            {(0, 1, 2): {4: np.array([[1.0]])}},
            {},
            calculation,
            source_order=5,
            target_order=3,
        )
