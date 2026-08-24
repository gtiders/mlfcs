import numpy as np
import pytest
from ase import Atoms
from scipy import sparse
from supercell_helpers import make_supercell

from mlfcs.fitting.backends.wick.lowering import (
    _target_orbit_intertwiner,
    _validate_missing_exact_contractions,
    build_fc1_lowering_transform,
    lowered_fc1,
)
from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
from mlfcs.fitting.linear_solvers import ConstraintNullSpace as _ConstraintNullSpace
from mlfcs.fitting.linear_solvers import solve_scaled_group_lasso
from mlfcs.interactions.keys import InteractionKey


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


def test_target_orbit_intertwiner_matches_joint_least_squares():
    rng = np.random.default_rng(9821)
    keys = (
        InteractionKey((0, 0), ((0, 0, 0),)),
        InteractionKey((0, 0), ((1, 0, 0),)),
        InteractionKey((0, 0), ((0, 1, 0),)),
    )
    columns = [rng.normal(size=(9, 4)) for _ in keys]
    stacked = np.vstack(columns)
    right_hand_sides = [rng.normal(size=(9, 7)) for _ in keys]

    intertwiner = _target_orbit_intertwiner(13, list(zip(keys, columns, strict=True)))
    actual = sum(
        intertwiner.dual_blocks[key] @ values
        for key, values in zip(keys, right_hand_sides, strict=True)
    )
    expected = np.linalg.lstsq(stacked, np.vstack(right_hand_sides), rcond=None)[0]

    assert intertwiner.offset == 13
    assert intertwiner.dimension == 4
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_explicit_fc1_transform_matches_reported_wick_contraction():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = make_supercell(primitive, (3, 3, 3))[0]
    calculations = tuple(
        FiniteDifferenceCalculation(
            primitive, order=order, reference=reference, cutoff=4.1, verbose=False
        )
        for order in (2, 3)
    )
    covariance = np.eye(len(calculations[0].supercell) * 3)
    rng = np.random.default_rng(19)
    parameters = rng.normal(
        size=sum(sum(orbit.dimension for orbit in item.orbit_space.orbits) for item in calculations)
    )
    transform = build_fc1_lowering_transform(calculations, covariance)
    np.testing.assert_allclose(
        (transform @ parameters).reshape(-1, 3),
        lowered_fc1(calculations, parameters, covariance),
        atol=1e-13,
    )


def test_fc1_transform_maps_supercell_anchor_to_primitive_site():
    primitive = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]),
        pbc=True,
    )
    reference = make_supercell(primitive, (2, 2, 2))[0]
    calculations = tuple(
        FiniteDifferenceCalculation(
            primitive, order=order, reference=reference, cutoff=-1, verbose=False
        )
        for order in (2, 3)
    )
    covariance = np.eye(len(calculations[0].supercell) * 3)
    transform = build_fc1_lowering_transform(calculations, covariance)
    assert transform.shape[0] == 3 * len(primitive)


def test_missing_negligible_exact_wick_contraction_is_accepted():
    key = InteractionKey((0, 0, 0), ((0, 0, 0), (0, 0, 0)))
    _validate_missing_exact_contractions(
        {key: {4: np.array([[2e-13]])}},
        {key: {4: np.array([[3.0]])}},
        {},
        source_order=5,
        target_order=3,
    )


def test_centrosymmetric_onsite_odd_tensor_has_zero_allowed_dimension():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = make_supercell(primitive, (3, 3, 3))[0]
    calculation = FiniteDifferenceCalculation(
        primitive, order=3, reference=reference, cutoff=4.1, verbose=False
    )
    onsite = InteractionKey((0, 0, 0), ((0, 0, 0), (0, 0, 0)))
    assert all(
        orbit.representative != onsite
        for orbit in calculation.interaction_space.primitive_orbit_space.orbits
    )


def test_missing_nonzero_exact_wick_contraction_is_a_support_error():
    key = InteractionKey((0, 0, 0), ((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="outside the configured"):
        _validate_missing_exact_contractions(
            {key: {4: np.array([[1e-4]])}},
            {key: {4: np.array([[1.0]])}},
            {},
            source_order=5,
            target_order=3,
        )
