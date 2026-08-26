import numpy as np
from ase import Atoms
from scipy import sparse
from supercell_helpers import make_supercell

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
    )
    parameters, stop_code = result[:2]
    assert stop_code == 0
    np.testing.assert_allclose(parameters[0], parameters[1], atol=1e-12)
    assert abs(parameters[2]) < 1e-6


def test_centrosymmetric_onsite_odd_tensor_has_zero_allowed_dimension():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = make_supercell(primitive, (3, 3, 3))[0]
    calculation = FiniteDifferenceCalculation(primitive, order=3, reference=reference, cutoff=4.1)
    onsite = InteractionKey((0, 0, 0), ((0, 0, 0), (0, 0, 0)))
    assert all(
        orbit.representative != onsite
        for orbit in calculation.interaction_space.primitive_orbit_space.orbits
    )
