import numpy as np
import pytest
from ase import Atoms
from scipy import sparse
from supercell_helpers import make_supercell

from mlfcs.constraints import TranslationalASRProjector
from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
from mlfcs.interactions.keys import InteractionKey


def test_shared_asr_projection_is_idempotent_and_satisfies_constraints():
    constraints = sparse.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    projector = TranslationalASRProjector(2, 3, constraints)
    values = np.array([2.0, -3.0, 7.0])
    first = projector.project(values, tolerance=1e-12)
    second = projector.project(first.parameters, tolerance=1e-12)

    np.testing.assert_allclose(constraints @ first.parameters, 0.0, atol=1e-12)
    np.testing.assert_allclose(second.parameters, first.parameters, atol=1e-12)


def test_shared_asr_projection_matches_dense_euclidean_oracle_with_redundant_rows():
    constraints = np.array(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -2.0, 1.0],
        ]
    )
    values = np.array([2.0, -3.0, 7.0, 1.5, -4.0])
    projector = TranslationalASRProjector(3, len(values), sparse.csr_matrix(constraints))

    result = projector.project(values, tolerance=1e-12)
    oracle = values - constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ (
        constraints @ values
    )

    np.testing.assert_allclose(result.parameters, oracle, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(constraints @ result.parameters, 0.0, atol=1e-11)


def test_shared_asr_projection_defines_empty_and_already_feasible_cases():
    empty = TranslationalASRProjector(2, 0, sparse.csr_matrix((0, 0))).project(
        np.empty(0), tolerance=1e-12
    )
    unconstrained = TranslationalASRProjector(2, 3, sparse.csr_matrix((0, 3))).project(
        np.array([1.0, 2.0, 3.0]), tolerance=1e-12
    )
    feasible = TranslationalASRProjector(2, 3, sparse.csr_matrix([[1.0, -1.0, 0.0]])).project(
        np.array([2.0, 2.0, 0.0]), tolerance=1e-12
    )

    assert empty.parameters.size == 0
    np.testing.assert_array_equal(unconstrained.parameters, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(feasible.parameters, [2.0, 2.0, 0.0])
    assert empty.iterations == unconstrained.iterations == feasible.iterations == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_shared_asr_projection_rejects_nonfinite_parameters_before_scipy(bad):
    projector = TranslationalASRProjector(2, 2, sparse.csr_matrix([[1.0, -1.0]]))

    with pytest.raises(ValueError, match="NaN or infinite"):
        projector.project(np.array([bad, 1.0]), tolerance=1e-12)


def test_shared_asr_projection_names_nonconvergence(monkeypatch):
    from mlfcs.constraints import translational

    projector = TranslationalASRProjector(4, 2, sparse.csr_matrix([[1.0, -1.0]]))

    def no_progress(matrix, residual, **kwargs):
        del matrix, kwargs
        return np.zeros(2), 0, 1, float(np.linalg.norm(residual))

    monkeypatch.setattr(translational, "lsmr", no_progress)
    with pytest.raises(
        RuntimeError,
        match=r"order-4 ASR projection did not converge for 2 parameters: residual .* -> .*,",
    ):
        projector.project(np.array([1.0, -1.0]), tolerance=1e-12)


def test_post_fit_projection_is_not_constrained_least_squares():
    design = np.diag([1.0, 4.0])
    target = np.array([1.0, 1.0])
    constraints = sparse.csr_matrix([[1.0, -1.0]])
    raw = np.linalg.solve(design, target)
    post = (
        TranslationalASRProjector(2, 2, constraints)
        .project(
            raw,
            tolerance=1e-12,
        )
        .parameters
    )
    constrained_value = float(np.linalg.lstsq(design @ np.ones((2, 1)), target, rcond=None)[0][0])
    constrained = np.full(2, constrained_value)

    assert not np.allclose(post, constrained)


def test_centrosymmetric_onsite_odd_tensor_has_zero_allowed_dimension():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = make_supercell(primitive, (3, 3, 3))[0]
    calculation = FiniteDifferenceCalculation(primitive, order=3, reference=reference, cutoff=4.1)
    onsite = InteractionKey((0, 0, 0), ((0, 0, 0), (0, 0, 0)))
    assert all(
        orbit.representative != onsite
        for orbit in calculation.interaction_space.primitive_orbit_space.orbits
    )
