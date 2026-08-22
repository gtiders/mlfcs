import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from mlfcs.fitting.model import (
    _BatchedForceOperator,
    _CachedForceOperator,
    _force_metrics,
    _OrderTensor,
    _StreamingGramSystem,
    _wick,
    _wick_to_taylor_sparse,
)
from mlfcs.model import SparseOrderForceConstants


def _one_parameter_fc2_tensor():
    representative = np.zeros((1, 9, 1))
    representative[0, 0, 0] = 1.0
    components = np.asarray(tuple(np.ndindex(3, 3)), dtype=np.int32)
    coordinates = components.reshape(1, 1, 1, 9, 2)
    return _OrderTensor(
        order=2,
        parameter_indices=np.zeros((1, 1), dtype=np.int32),
        parameter_mask=np.ones((1, 1), dtype=bool),
        representative_from_pivots=representative,
        rotations=np.eye(3).reshape(1, 1, 3, 3),
        component_permutations=np.arange(9).reshape(1, 1, 9),
        coordinates=coordinates,
        image_mask=np.ones((1, 1), dtype=bool),
    )


def test_matrix_free_force_operator_has_consistent_adjoint():
    rng = np.random.default_rng(3)
    displacement = rng.normal(size=(7, 1, 3))
    operator = _BatchedForceOperator(
        displacement, np.eye(3), (_one_parameter_fc2_tensor(),), 1, batch_size=3
    )
    parameters = np.array([2.7])
    residual = rng.normal(size=operator.shape[0])

    np.testing.assert_allclose(
        np.vdot(operator.matvec(parameters), residual),
        np.vdot(parameters, operator.rmatvec(residual)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_scaled_lsmr_recovers_force_constant_and_alamode_error():
    rng = np.random.default_rng(8)
    displacement = rng.normal(size=(12, 1, 3))
    operator = _BatchedForceOperator(
        displacement, np.eye(3), (_one_parameter_fc2_tensor(),), 1, batch_size=4
    )
    expected = np.array([4.25])
    force = operator.matvec(expected)
    scale = operator.estimate_column_scale(32, rng)
    solution = lsmr(operator.scaled(scale), force, atol=1e-12, btol=1e-12)
    actual = solution[0] * scale

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    rmse, relative = _force_metrics(operator.matvec(actual), force)
    assert rmse < 1e-12
    assert 100 * relative < 1e-9


def test_wick_recursion_supports_arbitrary_degree():
    displacement = np.array([[2.0, 3.0, 5.0]])
    covariance = np.diag([0.5, 0.7, 1.1])
    coordinates = np.array([0, 0, 1, 2])

    actual = _wick(displacement, covariance, coordinates, order=4)
    expected = (2.0**2 - 0.5) * 3.0 * 5.0

    np.testing.assert_allclose(actual, expected)


def test_wick_sparse_coefficients_are_converted_to_taylor_coefficients():
    fc3 = SparseOrderForceConstants(3, 1, 1, np.zeros((1, 3), dtype=int), np.ones((1, 3, 3, 3)))
    fc5 = SparseOrderForceConstants(5, 1, 1, np.zeros((1, 5), dtype=int), np.ones((1, 3, 3, 3, 3, 3)))
    covariance = np.diag([2.0, 3.0, 5.0])

    converted = _wick_to_taylor_sparse({3: fc3, 5: fc5}, covariance)

    # :u^5: = u^5 - 10 sigma u^3 + ..., while the IFC convention
    # divides the two potential terms by 5! and 3!, giving -sigma/2.
    np.testing.assert_allclose(converted[3].tensors[0], 1.0 - 0.5 * np.trace(covariance))
    np.testing.assert_allclose(converted[5].tensors, fc5.tensors)


def test_wick_to_taylor_conversion_preserves_polynomial_force():
    sigma = 0.7
    phi3 = 2.5
    phi5 = -1.2
    tensor3 = np.zeros((1, 3, 3, 3))
    tensor5 = np.zeros((1, 3, 3, 3, 3, 3))
    tensor3[(0, 0, 0, 0)] = phi3
    tensor5[(0, 0, 0, 0, 0, 0)] = phi5
    fc3 = SparseOrderForceConstants(3, 1, 1, np.zeros((1, 3), dtype=int), tensor3)
    fc5 = SparseOrderForceConstants(5, 1, 1, np.zeros((1, 5), dtype=int), tensor5)
    covariance = np.diag([sigma, 0.0, 0.0])
    converted = _wick_to_taylor_sparse({3: fc3, 5: fc5}, covariance)
    for displacement in (-1.3, -0.2, 0.8):
        wick_force = -phi3 * (displacement**2 - sigma) / 2
        wick_force -= phi5 * (displacement**4 - 6 * sigma * displacement**2 + 3 * sigma**2) / 24
        # Constants in the force are the derivative of an omitted FC1 term;
        # compare the displacement-dependent FC3+FC5 part represented by IFCs.
        taylor_force = -converted[3].tensors[(0, 0, 0, 0)] * displacement**2 / 2
        taylor_force -= converted[5].tensors[(0, 0, 0, 0, 0, 0)] * displacement**4 / 24
        wick_force_without_constant = wick_force - (phi3 * sigma / 2 - phi5 * sigma**2 / 8)
        np.testing.assert_allclose(taylor_force, wick_force_without_constant)


def test_cached_operator_matches_matrix_free_operator():
    rng = np.random.default_rng(9)
    displacement = rng.normal(size=(7, 1, 3))
    covariance = np.eye(3)
    tensor = _one_parameter_fc2_tensor()
    operator = _BatchedForceOperator(displacement, covariance, (tensor,), 1, batch_size=4)
    cached = _CachedForceOperator.from_operator(operator)
    try:
        parameters = np.array([3.5])
        residual = rng.normal(size=operator.shape[0])
        np.testing.assert_allclose(cached.matvec(parameters), operator.matvec(parameters))
        np.testing.assert_allclose(cached.rmatvec(residual), operator.rmatvec(residual))
        expected_scale = 1.0 / np.linalg.norm(operator.matvec(np.ones(1)))
        np.testing.assert_allclose(cached.exact_column_scale(), expected_scale)
    finally:
        cached.close()


def test_streaming_gram_matches_lsmr():
    rng = np.random.default_rng(12)
    displacement = rng.normal(size=(9, 1, 3))
    covariance = np.eye(3)
    tensor = _one_parameter_fc2_tensor()
    operator = _BatchedForceOperator(displacement, covariance, (tensor,), 1, batch_size=4)
    expected = np.array([2.75])
    target = operator.matvec(expected)
    gram = _StreamingGramSystem.from_operator(operator, target)
    scale = gram.exact_column_scale()
    actual = gram.solve(
        scale,
        sparse.csr_matrix((0, 1)),
        tolerance=1e-12,
        max_iterations=100,
        damping=0.0,
        verbose=False,
    )[0] * scale
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
