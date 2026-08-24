from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mlfcs.fitting.backends.interface import FittingBasisBackend
from mlfcs.fitting.backends.result import BasisDiagnostics, BasisLoweringResult
from mlfcs.fitting.backends.taylor.backend import TaylorFittingBackend
from mlfcs.fitting.backends.taylor.features import taylor_axis_derivatives
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives


def test_backend_result_objects_are_basis_independent():
    diagnostics = BasisDiagnostics(details={"backend": "test"})
    result = BasisLoweringResult(taylor_parameters=[1.0], diagnostics=diagnostics)

    assert result.diagnostics.details == {"backend": "test"}
    assert FittingBasisBackend is not None


def test_taylor_features_equal_zero_covariance_wick_features():
    displacement = jnp.asarray([[0.2, -0.3, 0.5]])
    coordinates = jnp.asarray([[[0, 1, 2, 0]]])
    covariance = jnp.zeros((3, 3))

    taylor_values = taylor_axis_derivatives(displacement, jnp.empty(0), coordinates, 4)
    wick_values = wick_axis_derivatives(displacement, covariance, coordinates, 4)

    for actual, expected in zip(taylor_values, wick_values, strict=True):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_taylor_lowering_is_identity():
    parameters = np.asarray([1.0, -2.0, 3.0])
    lowered = TaylorFittingBackend().lower(None, parameters)

    np.testing.assert_array_equal(lowered.taylor_parameters, parameters)
    assert lowered.diagnostics.covariance is None
    assert lowered.diagnostics.reference_fc1 is None
