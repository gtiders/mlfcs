import numpy as np

from mlfcs.fitting.taylor import TaylorModel


def test_taylor_lowering_is_identity():
    parameters = np.asarray([1.0, -2.0, 3.0])
    result = TaylorModel().lower(None, parameters)
    np.testing.assert_array_equal(result.taylor_parameters, parameters)


def test_taylor_result_does_not_create_lower_order_terms():
    result = TaylorModel().lower(None, np.asarray([1.0]))
    assert result.reference_fc1 is None
