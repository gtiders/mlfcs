"""Direct Wick force prediction for backend tests and diagnostics."""

from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.design_operator import predict_force


def predict_wick_force(parameters, displacements, covariance, parameterizations):
    return predict_force(
        parameters,
        displacements,
        covariance,
        parameterizations,
        wick_axis_derivatives,
    )


__all__ = ["predict_wick_force"]
