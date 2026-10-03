"""Shared failures raised by the rewritten scientific core."""


class MLFCSError(Exception):
    """Base class for errors with domain meaning in MLFCS."""


class AliasingError(MLFCSError):
    """A supercell cannot distinguish all requested primitive parameters."""


class UnobservedParameterError(MLFCSError):
    """Training structures do not excite every requested model parameter."""


class ConstraintProjectionError(MLFCSError):
    """A physical invariance projection did not reach its requested accuracy."""


class IntegerRangeError(MLFCSError, ArithmeticError):
    """An exact integer result cannot be represented by the requested dtype."""


class RankCertificateError(MLFCSError, ArithmeticError):
    """Exact modular arithmetic could not certify a matrix rank."""


__all__ = [
    "AliasingError",
    "ConstraintProjectionError",
    "IntegerRangeError",
    "MLFCSError",
    "RankCertificateError",
    "UnobservedParameterError",
]
