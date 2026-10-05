"""Failures raised by the scientific model and algebra operations."""


class MLFCSError(Exception):
    """Base class for errors with domain meaning in MLFCS."""


class AliasingError(MLFCSError):
    """A supercell cannot distinguish all requested primitive parameters."""


class UnobservedParameterError(MLFCSError):
    """Training structures do not excite every requested model parameter."""


class ConstraintProjectionError(MLFCSError):
    """A physical invariance projection did not reach its requested accuracy."""


class ArithmeticRangeError(MLFCSError, ArithmeticError):
    """An arithmetic value cannot be represented by the requested dtype."""


class RankError(MLFCSError, ArithmeticError):
    """Available modular computations cannot establish the required rank or kernel identity."""


__all__ = [
    "AliasingError",
    "ArithmeticRangeError",
    "ConstraintProjectionError",
    "MLFCSError",
    "RankError",
    "UnobservedParameterError",
]
