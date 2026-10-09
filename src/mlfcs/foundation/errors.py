"""Exception types shared by MLFCS domain models and numerical operations."""


class MLFCSError(Exception):
    """Base class for MLFCS-specific errors."""


class AliasingError(MLFCSError):
    """The supercell cannot distinguish all primitive force-constant parameters.

    Periodic folding aliases distinct primitive parameter directions, so the
    requested model is not structurally identifiable in the chosen supercell.
    """


class UnobservedParameterError(MLFCSError):
    """At least one fitting coordinate is absent from the training equations.

    This signals a completely unobserved column, not a general test for
    linear dependence or numerical rank deficiency.
    """


class ConstraintProjectionError(MLFCSError):
    """A physical-constraint projection failed to reach the requested tolerance."""


class ArithmeticRangeError(OverflowError, MLFCSError):
    """An exact numerical operation exceeds its supported arithmetic domain.

    This indicates a violated range contract rather than ordinary
    floating-point roundoff. It is an OverflowError so callers can handle
    domain-specific range failures through Python's standard overflow type.
    """


class RankError(MLFCSError, ArithmeticError):
    """An exact rank or kernel property could not be certified.

    This is raised when modular or integer evidence is insufficient to prove
    the required rank, nullity, or kernel identity.
    """


__all__ = [
    "AliasingError",
    "ArithmeticRangeError",
    "ConstraintProjectionError",
    "MLFCSError",
    "RankError",
    "UnobservedParameterError",
]
