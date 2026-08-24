class MLFCSError(Exception):
    """Base class for MLFCS domain failures."""


class StructureCompatibilityError(MLFCSError, ValueError):
    """Structures cannot be related without changing their physical definition."""


class IdentifiabilityError(MLFCSError, ValueError):
    """The requested interaction model is not identifiable from its reference."""


class InteractionAliasingError(IdentifiabilityError):
    """Distinct exact interactions fold onto the same observable response."""


class ConvergenceError(MLFCSError, RuntimeError):
    """An iterative physical or numerical procedure did not converge."""


class SerializationError(MLFCSError, ValueError):
    """A force-constant representation cannot be serialized or restored."""


class UnsupportedOperationError(MLFCSError, ValueError):
    """A requested operation is outside the supported physical semantics."""
