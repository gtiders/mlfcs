"""Internal fitting-coordinate backends.

Backends own basis-specific feature evaluation and lowering.  The fitting
orchestrator consumes only the protocol and result objects defined here.
"""

from mlfcs.fitting.backends.interface import FittingBasisBackend, PreparedBasis
from mlfcs.fitting.backends.result import LoweringResult

__all__ = [
    "FittingBasisBackend",
    "LoweringResult",
    "PreparedBasis",
]
