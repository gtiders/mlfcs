"""Public finite-temperature lattice-dynamics API."""

from importlib import import_module
from typing import TYPE_CHECKING

from mlfcs.anharmonic import LoopSCPH, LoopSCPHResult, harmonic_frequencies

if TYPE_CHECKING:
    from mlfcs.sscha import SSCHA, EnsembleDiagnostics, HarmonicEnsemble, SSCHAIteration

__all__ = [
    "SSCHA",
    "EnsembleDiagnostics",
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "SSCHAIteration",
    "harmonic_frequencies",
]


def __getattr__(name: str):
    """Load SSCHA only when its public symbols are requested."""
    if name in {"EnsembleDiagnostics", "HarmonicEnsemble", "SSCHA", "SSCHAIteration"}:
        return getattr(import_module("mlfcs.sscha"), name)
    raise AttributeError(name)
