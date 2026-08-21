"""Finite-temperature SSCHA and self-consistent phonon methods."""

from importlib import import_module
from typing import TYPE_CHECKING

from mlfcs.anharmonic.common.schedule import TemperatureSeriesResult
from mlfcs.anharmonic.scph import LoopSCPH, LoopSCPHResult, harmonic_frequencies

if TYPE_CHECKING:
    from mlfcs.anharmonic.sscha import (
        SSCHA,
        EnsembleDiagnostics,
        HarmonicEnsemble,
        SSCHAIteration,
        SSCHAResult,
    )

__all__ = [
    "SSCHA",
    "EnsembleDiagnostics",
    "HarmonicEnsemble",
    "LoopSCPH",
    "LoopSCPHResult",
    "SSCHAIteration",
    "SSCHAResult",
    "TemperatureSeriesResult",
    "harmonic_frequencies",
]


def __getattr__(name: str):
    """Load SSCHA only when its public symbols are requested."""
    if name in {"EnsembleDiagnostics", "HarmonicEnsemble", "SSCHA", "SSCHAIteration", "SSCHAResult"}:
        return getattr(import_module("mlfcs.anharmonic.sscha"), name)
    raise AttributeError(name)
