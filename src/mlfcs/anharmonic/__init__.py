"""Finite-temperature anharmonic lattice-dynamics methods."""

from mlfcs.anharmonic.scph import LoopSCPH, LoopSCPHResult, harmonic_frequencies

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
    if name in {"SSCHA", "SSCHAIteration", "HarmonicEnsemble", "EnsembleDiagnostics"}:
        from mlfcs.anharmonic import sscha

        return getattr(sscha, name)
    raise AttributeError(name)
