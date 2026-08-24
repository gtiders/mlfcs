"""Single dispatch point for fitting-coordinate backends."""

from __future__ import annotations


def create_fitting_backend(name: str):
    """Construct a supported fitting backend without eager backend imports."""
    normalized = str(name).casefold()
    if normalized == "wick":
        from mlfcs.fitting.backends.wick.backend import WickFittingBackend

        return WickFittingBackend()
    if normalized == "taylor":
        from mlfcs.fitting.backends.taylor.backend import TaylorFittingBackend

        return TaylorFittingBackend()
    raise ValueError("fitting_basis must be 'taylor' or 'wick'")


__all__ = ["create_fitting_backend"]
