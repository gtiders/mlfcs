"""Basis-independent lowering and diagnostic result objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class BasisDiagnostics:
    """Diagnostics owned by one fitting-coordinate backend."""

    covariance: np.ndarray | None = None
    reference_fc1: np.ndarray | None = None
    lowering_force_maximum: float | None = None
    lowering_force_relative: float | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BasisLoweringResult:
    """Canonical Taylor parameters produced from fitted coordinates."""

    taylor_parameters: np.ndarray
    diagnostics: BasisDiagnostics


__all__ = ["BasisDiagnostics", "BasisLoweringResult"]
