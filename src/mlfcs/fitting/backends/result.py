"""Basis-independent lowering result."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LoweringResult:
    """Canonical Taylor parameters produced from fitted coordinates."""

    taylor_parameters: np.ndarray
    reference_fc1: np.ndarray | None = None
    lowering_force_maximum: float | None = None
    lowering_force_relative: float | None = None
