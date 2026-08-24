"""Basis-backend contract used by force-constant fitting."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from mlfcs.fitting.backends.result import BasisLoweringResult

PreparedBasis = Any


class FittingBasisBackend(Protocol):
    """Uniform internal interface for fitting-coordinate implementations."""

    name: str

    def prepare(
        self,
        *,
        calculations,
        training_displacements: np.ndarray,
        parameterizations,
        n_parameters: int,
        batch_size: int,
        parameter_map,
        reporter,
        device,
    ) -> PreparedBasis: ...

    def build_operator(self, prepared: PreparedBasis, displacements: np.ndarray): ...

    def predict(
        self,
        prepared: PreparedBasis,
        displacements: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray: ...

    def lower(
        self,
        prepared: PreparedBasis,
        parameters: np.ndarray,
    ) -> BasisLoweringResult: ...


__all__ = ["FittingBasisBackend", "PreparedBasis"]
