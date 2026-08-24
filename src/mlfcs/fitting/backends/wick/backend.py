"""Wick fitting backend and its prepared fitting state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.fitting.backends.result import BasisDiagnostics, BasisLoweringResult
from mlfcs.fitting.backends.wick.covariance import symmetrized_covariance
from mlfcs.fitting.backends.wick.lowering import (
    build_wick_to_taylor_transform,
    lowered_fc1,
)
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.design_operator import ForceDesignOperator


@dataclass(slots=True)
class PreparedWickBasis:
    calculations: tuple
    covariance: np.ndarray
    operator: ForceDesignOperator


@dataclass(frozen=True, slots=True)
class WickDiagnostics(BasisDiagnostics):
    """Covariance and lowering diagnostics specific to Wick coordinates."""

    covariance: np.ndarray | None = None


class WickFittingBackend:
    """Covariance-orthogonalized fitting coordinates with Taylor lowering."""

    name = "wick"

    def prepare(
        self,
        *,
        calculations,
        training_displacements,
        parameterizations,
        n_parameters,
        batch_size,
        parameter_map,
        reporter,
        device,
    ) -> PreparedWickBasis:
        calculations = tuple(calculations)
        covariance = symmetrized_covariance(training_displacements, calculations[0])
        operator = ForceDesignOperator(
            training_displacements,
            covariance,
            parameterizations,
            n_parameters,
            batch_size,
            parameter_map=parameter_map,
            reporter=reporter,
            device=device,
            axis_derivatives=wick_axis_derivatives,
        )
        return PreparedWickBasis(calculations, covariance, operator)

    def build_operator(self, prepared, displacements):
        return prepared.operator.with_displacements(displacements)

    def predict(self, prepared, displacements, parameters):
        return self.build_operator(prepared, displacements).matvec(parameters)

    def lower(self, prepared, parameters) -> BasisLoweringResult:
        transform = build_wick_to_taylor_transform(prepared.calculations, prepared.covariance)
        fc1 = lowered_fc1(prepared.calculations, parameters, prepared.covariance)
        return BasisLoweringResult(
            np.asarray(transform @ np.asarray(parameters)),
            WickDiagnostics(
                covariance=prepared.covariance.copy(),
                reference_fc1=np.asarray(fc1),
                details={"folding_policy": "accepted_assumption"},
            ),
        )


__all__ = ["PreparedWickBasis", "WickDiagnostics", "WickFittingBackend"]
