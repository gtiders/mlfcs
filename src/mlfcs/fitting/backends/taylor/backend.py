"""Taylor fitting backend with identity physical lowering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.fitting.backends.result import LoweringResult
from mlfcs.fitting.backends.taylor.features import taylor_axis_derivatives
from mlfcs.fitting.design_operator import ForceDesignOperator


@dataclass(slots=True)
class PreparedTaylorBasis:
    calculations: tuple
    operator: ForceDesignOperator


class TaylorFittingBackend:
    """Ordinary Taylor fitting coordinates and identity lowering."""

    name = "taylor"

    def prepare(
        self,
        *,
        calculations,
        training_displacements,
        parameterizations,
        n_parameters,
        batch_size,
        parameter_map,
        device,
    ) -> PreparedTaylorBasis:
        operator = ForceDesignOperator(
            training_displacements,
            np.empty(0, dtype=float),
            parameterizations,
            n_parameters,
            batch_size,
            parameter_map=parameter_map,
            device=device,
            axis_derivatives=taylor_axis_derivatives,
        )
        return PreparedTaylorBasis(tuple(calculations), operator)

    def build_operator(self, prepared, displacements):
        return prepared.operator.with_displacements(displacements)

    def predict(self, prepared, displacements, parameters):
        return self.build_operator(prepared, displacements).matvec(parameters)

    def lower(self, prepared, parameters) -> LoweringResult:
        del prepared
        return LoweringResult(np.asarray(parameters, dtype=float).copy())


__all__ = ["PreparedTaylorBasis", "TaylorFittingBackend"]
