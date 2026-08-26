"""Taylor fitting model and its design-operator preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.fitting.design_operator import ForceDesignOperator
from mlfcs.fitting.taylor.features import taylor_axis_derivatives


@dataclass(frozen=True, slots=True)
class TaylorResult:
    """Taylor parameters after the identity lowering step."""

    taylor_parameters: np.ndarray
    reference_fc1: np.ndarray | None = None


@dataclass(slots=True)
class PreparedTaylorBasis:
    calculations: tuple
    operator: ForceDesignOperator


class TaylorModel:
    """Ordinary Taylor fitting coordinates."""

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

    def lower(self, prepared, parameters) -> TaylorResult:
        del prepared
        return TaylorResult(np.asarray(parameters, dtype=float).copy())


__all__ = ["PreparedTaylorBasis", "TaylorModel", "TaylorResult"]
