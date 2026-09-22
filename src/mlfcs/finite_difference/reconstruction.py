from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mlfcs.constraints.translational import (
    maximum_acoustic_sum_rule_drift,
    project_acoustic_sum_rule,
)
from mlfcs.finite_difference.sampling import DisplacementKey
from mlfcs.force_constants.expansion import expand_primitive_parameters
from mlfcs.force_constants.representation import SparseOrderForceConstants
from mlfcs.interactions.models import RealizedInteractionSpace
from mlfcs.structure.supercell_mapping import PeriodicIndex


@dataclass(frozen=True, slots=True)
class ASRProjectionReport:
    """Order-local ASR measurements attached to a reconstructed force constant set."""

    initial_residual: float
    final_residual: float
    correction_norm: float
    relative_correction: float
    iterations: int


def reconstruct_sparse(
    orbit_space: RealizedInteractionSpace,
    index: PeriodicIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
    asr_tolerance: float = 1e-10,
    report: Callable[[str], None] | None = None,
    primitive_interaction_space=None,
    return_diagnostics: bool = False,
) -> SparseOrderForceConstants | tuple[SparseOrderForceConstants, ASRProjectionReport]:
    """Reconstruct only symmetry-generated cluster tensors.

    The parameters of one orbit are the coefficients of its Cartesian basis $Q$, and the
    finite-difference plan measured the component rows ``observation_rows``.  Those rows
    therefore determine the parameters through the square system
    ``Q[observation_rows] @ theta = y``, which is solved explicitly here instead of
    assuming that an observed component *is* a parameter.
    """
    order = orbit_space.order
    coefficients: list[np.ndarray] = []
    for orbit in orbit_space.orbits:
        observed: list[float] = []
        for row in orbit.observation_rows:
            components = np.unravel_index(int(row), (3,) * order)
            key = tuple(
                (orbit.representative[axis], int(components[axis])) for axis in range(order - 1)
            )
            observed.append(derivatives[key][orbit.representative[-1], int(components[-1])])
        coefficients.append(np.linalg.solve(orbit.observation_matrix, np.asarray(observed)))

    original_parameters = np.concatenate(coefficients) if coefficients else np.empty(0, dtype=float)

    if enforce_asr:
        constraint_space = primitive_interaction_space or orbit_space
        coefficients, projection = project_acoustic_sum_rule(
            constraint_space,
            coefficients,
            tolerance=asr_tolerance,
            return_result=True,
        )
        if report is not None:
            report(
                f"- Max drift of fc{order}: {projection.initial_residual:.10e} -> "
                f"{projection.final_residual:.10e} eV/angstrom^{order}"
            )
            _report_parameter_correction(
                report,
                order,
                original_parameters,
                coefficients,
                label="ASR",
            )
            report(
                f"- ASR projection: relative correction="
                f"{projection.relative_correction:.10e}, iterations={projection.iterations}"
            )
        diagnostics = ASRProjectionReport(
            projection.initial_residual,
            projection.final_residual,
            projection.correction_norm,
            projection.relative_correction,
            projection.iterations,
        )
    else:
        drift = maximum_acoustic_sum_rule_drift(
            primitive_interaction_space or orbit_space, coefficients
        )
        if report is not None:
            report(f"- Max drift of fc{order}: {drift:.10e} eV/angstrom^{order} (ASR disabled)")
        diagnostics = ASRProjectionReport(drift, drift, 0.0, 0.0, 0)
    parameters = np.concatenate(coefficients) if coefficients else np.empty(0, dtype=float)
    if primitive_interaction_space is None:
        raise ValueError("reconstruction requires a primitive exact interaction space")
    reconstructed = expand_primitive_parameters(primitive_interaction_space, parameters)
    return (reconstructed, diagnostics) if return_diagnostics else reconstructed


def _report_parameter_correction(
    report: Callable[[str], None],
    order: int,
    original: np.ndarray,
    projected: list[np.ndarray],
    *,
    label: str,
) -> None:
    values = np.concatenate(projected) if projected else np.empty(0, dtype=float)
    correction = values - original
    maximum = float(np.max(np.abs(correction))) if len(correction) else 0.0
    denominator = max(float(np.linalg.norm(original)), np.finfo(float).tiny)
    relative = float(np.linalg.norm(correction) / denominator)
    report(
        f"- {label} parameter correction: maximum={maximum:.10e} "
        f"eV/angstrom^{order}, relative L2={relative:.10e}"
    )


__all__ = ["ASRProjectionReport", "reconstruct_sparse"]
