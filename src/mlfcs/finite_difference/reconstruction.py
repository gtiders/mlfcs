from __future__ import annotations

from collections.abc import Callable

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


def reconstruct_sparse(
    orbit_space: RealizedInteractionSpace,
    index: PeriodicIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
    report: Callable[[str], None] | None = None,
    primitive_interaction_space=None,
) -> SparseOrderForceConstants:
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
        coefficients, initial_drift, final_drift = project_acoustic_sum_rule(
            constraint_space, coefficients, return_drift=True
        )
        if report is not None:
            report(
                f"- Max drift of fc{order}: {initial_drift:.10e} -> "
                f"{final_drift:.10e} eV/angstrom^{order}"
            )
            _report_parameter_correction(
                report,
                order,
                original_parameters,
                coefficients,
                label="ASR",
            )
    elif report is not None:
        drift = maximum_acoustic_sum_rule_drift(
            primitive_interaction_space or orbit_space, coefficients
        )
        report(f"- Max drift of fc{order}: {drift:.10e} eV/angstrom^{order} (ASR disabled)")
    parameters = np.concatenate(coefficients) if coefficients else np.empty(0, dtype=float)
    if primitive_interaction_space is None:
        raise ValueError("reconstruction requires a primitive exact interaction space")
    return expand_primitive_parameters(primitive_interaction_space, parameters)


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
