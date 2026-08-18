from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mlfcs.clusters.orbits import OrbitSpace
from mlfcs.finite_difference.sampling import DisplacementKey
from mlfcs.ifc.expansion import expand_orbit_parameters
from mlfcs.ifc.model import SparseOrderForceConstants
from mlfcs.reconstruction.asr import maximum_acoustic_sum_rule_drift, project_acoustic_sum_rule
from mlfcs.structure.geometry import PeriodicIndex


def reconstruct_sparse(
    orbit_space: OrbitSpace,
    index: PeriodicIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
    report: Callable[[str], None] | None = None,
) -> SparseOrderForceConstants:
    """Reconstruct only symmetry-generated cluster tensors."""
    order = orbit_space.order
    pivot_values: list[np.ndarray] = []
    for orbit in orbit_space.orbits:
        values: list[float] = []
        for pivot in orbit.pivots:
            components = np.unravel_index(int(pivot), (3,) * order)
            key = tuple(
                (orbit.representative[axis], int(components[axis])) for axis in range(order - 1)
            )
            values.append(derivatives[key][orbit.representative[-1], int(components[-1])])
        pivot_values.append(np.asarray(values))

    original_parameters = np.concatenate(pivot_values) if pivot_values else np.empty(0, dtype=float)

    if enforce_asr:
        pivot_values, initial_drift, final_drift = project_acoustic_sum_rule(
            orbit_space, pivot_values, return_drift=True
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
                pivot_values,
                label="ASR",
            )
    elif report is not None:
        drift = maximum_acoustic_sum_rule_drift(orbit_space, pivot_values)
        report(f"- Max drift of fc{order}: {drift:.10e} eV/angstrom^{order} (ASR disabled)")
    return expand_orbit_parameters(
        orbit_space,
        np.concatenate(pivot_values) if pivot_values else np.empty(0, dtype=float),
        n_primitive=index.n_primitive,
        n_supercell=len(index.primitive),
        index=index,
    )


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
