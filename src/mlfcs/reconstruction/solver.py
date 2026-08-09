from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms

from mlfcs.core.geometry import SupercellIndex
from mlfcs.core.orbits import OrbitSpace
from mlfcs.finite_difference.sampling import DisplacementKey
from mlfcs.model import SparseOrderForceConstants
from mlfcs.reconstruction.asr import (
    maximum_acoustic_sum_rule_drift,
    project_acoustic_sum_rule,
    project_sum_rules,
)


def reconstruct_compact(
    orbit_space: OrbitSpace,
    index: SupercellIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
    enforce_rotational: bool = False,
    supercell: Atoms | None = None,
    report: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Reconstruct a compact, translation-reduced IFC tensor orbit by orbit."""
    sparse_result = reconstruct_sparse(
        orbit_space,
        index,
        derivatives,
        enforce_asr=enforce_asr,
        enforce_rotational=enforce_rotational,
        supercell=supercell,
        report=report,
    )
    return sparse_result.to_dense(max_bytes=None)


def reconstruct_sparse(
    orbit_space: OrbitSpace,
    index: SupercellIndex,
    derivatives: dict[DisplacementKey, np.ndarray],
    *,
    enforce_asr: bool = True,
    enforce_rotational: bool = False,
    supercell: Atoms | None = None,
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

    if enforce_rotational:
        if supercell is None:
            raise ValueError("supercell is required to enforce rotational sum rules")
        pivot_values, drifts = project_sum_rules(
            orbit_space,
            pivot_values,
            supercell=supercell,
            acoustic=enforce_asr,
            rotational=True,
        )
        if report is not None:
            before, after = drifts["translational"]
            suffix = "" if enforce_asr else " (ASR disabled)"
            report(
                f"- Max drift of fc{order}: {before:.10e} -> {after:.10e} "
                f"eV/angstrom^{order}{suffix}"
            )
            before, after = drifts["rotational"]
            rotational_unit = (
                "eV/angstrom" if order == 2 else f"eV/angstrom^{order - 1}"
            )
            report(
                f"- Max rotational drift of fc{order}: {before:.10e} -> "
                f"{after:.10e} {rotational_unit}"
            )
    elif enforce_asr:
        pivot_values, initial_drift, final_drift = project_acoustic_sum_rule(
            orbit_space, pivot_values, return_drift=True
        )
        if report is not None:
            report(
                f"- Max drift of fc{order}: {initial_drift:.10e} -> "
                f"{final_drift:.10e} eV/angstrom^{order}"
            )
    elif report is not None:
        drift = maximum_acoustic_sum_rule_drift(orbit_space, pivot_values)
        report(f"- Max drift of fc{order}: {drift:.10e} eV/angstrom^{order} (ASR disabled)")
    clusters: list[tuple[int, ...]] = []
    tensors: list[np.ndarray] = []
    for orbit, values in zip(orbit_space.orbits, pivot_values, strict=True):
        pivot_basis = orbit.basis[orbit.pivots]
        coefficients = np.linalg.solve(pivot_basis, values)
        representative = orbit.basis @ coefficients
        for image in orbit.images:
            tensor = image.action.apply_flat(representative).reshape((3,) * order)
            clusters.append(image.cluster)
            tensors.append(tensor)
    return SparseOrderForceConstants(
        order=order,
        n_primitive=index.n_primitive,
        n_supercell=len(index.primitive),
        clusters=np.asarray(clusters, dtype=np.int32).reshape((-1, order)),
        tensors=np.asarray(tensors, dtype=float).reshape((-1,) + (3,) * order),
    )
