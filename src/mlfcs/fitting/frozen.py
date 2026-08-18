"""Preparation and force evaluation for externally frozen Taylor IFC orders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import factorial

import numpy as np

from mlfcs.fitting.constraints import _image_columns
from mlfcs.io.export import build_export_view
from mlfcs.model import ForceConstants, SparseOrderForceConstants


@dataclass(frozen=True, slots=True)
class FrozenOrderDiagnostic:
    reconstruction_maximum: float
    reconstruction_relative: float
    missing_support: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedFrozenForceConstants:
    sparse: dict[int, SparseOrderForceConstants]
    diagnostics: dict[int, FrozenOrderDiagnostic]

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.sparse))


def prepare_frozen_force_constants(
    values: Mapping[int, ForceConstants] | None,
    *,
    primitive,
    reference,
    calculations,
) -> PreparedFrozenForceConstants:
    """Validate and align a consecutive low-order frozen Taylor prefix."""
    if values is None:
        return PreparedFrozenForceConstants({}, {})
    if not isinstance(values, Mapping):
        raise TypeError("frozen_force_constants must be a mapping from order to ForceConstants")
    orders = tuple(sorted(values))
    if orders and orders != tuple(range(2, orders[-1] + 1)):
        raise ValueError("frozen force-constant orders must be a consecutive prefix starting at 2")
    fitted_orders = tuple(calculation.config.order for calculation in calculations)
    if orders and (
        orders[-1] >= max(fitted_orders) or any(order not in fitted_orders for order in orders)
    ):
        raise ValueError("at least one fitted order above the frozen low-order prefix is required")

    by_order = {calculation.config.order: calculation for calculation in calculations}
    aligned = {}
    diagnostics = {}
    for order in orders:
        source = values[order]
        if not isinstance(source, ForceConstants):
            raise TypeError(f"frozen FC{order} value must be a ForceConstants object")
        if order not in source.sparse:
            raise ValueError(f"frozen ForceConstants does not contain sparse FC{order}")
        view = build_export_view(source, primitive=primitive, supercell=reference).force_constants
        sparse_order = view.sparse[order]
        aligned[order] = sparse_order
        diagnostics[order] = _representation_diagnostic(sparse_order, by_order[order])
    return PreparedFrozenForceConstants(aligned, diagnostics)


def frozen_forces(prepared, displacements, index) -> tuple[np.ndarray, dict[int, float]]:
    """Evaluate physical Taylor forces without dense IFC materialization."""
    displacements = np.asarray(displacements, dtype=float)
    total = np.zeros_like(displacements)
    rms = {}
    translations = np.unique(index.translations, axis=0)
    for order, values in prepared.sparse.items():
        contribution = np.zeros_like(displacements)
        coefficient = -1.0 / factorial(order - 1)
        for sites, relative, tensor in zip(
            values.sites, values.translation_representatives, values.tensors, strict=True
        ):
            for shift in translations:
                atoms = [index.atom(int(sites[0]), shift)]
                atoms.extend(
                    index.atom(int(site), shift + vector)
                    for site, vector in zip(sites[1:], relative, strict=True)
                )
                operands = [tensor, list(range(order))]
                for axis, atom in enumerate(atoms[1:], start=1):
                    operands.extend([displacements[:, atom, :], [order, axis]])
                operands.append([order, 0])
                contribution[:, atoms[0], :] += coefficient * np.einsum(*operands, optimize=True)
        total += contribution
        rms[order] = float(np.sqrt(np.mean(contribution**2)))
    return total, rms


def frozen_asr_residual(values, index) -> float:
    """Return the maximum direct atomic-sum residual of one sparse IFC order."""
    translations = np.unique(index.translations, axis=0)
    maximum = 0.0
    for summed_axis in range(1, values.order):
        sums = {}
        for sites, relative, tensor in zip(
            values.sites, values.translation_representatives, values.tensors, strict=True
        ):
            for shift in translations:
                atoms = [index.atom(int(sites[0]), shift)]
                atoms.extend(
                    index.atom(int(site), shift + vector)
                    for site, vector in zip(sites[1:], relative, strict=True)
                )
                key = tuple(atom for axis, atom in enumerate(atoms) if axis != summed_axis)
                sums[key] = sums.get(key, np.zeros_like(tensor)) + tensor
        maximum = max(
            maximum,
            max(
                (float(np.max(np.abs(value), initial=0.0)) for value in sums.values()), default=0.0
            ),
        )
    return maximum


def _representation_diagnostic(values, calculation) -> FrozenOrderDiagnostic:
    targets = {}
    for cluster, tensor in zip(values.clusters, values.tensors, strict=True):
        targets.setdefault(tuple(int(atom) for atom in cluster), []).append(tensor.reshape(-1))
    targets = {key: np.mean(rows, axis=0) for key, rows in targets.items()}
    grouped = {}
    current_support = set()
    for cluster, columns, offset in _image_columns(calculation):
        current_support.add(cluster)
        if cluster in targets:
            grouped.setdefault(offset, []).append((columns, targets[cluster]))
    squared_error = 0.0
    squared_norm = 0.0
    maximum = 0.0
    for rows in grouped.values():
        matrix = np.vstack([columns for columns, _target in rows])
        target = np.concatenate([target for _columns, target in rows])
        residual = matrix @ np.linalg.lstsq(matrix, target, rcond=None)[0] - target
        squared_error += float(residual @ residual)
        squared_norm += float(target @ target)
        maximum = max(maximum, float(np.max(np.abs(residual), initial=0.0)))
    missing = set(targets).difference(current_support)
    for cluster in missing:
        target = targets[cluster]
        squared_error += float(target @ target)
        squared_norm += float(target @ target)
        maximum = max(maximum, float(np.max(np.abs(target), initial=0.0)))
    digest = sha256()
    for array in (values.sites, values.translation_representatives, values.tensors):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    relative = np.sqrt(squared_error / squared_norm) if squared_norm else 0.0
    return FrozenOrderDiagnostic(maximum, float(relative), len(missing), digest.hexdigest())
