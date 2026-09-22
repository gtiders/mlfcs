"""Serializable, device-independent least-squares sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DESIGN_IDENTITY_FIELDS = (
    "design_schema",
    "design_fingerprint",
    "physical_parameter_count",
    "orders",
)


@dataclass(frozen=True, slots=True)
class GramStatistics:
    """Sufficient statistics for a fixed fitting design and target."""

    gram: np.ndarray
    rhs: np.ndarray
    target_norm: float
    n_equations: int
    metadata: dict[str, Any]

    @property
    def design_identity(self) -> dict[str, Any]:
        """Return the stable identity of the physical columns of this Gram."""
        missing = [field for field in DESIGN_IDENTITY_FIELDS if field not in self.metadata]
        if missing:
            raise ValueError(
                "these Gram statistics have no complete physical-design identity; "
                f"missing {missing}. Rebuild them with ForceConstantFitter.prepare_gram()."
            )
        return {field: self.metadata[field] for field in DESIGN_IDENTITY_FIELDS}

    def require_design(self, expected: dict[str, Any]) -> None:
        """Raise unless the statistics use exactly the expected physical columns."""
        identity = self.design_identity
        for field in DESIGN_IDENTITY_FIELDS:
            if identity[field] != expected[field]:
                raise ValueError(
                    "these Gram statistics were prepared for a different physical design: "
                    f"{field} is {identity[field]!r}, expected {expected[field]!r}"
                )

    def merge(self, other: GramStatistics) -> GramStatistics:
        """Combine statistics only when they describe the same design space."""
        if not isinstance(other, GramStatistics):
            raise TypeError("can only merge GramStatistics")
        if self.gram.shape != other.gram.shape or self.rhs.shape != other.rhs.shape:
            raise ValueError("incompatible Gram dimensions")
        first = self.design_identity
        second = other.design_identity
        for field in DESIGN_IDENTITY_FIELDS:
            if first[field] != second[field]:
                raise ValueError(
                    "cannot merge Gram statistics from different physical designs: "
                    f"{field} is {first[field]!r} here and {second[field]!r} there"
                )
        return GramStatistics(
            self.gram + other.gram,
            self.rhs + other.rhs,
            self.target_norm + other.target_norm,
            self.n_equations + other.n_equations,
            dict(self.metadata),
        )

    def exact_column_scale(self):
        diagonal = np.diag(self.gram)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0.0):
            raise ValueError("Gram diagonal must contain finite nonnegative column norms")
        norm = np.sqrt(diagonal)
        if norm.size == 0:
            return norm
        result = np.zeros_like(norm)
        active = norm > 0.0
        result[active] = 1.0 / norm[active]
        return result

    def force_metrics(self, parameters):
        model_norm = float(parameters @ self.gram @ parameters)
        cross = float(parameters @ self.rhs)
        residual_squared = model_norm - 2 * cross + self.target_norm
        cancellation_bound = (
            32 * np.finfo(float).eps * (abs(model_norm) + 2 * abs(cross) + abs(self.target_norm))
        )
        if abs(residual_squared) <= cancellation_bound:
            residual_squared = 0.0
        residual_squared = max(float(residual_squared), 0.0)
        relative = (
            float(np.sqrt(residual_squared / self.target_norm))
            if self.target_norm > 0
            else (0.0 if residual_squared == 0 else float("inf"))
        )
        rmse = float(np.sqrt(residual_squared / self.n_equations)) if self.n_equations else 0.0
        return rmse, relative

    def order_force_rms(self, parameters, orders, counts, n_equations):
        result = {}
        offset = 0
        for order, count in zip(orders, counts, strict=True):
            values = parameters[offset : offset + count]
            block = self.gram[offset : offset + count, offset : offset + count]
            result[order] = float(np.sqrt(max(float(values @ block @ values), 0.0) / n_equations))
            offset += count
        return result

    def solve(self, scale, *, tolerance, max_iterations):
        from mlfcs.fitting.linear_solvers import solve_gram_system

        return solve_gram_system(
            self.gram,
            self.rhs,
            self.target_norm,
            scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def save(self, path: str | Path) -> None:
        """Write statistics and metadata in a portable NumPy archive."""
        _ = self.design_identity
        arrays = {
            "gram": np.asarray(self.gram),
            "rhs": np.asarray(self.rhs),
            "target_norm": np.asarray(self.target_norm),
            "n_equations": np.asarray(self.n_equations, dtype=np.int64),
        }
        for key, value in self.metadata.items():
            if isinstance(value, np.ndarray):
                arrays[f"metadata::{key}"] = value
        serializable = {
            key: value
            for key, value in self.metadata.items()
            if not isinstance(value, np.ndarray)
            and isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))
        }
        arrays["metadata::json"] = np.asarray(serializable, dtype=object)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> GramStatistics:
        """Load a portable statistics archive without device-specific state."""
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive["metadata::json"].item()
            metadata.update(
                {
                    key.removeprefix("metadata::"): archive[key].copy()
                    for key in archive.files
                    if key.startswith("metadata::") and key != "metadata::json"
                }
            )
            statistics = cls(
                archive["gram"].copy(),
                archive["rhs"].copy(),
                float(archive["target_norm"]),
                int(archive["n_equations"]),
                metadata,
            )
        _ = statistics.design_identity
        return statistics
