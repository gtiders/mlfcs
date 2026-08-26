"""Serializable, device-independent least-squares sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GramStatistics:
    """Sufficient statistics for a fixed fitting design and target."""

    gram: np.ndarray
    rhs: np.ndarray
    target_norm: float
    n_equations: int
    metadata: dict[str, Any]

    def merge(self, other: GramStatistics) -> GramStatistics:
        """Combine statistics only when they describe the same design space."""
        if not isinstance(other, GramStatistics):
            raise TypeError("can only merge GramStatistics")
        for key in ("fitting_basis",):
            if self.metadata.get(key) != other.metadata.get(key):
                raise ValueError(f"incompatible Gram metadata: {key}")
        if self.gram.shape != other.gram.shape or self.rhs.shape != other.rhs.shape:
            raise ValueError("incompatible Gram dimensions")
        return GramStatistics(
            self.gram + other.gram,
            self.rhs + other.rhs,
            self.target_norm + other.target_norm,
            self.n_equations + other.n_equations,
            dict(self.metadata),
        )

    def exact_column_scale(self):
        norm = np.sqrt(np.maximum(np.diag(self.gram), 0.0))
        threshold = max(float(np.max(norm)) * 1e-12, np.finfo(float).tiny)
        result = np.zeros_like(norm)
        active = norm > threshold
        result[active] = 1.0 / norm[active]
        return result

    def force_metrics(self, parameters):
        residual_squared = max(
            float(parameters @ self.gram @ parameters - 2 * parameters @ self.rhs + self.target_norm),
            0.0,
        )
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

    def solve(self, scale, constraints, *, tolerance, max_iterations):
        from mlfcs.fitting.linear_solvers import solve_gram_system

        return solve_gram_system(
            self.gram, self.rhs, self.target_norm, scale, constraints,
            tolerance=tolerance, max_iterations=max_iterations,
        )

    def save(self, path: str | Path) -> None:
        """Write statistics and metadata in a portable NumPy archive."""
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
            if not isinstance(value, np.ndarray) and isinstance(value, (str, int, float, bool, type(None)))
        }
        for key in tuple(serializable):
            if key == "parameter_map":
                serializable.pop(key)
        if isinstance(self.metadata.get("parameter_map"), np.ndarray):
            arrays["metadata::parameter_map"] = self.metadata["parameter_map"]
        elif self.metadata.get("parameter_map") is not None:
            arrays["metadata::parameter_map"] = self.metadata["parameter_map"].toarray()
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
            return cls(
                archive["gram"].copy(),
                archive["rhs"].copy(),
                float(archive["target_norm"]),
                int(archive["n_equations"]),
                metadata,
            )
