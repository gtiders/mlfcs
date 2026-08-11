from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from ase import Atoms

from mlfcs.core.orbits import OrbitSpace
from mlfcs.finite_difference.sampling import (
    DisplacementKey,
    DisplacementPlan,
    build_displacement_plan,
)


@dataclass(frozen=True, slots=True)
class ExtrapolationMetrics:
    maximum_correction: float
    relative_l2_correction: float
    maximum_fit_residual: float


@dataclass(frozen=True, slots=True)
class ExtrapolationBackend:
    """Zero-step extrapolation from symmetric grids of positive step sizes."""

    displacement: float
    spacing: float
    side_steps: int = 1
    degree: int = 1

    def __post_init__(self) -> None:
        if self.displacement <= 0 or self.spacing <= 0:
            raise ValueError("displacement and extrapolation_spacing must be positive")
        if self.side_steps < 1:
            raise ValueError("extrapolation_side_steps must be at least one")
        if self.degree < 1:
            raise ValueError("extrapolation_degree must be at least one")
        if self.displacement - self.side_steps * self.spacing <= 0:
            raise ValueError("the extrapolation displacement grid must remain strictly positive")
        if self.degree >= len(self.grid):
            raise ValueError(
                "extrapolation_degree must be smaller than the number of displacement steps"
            )

    @property
    def grid(self) -> np.ndarray:
        offsets = np.arange(-self.side_steps, self.side_steps + 1, dtype=float)
        return self.displacement + self.spacing * offsets

    def plans(self, supercell: Atoms, orbit_space: OrbitSpace) -> tuple[DisplacementPlan, ...]:
        return tuple(
            build_displacement_plan(supercell, orbit_space, displacement=float(step))
            for step in self.grid
        )

    def plan_hash(self, plans: tuple[DisplacementPlan, ...]) -> str:
        digest = sha256()
        digest.update(b"mlfcs-extrapolate-v1")
        digest.update(np.ascontiguousarray(self.grid).tobytes())
        digest.update(np.int64(self.degree).tobytes())
        for plan in plans:
            digest.update(plan.hash.encode())
        return digest.hexdigest()

    def extrapolate(
        self,
        derivatives: list[dict[DisplacementKey, np.ndarray]],
    ) -> tuple[dict[DisplacementKey, np.ndarray], ExtrapolationMetrics]:
        if len(derivatives) != len(self.grid):
            raise ValueError("one derivative set is required for every extrapolation step")
        keys = tuple(derivatives[0])
        if any(tuple(values) != keys for values in derivatives[1:]):
            raise ValueError("all extrapolation steps must contain identical displacement keys")

        squared = np.square(self.grid / self.displacement)
        design = np.vander(squared, N=self.degree + 1, increasing=True)
        center = self.side_steps
        result: dict[DisplacementKey, np.ndarray] = {}
        corrections: list[np.ndarray] = []
        references: list[np.ndarray] = []
        residual_maximum = 0.0
        for key in keys:
            samples = np.asarray([values[key] for values in derivatives], dtype=float)
            flat = samples.reshape(len(self.grid), -1)
            coefficients = np.linalg.lstsq(design, flat, rcond=None)[0]
            extrapolated = coefficients[0].reshape(samples.shape[1:])
            fitted = (design @ coefficients).reshape(samples.shape)
            correction = extrapolated - samples[center]
            result[key] = extrapolated
            corrections.append(correction.ravel())
            references.append(samples[center].ravel())
            residual_maximum = max(
                residual_maximum,
                float(np.max(np.abs(fitted - samples))),
            )

        correction_vector = np.concatenate(corrections) if corrections else np.empty(0)
        reference_vector = np.concatenate(references) if references else np.empty(0)
        maximum = float(np.max(np.abs(correction_vector))) if len(correction_vector) else 0.0
        denominator = max(float(np.linalg.norm(reference_vector)), np.finfo(float).tiny)
        relative = float(np.linalg.norm(correction_vector) / denominator)
        return result, ExtrapolationMetrics(maximum, relative, residual_maximum)


__all__ = ["ExtrapolationBackend", "ExtrapolationMetrics"]
