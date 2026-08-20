"""Shared temperature scheduling for finite-temperature workflows."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np


ResultT = TypeVar("ResultT")


def normalize_temperature_schedule(temperature: float | Sequence[float]) -> tuple[float, ...]:
    """Validate, deduplicate, and sort one or more temperatures in kelvin."""
    if isinstance(temperature, (str, bytes)):
        raise TypeError("temperature must be a number or a sequence of numbers")
    if np.isscalar(temperature):
        values = (float(temperature),)
    else:
        values = tuple(float(value) for value in temperature)
    if not values:
        raise ValueError("temperature sequence must not be empty")
    if not np.isfinite(values).all() or any(value < 0.0 for value in values):
        raise ValueError("temperatures must be finite and non-negative")
    ordered = tuple(sorted(values))
    if any(left == right for left, right in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("temperature sequence must not contain duplicates")
    return ordered


@dataclass(frozen=True, slots=True)
class TemperatureSeriesResult(Generic[ResultT]):
    """Ordered results of an ascending multi-temperature calculation."""

    temperatures: tuple[float, ...]
    results: tuple[ResultT, ...]
    continuation: bool

    def __post_init__(self) -> None:
        if len(self.temperatures) != len(self.results):
            raise ValueError("temperatures and results must have equal length")

    def __iter__(self) -> Iterator[ResultT]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> ResultT:
        return self.results[index]

    def at_temperature(self, temperature: float) -> ResultT:
        """Return the result at one exact scheduled temperature."""
        target = float(temperature)
        try:
            return self.results[self.temperatures.index(target)]
        except ValueError as error:
            raise KeyError(f"temperature {target:g} K was not scheduled") from error

    @property
    def iterations(self) -> tuple[int, ...]:
        """Return the recorded iteration count at every temperature."""
        return tuple(
            int(getattr(result, "iterations", len(getattr(result, "history", ()))))
            for result in self.results
        )

    @property
    def converged(self) -> tuple[bool | None, ...]:
        """Return convergence flags when the underlying method defines them."""
        return tuple(
            None if not hasattr(result, "converged") else bool(result.converged)
            for result in self.results
        )


__all__ = ["TemperatureSeriesResult", "normalize_temperature_schedule"]
