from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True, slots=True)
class CentralDifferenceStencil:
    """Recursive mixed central-difference stencil.

    An order-n force constant differentiates force n-1 times, hence it has
    ``2 ** (n - 1)`` sign configurations. This representation is valid for
    arbitrary n.
    """

    derivative_order: int
    step: float

    def __post_init__(self) -> None:
        if self.derivative_order < 1:
            raise ValueError("derivative_order must be positive")
        if self.step <= 0:
            raise ValueError("step must be positive")

    @classmethod
    def for_force_constant(cls, order: int, step: float) -> CentralDifferenceStencil:
        if order < 2:
            raise ValueError("force-constant order must be at least 2")
        return cls(order - 1, step)

    @property
    def signs(self) -> np.ndarray:
        return np.asarray(list(product((-1, 1), repeat=self.derivative_order)), dtype=np.int8)

    @property
    def weights(self) -> np.ndarray:
        return np.prod(self.signs, axis=1, dtype=np.int8).astype(float)

    @property
    def denominator(self) -> float:
        return (2.0 * self.step) ** self.derivative_order

    def contract(self, values: np.ndarray) -> np.ndarray:
        """Contract values whose leading axis follows :attr:`signs`."""
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} samples, got {values.shape[0]}")
        return np.tensordot(self.weights, values, axes=(0, 0)) / self.denominator
