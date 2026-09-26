"""Exact primitive-lattice site labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True, slots=True)
class LatticeSite:
    """One primitive motif site translated by an exact integer lattice vector."""

    site: int
    translation: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if self.site < 0:
            raise ValueError("primitive site must be non-negative")
        if len(self.translation) != 3 or any(
            not isinstance(value, (int, np.integer)) for value in self.translation
        ):
            raise TypeError("lattice translation must contain exactly three integers")
        object.__setattr__(self, "site", int(self.site))
        object.__setattr__(self, "translation", tuple(int(value) for value in self.translation))


__all__ = ["LatticeSite"]
