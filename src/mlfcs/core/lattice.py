"""Canonical primitive frames and exact lattice-site labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from mlfcs.core.algebra.integer import integer_inverse
from mlfcs.core.primitive import PrimitiveCell


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


@dataclass(frozen=True, slots=True)
class PrimitiveFrame:
    """An integer change of basis to spglib's non-idealized primitive standard."""

    source: PrimitiveCell
    canonical: PrimitiveCell
    canonical_to_source: np.ndarray
    source_to_canonical: np.ndarray

    @classmethod
    def from_primitive(cls, primitive: PrimitiveCell) -> PrimitiveFrame:
        standardized = spglib.standardize_cell(
            (primitive.cell, primitive.scaled_positions, primitive.numbers),
            to_primitive=True,
            no_idealize=True,
            symprec=primitive.symprec,
        )
        if standardized is None:
            raise ValueError("spglib could not construct a non-idealized primitive frame")
        cell, positions, numbers = standardized
        if len(numbers) != primitive.size:
            raise RuntimeError("primitive standardization changed the number of motif sites")

        coefficients = np.asarray(cell) @ np.linalg.inv(primitive.cell)
        canonical_to_source = np.rint(coefficients).astype(np.int64)
        cell_residual = np.linalg.norm(
            canonical_to_source @ primitive.cell - np.asarray(cell),
            axis=1,
        )
        if float(np.max(cell_residual)) >= primitive.symprec:
            raise RuntimeError(
                "non-idealized standardization is not an integer primitive-basis change: "
                f"maximum residual {float(np.max(cell_residual)):.10g} angstrom"
            )
        source_to_canonical = integer_inverse(canonical_to_source)
        canonical = PrimitiveCell(
            cell=np.asarray(cell),
            scaled_positions=np.asarray(positions),
            numbers=np.asarray(numbers),
            symprec=primitive.symprec,
        )
        canonical_to_source.setflags(write=False)
        source_to_canonical.setflags(write=False)
        return cls(primitive, canonical, canonical_to_source, source_to_canonical)

    def to_canonical_translations(self, translations: object) -> np.ndarray:
        """Map source-basis lattice translations to the canonical basis exactly."""
        values = _integer_vectors(translations)
        return values @ self.source_to_canonical

    def to_source_translations(self, translations: object) -> np.ndarray:
        """Map canonical-basis lattice translations to the source basis exactly."""
        values = _integer_vectors(translations)
        return values @ self.canonical_to_source


def _integer_vectors(values: object) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[-1:] != (3,) or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("lattice translations must be an integer array ending in shape (3,)")
    return array.astype(np.int64, copy=False)


__all__ = ["LatticeSite", "PrimitiveFrame"]
