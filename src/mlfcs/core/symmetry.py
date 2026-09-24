"""Primitive-cell symmetry without supercell realization state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from mlfcs.core.lattice import LatticeSite
from mlfcs.core.primitive import PrimitiveCell


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int32(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain declared integers")
    limit = np.iinfo(np.int32)
    if any(int(value) < limit.min or int(value) > limit.max for value in array.flat):
        raise OverflowError(f"{name} does not fit in int32")
    return _readonly(array, dtype=np.int32)


@dataclass(frozen=True, slots=True)
class PrimitiveSymmetry:
    """The affine space group acting on one primitive motif.

    No atom permutation for a supercell is stored here.  ``site_permutations``
    acts only on the primitive motif, while ``site_shifts`` records the exact
    integer lattice translation needed after that primitive-site mapping.
    """

    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    symbol: str
    symprec: float

    def __post_init__(self) -> None:
        rotations = _readonly_int32(self.rotations, name="primitive rotations")
        translations = _readonly(self.translations, dtype=np.float64)
        cartesian = _readonly(self.cartesian_rotations, dtype=np.float64)
        permutations = _readonly_int32(self.site_permutations, name="site permutations")
        shifts = _readonly_int32(self.site_shifts, name="site shifts")
        count = len(rotations)
        if rotations.shape != (count, 3, 3):
            raise ValueError("primitive rotations must have shape (n, 3, 3)")
        if translations.shape != (count, 3) or cartesian.shape != (count, 3, 3):
            raise ValueError("primitive symmetry representations have inconsistent shapes")
        if permutations.ndim != 2 or shifts.shape != (*permutations.shape, 3):
            raise ValueError("primitive site actions have inconsistent shapes")
        if permutations.shape[0] != count:
            raise ValueError("every primitive operation must act on the motif")
        if not np.all(np.isfinite(translations)) or not np.all(np.isfinite(cartesian)):
            raise ValueError("primitive symmetry representations must be finite")
        expected = np.arange(permutations.shape[1], dtype=np.int32)
        if not np.array_equal(
            np.sort(permutations, axis=1), np.broadcast_to(expected, permutations.shape)
        ):
            raise ValueError("every primitive operation must permute every motif site exactly once")
        if not np.isfinite(self.symprec) or self.symprec <= 0.0:
            raise ValueError("symprec must be a positive finite length in angstrom")
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "cartesian_rotations", cartesian)
        object.__setattr__(self, "site_permutations", permutations)
        object.__setattr__(self, "site_shifts", shifts)
        object.__setattr__(self, "symbol", str(self.symbol))
        object.__setattr__(self, "symprec", float(self.symprec))

    @classmethod
    def from_primitive(cls, primitive: PrimitiveCell) -> PrimitiveSymmetry:
        """Discover symmetry using the primitive cell's single length tolerance."""
        symprec = primitive.symprec
        dataset = spglib.get_symmetry_dataset(
            (primitive.cell, primitive.scaled_positions, primitive.numbers),
            symprec=symprec,
        )
        if dataset is None:
            raise ValueError(
                "spglib could not determine primitive symmetry "
                f"for {primitive.size} atoms at symprec {symprec:g} angstrom"
            )

        rotations = np.asarray(dataset.rotations, dtype=np.int32)
        translations = np.asarray(dataset.translations, dtype=np.float64)
        inverse_cell = np.linalg.inv(primitive.cell)
        cartesian = np.asarray(
            [inverse_cell @ rotation.T @ primitive.cell for rotation in rotations]
        )
        permutations = np.empty((len(rotations), primitive.size), dtype=np.int32)
        shifts = np.empty((len(rotations), primitive.size, 3), dtype=np.int32)

        for operation, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        ):
            transformed = primitive.scaled_positions @ rotation.T + translation
            for site, position in enumerate(transformed):
                candidates = np.flatnonzero(primitive.numbers == primitive.numbers[site])
                differences = position - primitive.scaled_positions[candidates]
                lattice_shifts = np.rint(differences).astype(np.int32)
                residuals = np.linalg.norm(
                    (differences - lattice_shifts) @ primitive.cell,
                    axis=1,
                )
                matches = np.flatnonzero(residuals < symprec)
                if len(matches) != 1:
                    nearest = float(np.min(residuals)) if residuals.size else float("inf")
                    raise ValueError(
                        f"operation {operation} maps primitive site {site} to "
                        f"{len(matches)} sites within symprec {symprec:g} angstrom; "
                        f"nearest residual is {nearest:.10g} angstrom"
                    )
                match = int(matches[0])
                permutations[operation, site] = int(candidates[match])
                shifts[operation, site] = lattice_shifts[match]

        return cls(
            rotations=_readonly_int32(rotations, name="primitive rotations"),
            translations=_readonly(translations, dtype=np.float64),
            cartesian_rotations=_readonly(cartesian, dtype=np.float64),
            site_permutations=_readonly_int32(permutations, name="site permutations"),
            site_shifts=_readonly_int32(shifts, name="site shifts"),
            symbol=str(dataset.international).strip(),
            symprec=float(symprec),
        )

    @property
    def size(self) -> int:
        return len(self.rotations)

    def transform_site(self, operation: int, label: LatticeSite) -> LatticeSite:
        """Apply an operation to one exact primitive lattice site."""
        site = label.site
        if not 0 <= operation < self.size:
            raise IndexError("symmetry operation is outside the group")
        if not 0 <= site < self.site_permutations.shape[1]:
            raise IndexError("primitive site is outside the motif")
        translation = np.asarray(label.translation, dtype=np.int64)
        transformed = translation @ self.rotations[operation].T + self.site_shifts[operation, site]
        return LatticeSite(
            int(self.site_permutations[operation, site]),
            tuple(int(value) for value in transformed),
        )
