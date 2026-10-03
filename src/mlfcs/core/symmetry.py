"""Primitive-cell symmetry without supercell realization state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from mlfcs._arrays import integer_array, require_bound
from mlfcs.core.structure import LatticeSite


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int64(values: object, *, name: str) -> np.ndarray:
    try:
        return integer_array(values, name=name)
    except ValueError as error:
        raise TypeError(f"{name} must contain declared integers") from error


def prove_site_actions(translations, rotations, shifts, *, reanchor=False):
    maxima = [max((abs(int(v)) for v in translations[:, k]), default=0) for k in range(3)]
    bound = 0
    for rotation, operation_shifts in zip(rotations, shifts, strict=True):
        for j in range(3):
            shift = max(
                (abs(int(v)) for v in np.asarray(operation_shifts).reshape(-1, 3)[:, j]), default=0
            )
            bound = max(bound, shift + sum(abs(int(rotation[j, k])) * maxima[k] for k in range(3)))
    return require_bound("primitive site action", bound * (2 if reanchor else 1))


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
        rotations = _readonly_int64(self.rotations, name="primitive rotations")
        translations = _readonly(self.translations, dtype=np.float64)
        cartesian = _readonly(self.cartesian_rotations, dtype=np.float64)
        permutations = _readonly_int64(self.site_permutations, name="site permutations")
        shifts = _readonly_int64(self.site_shifts, name="site shifts")
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
        expected = np.arange(permutations.shape[1], dtype=np.int64)
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
        translation = integer_array(label.translation, name="primitive translation")
        prove_site_actions(
            translation.reshape(1, 3),
            self.rotations[operation : operation + 1],
            self.site_shifts[operation : operation + 1, site : site + 1],
        )
        transformed = translation @ self.rotations[operation].T + self.site_shifts[operation, site]
        return LatticeSite(
            int(self.site_permutations[operation, site]),
            tuple(int(value) for value in transformed),
        )

    def __reduce__(self):
        return type(self), (
            self.rotations,
            self.translations,
            self.cartesian_rotations,
            self.site_permutations,
            self.site_shifts,
            self.symbol,
            self.symprec,
        )


def discover_symmetry(cell, scaled_positions, atomic_numbers, symprec) -> PrimitiveSymmetry:
    """Discover symmetry using the primitive cell's single length tolerance."""
    dataset = spglib.get_symmetry_dataset(
        (cell, scaled_positions, atomic_numbers),
        symprec=symprec,
    )
    if dataset is None:
        raise ValueError(
            "spglib could not determine primitive symmetry "
            f"for {len(atomic_numbers)} atoms at symprec {symprec:g} angstrom"
        )

    rotations = np.asarray(dataset.rotations, dtype=np.int64)
    translations = np.asarray(dataset.translations, dtype=np.float64)
    inverse_cell = np.linalg.inv(cell)
    cartesian = np.asarray([inverse_cell @ rotation.T @ cell for rotation in rotations])
    permutations = np.empty((len(rotations), len(atomic_numbers)), dtype=np.int64)
    shifts = np.empty((len(rotations), len(atomic_numbers), 3), dtype=np.int64)

    for operation, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        transformed = scaled_positions @ rotation.T + translation
        for site, position in enumerate(transformed):
            candidates = np.flatnonzero(atomic_numbers == atomic_numbers[site])
            differences = position - scaled_positions[candidates]
            rounded = np.rint(differences)
            if not np.all(np.isfinite(rounded)) or np.any(np.abs(rounded) >= float(1 << 63)):
                raise OverflowError("primitive site shifts cannot enter int64")
            lattice_shifts = rounded.astype(np.int64)
            residuals = np.linalg.norm(
                (differences - lattice_shifts) @ cell,
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

    return PrimitiveSymmetry(
        rotations=_readonly_int64(rotations, name="primitive rotations"),
        translations=_readonly(translations, dtype=np.float64),
        cartesian_rotations=_readonly(cartesian, dtype=np.float64),
        site_permutations=_readonly_int64(permutations, name="site permutations"),
        site_shifts=_readonly_int64(shifts, name="site shifts"),
        symbol=str(dataset.international).strip(),
        symprec=float(symprec),
    )
