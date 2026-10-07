"""Primitive-cell space-group actions on lattice sites and Cartesian tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from mlfcs.foundation.arrays import as_int64_array, require_bound
from mlfcs.geometry.periodic import PeriodicGeometry
from mlfcs.geometry.primitive import LatticeSite


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    """Return an owned readonly array with the requested dtype."""
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int64(values: object, *, name: str) -> np.ndarray:
    """Return an owned readonly int64 array, requiring integer-valued input."""
    try:
        return as_int64_array(values, name=name)
    except ValueError as error:
        raise TypeError(f"{name} must contain declared integers") from error


def validate_site_actions(translations, rotations, shifts, *, reanchor=False):
    """Validate integer bounds for affine actions on lattice translations.

    A primitive lattice translation ``n`` transforms as

        n' = n @ R.T + s,

    where ``R`` is an integer lattice rotation and ``s`` is the site-dependent
    lattice shift. If ``reanchor`` is true, the bound also covers subtraction
    of a transformed reference translation.

    Returns
    -------
    int
        Maximum admitted absolute integer magnitude.

    Raises
    ------
    OverflowError
        If any transformed or re-anchored translation may exceed the supported
        integer range.
    """
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
    """Space-group actions on a primitive motif and its periodic lattice sites.

    Spglib symmetry operations act on fractional row coordinates as

        x' = x @ R_g.T + t_g.

    Each motif site ``i`` is mapped to ``sigma_g(i)`` with an integer lattice
    shift ``s_g(i)`` satisfying

        x_i @ R_g.T + t_g
        = x_{sigma_g(i)} + s_g(i).

    Hence a periodic lattice site ``(i, n)`` transforms as

        (i, n) -> (sigma_g(i), n @ R_g.T + s_g(i)).

    The corresponding Cartesian row-vector action is

        Q_g = inv(cell) @ R_g.T @ cell.
    """

    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    symbol: str
    symprec: float

    def __post_init__(self) -> None:
        """Validate and normalize the primitive symmetry representation."""
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
        """Number of affine operations acting on the primitive motif."""
        return len(self.rotations)

    def transform_site(self, operation: int, label: LatticeSite) -> LatticeSite:
        """Apply one space-group operation to a periodic lattice site.

        For ``label = (i, n)``, operation ``g`` returns

            (sigma_g(i), n @ R_g.T + s_g(i)).

        Raises
        ------
        IndexError
            If the operation or primitive motif index is outside its valid range.
        OverflowError
            If the transformed lattice translation exceeds the supported integer
            range.
        """
        site = label.site
        if not 0 <= operation < self.size:
            raise IndexError("symmetry operation is outside the group")
        if not 0 <= site < self.site_permutations.shape[1]:
            raise IndexError("primitive site is outside the motif")
        translation = as_int64_array(label.translation, name="primitive translation")
        validate_site_actions(
            translation.reshape(1, 3),
            self.rotations[operation : operation + 1],
            self.site_shifts[operation : operation + 1, site : site + 1],
        )
        transformed = translation @ self.rotations[operation].T + self.site_shifts[operation, site]
        return LatticeSite(
            int(self.site_permutations[operation, site]),
            tuple(int(value) for value in transformed),
        )


def discover_symmetry(cell, scaled_positions, atomic_numbers, symprec) -> PrimitiveSymmetry:
    """Discover primitive-cell space-group actions and motif-site mappings.

    Spglib operations act on fractional row coordinates as

        x' = x @ R.T + t.

    For every operation and primitive motif site ``i``, this function finds
    the unique same-species motif site ``sigma(i)`` and integer shift ``s(i)``
    satisfying

        x_i @ R.T + t = x_{sigma(i)} + s(i)

    within the Cartesian tolerance ``symprec``.

    The corresponding Cartesian rotation is

        Q = inv(cell) @ R.T @ cell.

    Parameters
    ----------
    cell
        Primitive lattice vectors as rows, shape ``(3, 3)``, in angstrom.
    scaled_positions
        Fractional motif coordinates of shape ``(n_atoms, 3)``.
    atomic_numbers
        Atomic numbers in primitive motif order.
    symprec
        Positive Cartesian matching tolerance in angstrom.

    Returns
    -------
    PrimitiveSymmetry
        Primitive symmetry operations together with motif permutations,
        lattice shifts and Cartesian rotations.

    Raises
    ------
    ValueError
        If spglib cannot determine the symmetry or if a transformed motif
        site does not have a unique same-species periodic match.
    """
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
    # Convert the fractional row action before any Cartesian tensor contraction.
    cartesian = np.asarray([inverse_cell @ rotation.T @ cell for rotation in rotations])
    permutations = np.empty((len(rotations), len(atomic_numbers)), dtype=np.int64)
    shifts = np.empty((len(rotations), len(atomic_numbers), 3), dtype=np.int64)
    geometry = PeriodicGeometry(cell)

    for operation, (rotation, translation) in enumerate(zip(rotations, translations, strict=True)):
        transformed = scaled_positions @ rotation.T + translation
        for site, position in enumerate(transformed):
            candidates = np.flatnonzero(atomic_numbers == atomic_numbers[site])
            cartesian_differences = (position - scaled_positions[candidates]) @ cell
            matches, image_shifts = geometry.matching_images(
                cartesian_differences, tolerance=symprec
            )
            if len(matches) != 1:
                raise ValueError(
                    f"operation {operation} maps primitive site {site} to "
                    f"{len(matches)} sites within symprec {symprec:g} angstrom"
                )
            permutations[operation, site] = int(candidates[matches[0]])
            # matching_images returns a shift added to the difference; the affine
            # site address needs the opposite shift on the mapped motif position.
            shifts[operation, site] = -image_shifts[0]

    return PrimitiveSymmetry(
        rotations=_readonly_int64(rotations, name="primitive rotations"),
        translations=_readonly(translations, dtype=np.float64),
        cartesian_rotations=_readonly(cartesian, dtype=np.float64),
        site_permutations=_readonly_int64(permutations, name="site permutations"),
        site_shifts=_readonly_int64(shifts, name="site shifts"),
        symbol=str(dataset.international).strip(),
        symprec=float(symprec),
    )
