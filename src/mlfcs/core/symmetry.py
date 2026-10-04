"""Primitive-cell symmetry without supercell realization state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib

from mlfcs._arrays import integer_array, require_bound
from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.structure import LatticeSite


def _readonly(values: object, *, dtype: object) -> np.ndarray:
    """Copy symmetry values into owned, readonly C-contiguous storage of the requested dtype."""
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_int64(values: object, *, name: str) -> np.ndarray:
    """Normalize exact symmetry integers, reporting undeclared integer input as TypeError."""
    try:
        return integer_array(values, name=name)
    except ValueError as error:
        raise TypeError(f"{name} must contain declared integers") from error


def prove_site_actions(translations, rotations, shifts, *, reanchor=False):
    """Validate affine site-action intermediates for the supplied translations.

    Translations have shape (n, 3), rotations (s, 3, 3), and shifts carry a final
    coordinate axis of length three. Absolute dot-product sums include site
    shifts; ``reanchor`` doubles the bound to cover subtracting an anchor.
    Python integer arithmetic computes the bound before require_bound admits
    unchecked int64 action loops. Return the admitted bound or raise OverflowError.
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
    """Immutable affine space-group actions on one primitive motif.

    Attributes
    ----------
    rotations : ndarray of int64, shape (s, 3, 3)
        Spglib fractional lattice actions: x' = x @ rotation.T + translation.
    translations : ndarray of float64, shape (s, 3)
        Fractional affine offsets.
    cartesian_rotations : ndarray of float64, shape (s, 3, 3)
        Cartesian row actions inv(cell) @ rotation.T @ cell. Tensor contractions
        use their transpose; these are not generally integer signed permutations.
    site_permutations : ndarray of int64, shape (s, n_atoms)
        Primitive motif indices reached by each operation.
    site_shifts : ndarray of int64, shape (s, n_atoms, 3)
        Integer shifts satisfying transformed motif position = mapped position
        + shift. Translated lattice addresses additionally receive t @ R.T.
    symbol : str
        Space-group symbol supplied by spglib.
    symprec : float
        Positive Cartesian matching tolerance in angstrom.

    Notes
    -----
    Construction validates buffer shapes and permutations and makes them readonly.
    No supercell atom permutation, cell matrix or mutable cache is stored here.
    """

    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    symbol: str
    symprec: float

    def __post_init__(self) -> None:
        """Normalize readonly buffers and validate operation shapes and motif permutations."""
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
        """Serialize declared symmetry buffers for reconstruction with validation."""
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
    """Discover affine primitive symmetry and uniquely match every mapped site.

    ``cell`` has lattice vectors as rows in angstrom; ``scaled_positions`` are
    fractional rows and ``atomic_numbers`` preserves motif order. ``symprec`` is
    a Cartesian tolerance in angstrom. Return readonly PrimitiveSymmetry data.
    Raise ValueError if spglib fails or any same-species image is not unique.

    Spglib uses x' = x @ R.T + t in fractional coordinates. The Cartesian row
    rotation is inv(cell) @ R.T @ cell; stored site shifts satisfy
    x_i @ R.T + t = x_permutation[i] + shift[i]. No supercell state is created.
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
