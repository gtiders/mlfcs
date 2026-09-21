"""Exact integer change of frame between a user cell and a canonical algebra cell.

The orbit algebra is only defined once a lattice basis is fixed: two user cells
describing the same crystal -- same lattice, same motif, but a different
unimodular basis choice -- must produce identical interaction tensors and
identical integer orbit labels.  :class:`LatticeFrame` fixes that choice once:
it reduces the user cell, selects a canonical representative among the
Minkowski-reduced bases of that lattice, and stores the resulting unimodular
integer map together with every exact conversion between the frames.

Conventions (row vectors, as used throughout :mod:`mlfcs.structure`): lattice
vectors are the rows of a ``(3, 3)`` cell, a fractional row vector ``f`` sits at
Cartesian ``f @ cell``, a translation ``t`` is an integer row vector with
Cartesian offset ``t @ cell``, spglib rotations act on fractional *column*
vectors (``x' = R @ x + s``), and ``source_to_algebra`` is the integer matrix
``U`` with ``algebra_cell == U @ source_cell``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce

from mlfcs.structure.integer_lattice import adjugate_3x3, determinant_3x3

_CANONICAL_DECIMALS = 8
"""Decimals used for the canonical comparison keys."""

_WRAP_EPSILON = 1e-12
"""Fractional coordinates this close to 1 wrap to 0, keeping keys representation independent."""

_NORM_TOLERANCE = 1e-9
"""Relative slack when comparing lattice vector lengths, so float noise cannot drop a basis."""

_REDUCTION_TOLERANCE = 1e-6
"""Relative slack for the Minkowski conditions, keeping degenerate equalities stable."""

_LABEL_TOLERANCE = 1e-8
"""Absolute slack when rounding a label translation back to integers, in fractional units."""


def _readonly(array: np.ndarray, dtype: object = None) -> np.ndarray:
    """Return an owning, contiguous, read-only copy of ``array``."""
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _exact_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the exact integer inverse of a unimodular matrix."""
    determinant = determinant_3x3(matrix)
    if abs(determinant) != 1:
        raise ValueError("matrix must be unimodular to invert exactly")
    return determinant * adjugate_3x3(matrix)


def _wrap(scaled: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into ``[0, 1)`` with input-independent boundaries."""
    wrapped = np.mod(scaled, 1.0)
    boundary = (wrapped == 0.0) | (wrapped >= 1.0 - _WRAP_EPSILON)
    return np.where(boundary, 0.0, wrapped)


def _rounded(position: np.ndarray) -> tuple[float, ...]:
    """Return the rounded value used for canonical ordering."""
    return tuple(round(float(value), _CANONICAL_DECIMALS) for value in position)


def _seed_basis(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a reduced seed basis ``reduced = op @ cell`` of the same lattice.

    Minkowski reduction describes the lattice, not the basis: a reduced basis
    realizes the successive minima, which is what makes the search bound in
    :func:`_short_vectors` a lattice invariant.
    """
    reduced, op = minkowski_reduce(cell, pbc=True)
    reduced = np.asarray(reduced, dtype=np.float64)
    integers = np.rint(np.asarray(op)).astype(np.int64)
    scale = max(1.0, float(np.max(np.abs(cell))))
    if float(np.max(np.abs(integers @ cell - reduced))) > 1e-10 * scale:
        raise RuntimeError("Minkowski reduction is not an exact integer change of basis")
    if abs(determinant_3x3(integers)) != 1:
        raise RuntimeError("Minkowski reduction is not unimodular")
    if not bool(_reduced_mask(reduced[None, :, :], tolerance=_REDUCTION_TOLERANCE)[0]):
        raise RuntimeError("Minkowski reduction did not return a reduced basis")
    return reduced, integers


def _gram_schmidt_lengths(basis: np.ndarray) -> np.ndarray:
    """Return the lengths ``|b_i^*|`` of the Gram-Schmidt vectors of a row basis."""
    directions: list[np.ndarray] = []
    lengths: list[float] = []
    for row in basis:
        vector = np.asarray(row, dtype=np.float64).copy()
        for direction in directions:
            vector -= (vector @ direction) * direction
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            raise ValueError("basis is singular")
        directions.append(vector / length)
        lengths.append(length)
    return np.asarray(lengths, dtype=np.float64)


def _reduced_mask(cells: np.ndarray, *, tolerance: float) -> np.ndarray:
    """Return which candidate cells are Minkowski-reduced up to ``tolerance``.

    ``cells`` holds Cartesian row bases, so the conditions are plain Euclidean
    norms.  They are evaluated in permutation- and sign-invariant form (each vector
    pair against its longer member, then all four mixed sums against the longest
    vector).  ``ase.geometry.is_minkowski_reduced`` compares at machine precision
    instead, so a basis sitting exactly on an equality -- which is what degenerate
    lattices produce -- can be accepted for one representation of a lattice and
    rejected for another, making the canonical basis depend on the caller's basis
    after all.
    """
    squared = np.einsum("hij,hij->hi", cells, cells)
    slack = (1.0 - tolerance) ** 2
    accepted = np.ones(cells.shape[0], dtype=bool)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        longest = np.maximum(squared[:, first], squared[:, second])
        for sign in (1.0, -1.0):
            combined = cells[:, first] + sign * cells[:, second]
            accepted &= np.einsum("hi,hi->h", combined, combined) >= longest * slack
    lengthiest = np.max(squared, axis=1)
    signs = np.asarray(((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)), dtype=np.float64)
    combined = np.einsum("ks,hsi->hki", signs, cells)
    lengths = np.einsum("hki,hki->hk", combined, combined)
    accepted &= np.all(lengths >= (lengthiest * slack)[:, None], axis=1)
    return accepted


def _short_vectors(basis: np.ndarray, bound: float) -> np.ndarray:
    """Return every nonzero lattice vector of squared length at most ``bound``.

    Vectors are returned as integer coefficient rows in ``basis``.  The search box
    is rigorous: a lattice vector ``n @ basis`` expands in the Gram-Schmidt basis
    as ``sum_i n_i b_i^*``, so its squared length is ``sum_i (n_i |b_i^*|)^2`` and
    each ``|n_i|`` is bounded by ``sqrt(bound) / |b_i^*|``.  The result is sorted
    by increasing length.
    """
    gram = basis @ basis.T
    limits = [int(np.sqrt(bound) / length) + 1 for length in _gram_schmidt_lengths(basis)]
    box = np.asarray(
        list(product(*(range(-limit, limit + 1) for limit in limits))), dtype=np.int64
    ).reshape(-1, 3)
    box = box[np.any(box != 0, axis=1)]
    squared = np.einsum("ij,jk,ik->i", box, gram, box)
    inside = box[squared <= bound]
    return inside[np.argsort(squared[squared <= bound], kind="stable")]


def _candidate_bases(reduced: np.ndarray) -> np.ndarray:
    """Return the integer coefficient matrices of every short reduced basis of the lattice.

    Minkowski reduction is not unique up to signed permutations when the shortest
    lattice vectors are degenerate (fcc, diamond and hexagonal lattices admit
    reduced bases that are not even isometric), so a sign/permutation orbit of one
    seed basis would make the canonical choice depend on the input basis.  Instead
    the whole lattice is searched: for a three-dimensional lattice every
    Minkowski-reduced basis consists of successive-minima vectors, and the seed
    realizes those minima, so enumerating the vectors up to ``|b_3|`` and keeping
    the reduced triples yields the complete, lattice-only candidate set.

    Every returned matrix ``N`` satisfies ``det N == sign(det reduced)``, hence
    ``N @ reduced`` is right-handed -- the handedness the algebra assumes.
    """
    norms_sq = np.sort(np.einsum("ij,ij->i", reduced, reduced))
    _, second_sq, third_sq = (float(value) for value in norms_sq)
    gram = reduced @ reduced.T
    tolerance = 1.0 + _NORM_TOLERANCE
    short = _short_vectors(reduced, third_sq * tolerance)
    short_sq = np.einsum("ij,jk,ik->i", short, gram, short)
    near = short[short_sq <= second_sq * tolerance]
    near_sq = short_sq[short_sq <= second_sq * tolerance]
    determinant = 1 if np.linalg.det(reduced) > 0.0 else -1
    crosses = np.cross(near[:, None, :], near[None, :, :]).reshape(-1, 3)
    hits = np.argwhere(crosses @ short.T == determinant)
    if hits.shape[0] == 0:
        raise RuntimeError("no short basis of the lattice was found")
    pairs, third = hits[:, 0], hits[:, 1]
    first, second = np.divmod(pairs, near.shape[0])
    # Minkowski-reduced bases are sorted by increasing vector length.
    ordered = (near_sq[first] <= near_sq[second] * tolerance) & (
        near_sq[second] <= short_sq[third] * tolerance
    )
    triples = np.stack([near[first[ordered]], near[second[ordered]], short[third[ordered]]], axis=1)
    return triples[_reduced_mask(triples @ reduced, tolerance=_REDUCTION_TOLERANCE)]


def _assert_no_collisions(
    numbers: np.ndarray, wrapped: np.ndarray, cell: np.ndarray, symprec: float
) -> None:
    """Reject a candidate that puts two atoms of one species closer than ``symprec``."""
    for number in np.unique(numbers):
        group = wrapped[numbers == number]
        if group.shape[0] < 2:
            continue
        offset = group[:, None, :] - group[None, :, :]
        offset -= np.rint(offset)
        distance = np.linalg.norm(offset @ cell, axis=-1)
        np.fill_diagonal(distance, np.inf)
        if float(distance.min()) < symprec:
            raise RuntimeError(
                f"two atoms of atomic number {int(number)} collide in the candidate cell"
            )


@dataclass(frozen=True, slots=True, eq=False)
class _Candidate:
    """One candidate algebra basis.

    Only candidates whose rounded cell is the lexicographic minimum ever reach
    :meth:`canonical_motif`: the cell part of the canonical key dominates, so the
    motif is mapped for the tied candidates alone.
    """

    cell: np.ndarray
    transform: np.ndarray
    inverse: np.ndarray

    @classmethod
    def build(cls, cell: np.ndarray, transform: np.ndarray) -> _Candidate:
        """Return the candidate with ``cell == transform @ source_cell``."""
        return cls(cell=cell, transform=transform, inverse=_exact_inverse(transform))

    def canonical_motif(
        self, numbers: np.ndarray, scaled: np.ndarray, symprec: float
    ) -> tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map the motif into this basis.

        Returns ``(motif_key, positions, source_positions, numbers, order)``: the
        algebra fractional coordinates, the wrapped source fractional coordinates of
        the same atoms, and the source index of each canonical atom.  Every tied
        candidate shares the cell part of the key, so ``motif_key`` decides.
        """
        wrapped = _wrap(scaled @ self.inverse)
        if wrapped.shape != scaled.shape:
            raise RuntimeError("candidate frame lost atoms while changing basis")
        _assert_no_collisions(numbers, wrapped, self.cell, symprec)
        decorated = sorted(
            (int(numbers[index]), _rounded(wrapped[index]), index)
            for index in range(wrapped.shape[0])
        )
        order = np.asarray([entry[2] for entry in decorated], dtype=np.int64)
        ordered_numbers = numbers[order]
        ordered_positions = wrapped[order]
        key = tuple(
            zip((int(value) for value in ordered_numbers), map(_rounded, ordered_positions))
        )
        return key, ordered_positions, _wrap(scaled[order]), ordered_numbers, order


@dataclass(frozen=True, slots=True)
class LatticeFrame:
    """Unimodular integer map from a user (source) cell to a canonical algebra cell."""

    source_cell: np.ndarray
    """``(3, 3)`` float64 cell exactly as the user supplied it, rows = lattice vectors."""
    algebra_cell: np.ndarray
    """``(3, 3)`` float64 canonical reduced cell of the same lattice, det > 0."""
    source_to_algebra: np.ndarray
    """``(3, 3)`` int64 ``U`` with ``algebra_cell == U @ source_cell`` and ``abs(det U) == 1``."""
    algebra_to_source: np.ndarray
    """``(3, 3)`` int64 exact inverse of ``U``."""
    positions: np.ndarray
    """``(n, 3)`` float64 fractional coordinates in the algebra cell, canonical atom order."""
    source_positions: np.ndarray
    """``(n, 3)`` float64 wrapped source fractional coordinates, same atoms, same order.

    ``positions[i]`` and ``source_positions[i]`` are the same physical atom, whose
    index in the source ``Atoms`` is ``atom_map[i]``.
    """
    numbers: np.ndarray
    """``(n,)`` int32 atomic numbers in the same canonical order."""
    atom_map: np.ndarray
    """``(n,)`` int32 algebra atom index -> index of the same atom in the source ``Atoms``."""

    @classmethod
    def from_atoms(cls, primitive: Atoms, *, symprec: float = 1e-5) -> LatticeFrame:
        """Build the canonical frame of a fully periodic ``primitive`` cell.

        The candidate bases are all short Minkowski-reduced bases of the lattice,
        so the frame depends only on the crystal, never on the unimodular basis the
        caller used: a cell expressed as ``U0 @ primitive.cell`` yields the same
        frame and the same composed integer map.  ``symprec`` is the minimum
        Cartesian distance at which two atoms of one species stay distinguishable.
        """
        if not primitive.pbc.all():
            raise ValueError("LatticeFrame requires a fully periodic primitive cell")
        cell = np.asarray(primitive.cell, dtype=np.float64)
        if cell.shape != (3, 3):
            raise ValueError("LatticeFrame requires a 3x3 cell")
        scale = float(np.max(np.abs(cell)))
        if scale == 0.0 or abs(np.linalg.det(cell)) <= 1e-12 * scale**3:
            raise ValueError("LatticeFrame requires a non-singular cell")
        numbers = np.asarray(primitive.numbers, dtype=np.int32)
        scaled = np.asarray(primitive.get_scaled_positions(wrap=False), dtype=np.float64)
        reduced, op = _seed_basis(cell)
        coefficients = _candidate_bases(reduced)
        cells = coefficients @ reduced
        # The key is ``(rounded cell, motif)`` compared as plain tuples; comparing
        # the nine rounded cell entries lexicographically is what selects the
        # candidate, and only the tied ones need their motif mapped.
        keys = np.round(cells, _CANONICAL_DECIMALS).reshape(cells.shape[0], 9)
        ranking = np.lexsort(keys[:, ::-1].T)
        tied = np.flatnonzero(np.all(keys == keys[ranking[0]], axis=1))
        motifs = [
            _Candidate.build(cells[index], coefficients[index] @ op).canonical_motif(
                numbers, scaled, symprec
            )
            for index in tied
        ]
        best_motif, best_index = min(zip(motifs, tied), key=lambda pair: pair[0][0])
        _, positions, source_positions, canonical_numbers, order = best_motif
        transform = coefficients[best_index] @ op
        return cls(
            source_cell=_readonly(cell),
            algebra_cell=_readonly(cells[best_index]),
            source_to_algebra=_readonly(transform),
            algebra_to_source=_readonly(_exact_inverse(transform)),
            positions=_readonly(positions),
            source_positions=_readonly(source_positions),
            numbers=_readonly(canonical_numbers),
            atom_map=_readonly(order, dtype=np.int32),
        )

    def source_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map ``(site, translation)`` label rows from the algebra frame to the source frame.

        ``labels`` is an ``(..., 4)`` integer array of ``(algebra site, t_x, t_y, t_z)``
        and describes the atom anchored at ``positions[site] + t``.  The result has the
        same shape, with ``(source site, t_x, t_y, t_z)`` describing the same physical
        atom in the source cell: ``source_positions[source site] + t`` is that atom's
        source fractional coordinate, and both descriptions share the same Cartesian
        position.  Translations are exact integers and are never wrapped or reduced; a
        non-integral residual means the two frames do not describe the same lattice.
        """
        values = _label_array(labels)
        site = values[..., 0]
        if np.any(site < 0) or np.any(site >= self.numbers.shape[0]):
            raise ValueError("label site is outside the canonical atom range")
        anchor = self.positions[site] + values[..., 1:].astype(np.float64)
        # ``source_positions`` is already in the canonical atom order, so the source
        # coordinate of canonical atom ``site`` is ``source_positions[site]``; looking it
        # up through ``atom_map`` would apply the canonical permutation twice.
        residual = anchor @ self.source_to_algebra - self.source_positions[site]
        rounded = np.rint(residual)
        deviation = np.abs(residual - rounded)
        if np.any(deviation > _LABEL_TOLERANCE):
            offending = np.argwhere(np.any(deviation > _LABEL_TOLERANCE, axis=-1))
            row = tuple(int(value) for value in values[tuple(offending[0])])
            raise RuntimeError(f"label {row} is not reachable in the source frame")
        result = np.empty(values.shape, dtype=np.int64)
        result[..., 0] = self.atom_map[site]
        result[..., 1:] = rounded.astype(np.int64)
        return result

    def algebra_atoms(self) -> Atoms:
        """Return the source motif expressed in the canonical algebra cell.

        ``from_atoms`` only accepts fully periodic cells, so the result is fully
        periodic as well.
        """
        return Atoms(
            numbers=self.numbers,
            scaled_positions=self.positions,
            cell=self.algebra_cell,
            pbc=True,
        )

    def tensor_frame(self, order: int) -> np.ndarray:
        """Return the order-``order`` lattice-to-Cartesian map.

        The map is ``kron(algebra_cell.T, order times)``.
        """
        if order < 0:
            raise ValueError("tensor order must be non-negative")
        result = np.ones((1, 1), dtype=np.float64)
        for _ in range(order):
            result = np.kron(result, self.algebra_cell.T)
        return result

    def source_to_algebra_positions(self, scaled: np.ndarray) -> np.ndarray:
        """Map source fractional coordinates to algebra fractional coordinates."""
        return _fractional_array(scaled, "scaled") @ self.algebra_to_source

    def algebra_to_source_positions(self, scaled: np.ndarray) -> np.ndarray:
        """Map algebra fractional coordinates to source fractional coordinates."""
        return _fractional_array(scaled, "scaled") @ self.source_to_algebra

    def source_to_algebra_translations(self, translations: np.ndarray) -> np.ndarray:
        """Map integer lattice translations to the algebra basis, exactly and without reduction."""
        return _integer_array(translations, "translations") @ self.algebra_to_source

    def algebra_to_source_translations(self, translations: np.ndarray) -> np.ndarray:
        """Map integer algebra translations to the source basis, exactly and without reduction."""
        return _integer_array(translations, "translations") @ self.source_to_algebra

    def source_to_algebra_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Map a spglib rotation to the algebra basis.

        spglib rotations act on fractional column vectors, so the coordinate map
        conjugates as ``algebra_to_source.T @ R @ source_to_algebra.T``.
        """
        return _conjugate_rotation(self.algebra_to_source.T, rotation, self.source_to_algebra.T)

    def algebra_to_source_rotation(self, rotation: np.ndarray) -> np.ndarray:
        """Map a spglib rotation in the algebra basis back to the source basis."""
        return _conjugate_rotation(self.source_to_algebra.T, rotation, self.algebra_to_source.T)


def _label_array(array: np.ndarray) -> np.ndarray:
    """Return an integer ``(..., 4)`` label array of ``(site, translation)`` rows."""
    values = np.asarray(array)
    if values.shape[-1:] != (4,):
        raise ValueError("labels must have a trailing axis of length 4")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("labels must contain integers")
    return rounded.astype(np.int64)


def _fractional_array(array: np.ndarray, name: str) -> np.ndarray:
    """Return ``array`` as float64 with a validated trailing axis of length three."""
    values = np.asarray(array, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a trailing axis of length 3")
    return values


def _integer_array(array: np.ndarray, name: str) -> np.ndarray:
    """Return ``array`` as int64 with a validated trailing axis of length three."""
    values = np.asarray(array)
    if values.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a trailing axis of length 3")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{name} must contain integers")
    return rounded.astype(np.int64)


def _rotation_array(rotation: np.ndarray) -> np.ndarray:
    """Return a validated integral ``(3, 3)`` spglib rotation."""
    values = np.asarray(rotation)
    if values.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError("rotation must contain integers")
    return rounded.astype(np.int64)


def _conjugate_rotation(left: np.ndarray, rotation: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Conjugate an integral rotation and round the result back to exact integers."""
    values = _rotation_array(rotation).astype(np.float64)
    products = np.asarray(left, dtype=np.float64) @ values @ np.asarray(right, dtype=np.float64)
    rounded = np.rint(products)
    scale = max(1.0, float(np.max(np.abs(products))))
    if float(np.max(np.abs(products - rounded))) > 1e-8 * scale:
        raise RuntimeError("mapped rotation is not integral; the frame is not unimodular")
    return rounded.astype(np.int64)


def reduction_is_unimodular(frame: LatticeFrame, *, tolerance: float = 1e-10) -> bool:
    """Return whether the stored change of basis is exactly unimodular.

    A check for a hand-built frame: ``abs(det U) == 1`` and
    ``U @ U^{-1} == I`` in exact integer arithmetic.
    """
    forward = np.asarray(frame.source_to_algebra)
    backward = np.asarray(frame.algebra_to_source)
    if forward.shape != (3, 3) or backward.shape != (3, 3):
        return False
    rounded_forward = np.rint(forward)
    rounded_backward = np.rint(backward)
    if not np.array_equal(forward, rounded_forward):
        return False
    if not np.array_equal(backward, rounded_backward):
        return False
    integers_forward = rounded_forward.astype(np.int64)
    integers_backward = rounded_backward.astype(np.int64)
    if abs(abs(determinant_3x3(integers_forward)) - 1) > tolerance:
        return False
    return np.array_equal(integers_forward @ integers_backward, np.eye(3, dtype=np.int64))


__all__ = ["LatticeFrame", "reduction_is_unimodular"]
