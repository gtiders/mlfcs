"""Exact reciprocal characters of finite supercell translation groups.

The module owns three things:

* the full q grid of a supercell, ``ReciprocalQuotientGrid``, whose integer labels
  satisfy ``q = label / denominator`` and ``q S^T in Z^3``;
* the subgroup of the primitive crystal symmetry that keeps that grid,
  ``ReciprocalGridSymmetry``, together with its exact action on the grid labels;
* the star decomposition of the grid into irreducible representatives,
  ``IrreducibleReciprocalGrid``.

Every step is an integer statement.  Labels are residues modulo the denominator, the
action of a rotation is the exact integer map :func:`rotate_labels`, and no code path
compares floating point q coordinates with a tolerance, so a point is never discarded
or accepted because a rounding mismatch happened to look small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.structure.integer_lattice import (
    IntegerLatticeQuotient,
    adjugate_3x3,
    determinant_3x3,
    exact_modular_product,
    normalize_supercell_matrix,
    supercell_lattice_compatible_indices,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

#: Largest denominator whose mixed-radix label key ``l0 D^2 + l1 D + l2`` fits in int64.
_MAX_KEY_DENOMINATOR = 2_097_151


@dataclass(frozen=True, slots=True)
class ReciprocalQuotientGrid:
    """Integer labels and fractional coordinates for reciprocal characters."""

    labels: np.ndarray
    denominator: int
    points: np.ndarray

    def negative_label(self, label: np.ndarray) -> tuple[int, int, int]:
        values = np.mod(-np.asarray(label, dtype=np.int64), self.denominator)
        return tuple(int(value) for value in values)


def reciprocal_quotient_grid(integer_matrix: object) -> ReciprocalQuotientGrid:
    """Return the reciprocal quotient in deterministic exact-label order."""
    matrix = normalize_supercell_matrix(integer_matrix)
    determinant = abs(determinant_3x3(matrix))
    representatives = IntegerLatticeQuotient(matrix.T).representatives
    labels = exact_modular_product(representatives, adjugate_3x3(matrix).T, determinant).astype(
        np.int64
    )
    points = labels.astype(float) / determinant
    if len(np.unique(labels, axis=0)) != determinant:
        raise RuntimeError("reciprocal quotient contains duplicate q points")
    # Label identity is a congruence, never a floating comparison: q = l / D is a reciprocal
    # lattice point of the supercell exactly when l S^T = 0 (mod D).
    if np.any(exact_modular_product(labels, matrix.T, determinant) != 0):
        offending = int(np.flatnonzero(np.any(exact_modular_product(labels, matrix.T, determinant) != 0, axis=1))[0])
        raise RuntimeError(
            f"reciprocal quotient contains an incompatible q point: label "
            f"{labels[offending].tolist()} of denominator {determinant} is not annihilated by "
            f"the supercell matrix {matrix.tolist()}"
        )
    return ReciprocalQuotientGrid(labels, determinant, points)


def quotient_qpoints(integer_matrix: object) -> np.ndarray:
    """Return floating q coordinates in exact quotient-label order."""
    return reciprocal_quotient_grid(integer_matrix).points


def _positive_denominator(denominator: object) -> int:
    """Return ``denominator`` as an ``int`` after a positivity check."""
    value = int(denominator)
    if value < 1 or value != denominator:
        raise ValueError(f"denominator must be a positive integer, not {denominator!r}")
    return value


def _integer_inverse(rotation: object) -> np.ndarray:
    """Return the exact integer inverse of one unimodular 3x3 rotation."""
    values = np.asarray(rotation)
    if values.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("rotation must be an integer array; the label action never rounds")
    determinant = determinant_3x3(values)
    if abs(determinant) != 1:
        raise ValueError(f"rotation must be unimodular, not share determinant {determinant}")
    inverse = adjugate_3x3(values)
    if determinant < 0:
        inverse = -inverse
    if not np.array_equal(inverse @ values, np.eye(3, dtype=np.int64)):
        raise RuntimeError(f"integer inverse of {values.tolist()} is inconsistent")
    return inverse


def rotate_labels(
    labels: object, rotation: object, denominator: int, *, reduce: bool = True
) -> np.ndarray:
    r"""Map integer q labels by one integer rotation, exactly.

    Fractional coordinates are row vectors and a space group operation is
    ``x' = x R^T + t``, so the Bloch phase ``exp(2 pi i q . x)`` is preserved only when

    .. math::

        q' . x' = q' R x^T + q' . t = q . x + \text{const}

    for every ``x``; hence ``q' R = q`` and ``q' = q R^{-1}``.  With integer labels
    ``l = D q`` the same statement reads ``l' = l R^{-1} mod D``, evaluated here as
    ``labels @ R^{-1}`` with the exact integer inverse of ``R`` and then reduced modulo
    the denominator.  The transposed reading ``labels @ R^{-T}`` belongs to the other
    row-vector convention and is wrong for this repository.

    Parameters
    ----------
    labels : (..., 3) integer array
        Labels of ``q = label / denominator``.  Floating point input is rejected: the
        action is exact and never rounds a coordinate onto a grid point.
    rotation : (3, 3) integer array
        A unimodular spglib rotation ``R`` in the primitive fractional basis.
    denominator : int
        Positive denominator ``D`` of the grid.
    reduce : bool
        Reduce the result modulo ``D``, which is what a *label set* needs: a star, an orbit
        or a permutation of grid points.  A covariance relation must pass ``reduce=False``,
        because reducing by ``D`` is a primitive reciprocal lattice translation and the
        positional gauge of the dynamical matrix turns that into a site-diagonal factor:
        ``D(q + G) = \\Gamma_G D(q) \\Gamma_G^{-1}`` with
        ``\\Gamma_G = diag(exp(2 pi i G . \\tau_a))``.  Reducing first would therefore
        attribute a pure gauge change to the representation and report a large residual.

    Returns
    -------
    np.ndarray
        ``int64`` labels ``l R^{-1}`` (optionally ``mod D``) with the shape of ``labels``.
    """
    values = np.asarray(labels)
    if values.shape[-1:] != (3,):
        raise ValueError("labels must be an array ending in shape (3,)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be an integer array; the label action never rounds")
    moved = values @ _integer_inverse(rotation)
    if not reduce:
        return moved
    return np.mod(moved, _positive_denominator(denominator))


def rotate_label(
    label: object, rotation: object, denominator: int, *, reduce: bool = True
) -> tuple[int, int, int]:
    """Scalar form of :func:`rotate_labels` for one ``(3,)`` label."""
    values = np.asarray(label)
    if values.shape != (3,):
        raise ValueError("label must have shape (3,)")
    moved = rotate_labels(values, rotation, denominator, reduce=reduce)
    return (int(moved[0]), int(moved[1]), int(moved[2]))


def _label_keys(labels: np.ndarray, denominator: int) -> np.ndarray:
    """Return the unique mixed-radix int64 key of each reduced label."""
    weights = np.asarray((denominator * denominator, denominator, 1), dtype=np.int64)
    return labels @ weights


def _label_lookup(labels: np.ndarray, denominator: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sorted_keys, order)`` for exact lookups on one label set."""
    if denominator > _MAX_KEY_DENOMINATOR:
        raise RuntimeError(f"denominator {denominator} is too large for exact int64 label keys")
    keys = _label_keys(labels, denominator)
    order = np.argsort(keys, kind="stable")
    return keys[order], order


def _lookup_indices(
    sorted_keys: np.ndarray, order: np.ndarray, denominator: int, queries: np.ndarray
) -> np.ndarray:
    """Return full-grid indices of ``queries``, or ``-1`` where the label is absent."""
    keys = _label_keys(np.asarray(queries), denominator)
    positions = np.searchsorted(sorted_keys, keys)
    clipped = np.minimum(positions, len(sorted_keys) - 1)
    found = (positions < len(sorted_keys)) & (sorted_keys[clipped] == keys)
    return np.where(found, order[clipped], np.int64(-1))


@dataclass(frozen=True, slots=True)
class ReciprocalGridSymmetry:
    """The exact subgroup of the primitive symmetry that keeps one reciprocal grid.

    ``operation_indices`` index the primitive symmetry operations and ascend, so the
    subgroup does not depend on the order spglib returned the operations in.
    ``rotations`` are those integer spglib rotations and ``label_permutations[k, i]`` is
    the full-grid index that operation ``operation_indices[k]`` maps the label of grid
    point ``i`` onto.  Each row is a permutation of ``range(N_q)`` because a
    grid-preserving rotation acts bijectively on the labels, exactly and without any
    coordinate tolerance.
    """

    operation_indices: np.ndarray
    rotations: np.ndarray
    label_permutations: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.operation_indices, dtype=np.int64)
        rotations = np.asarray(self.rotations)
        permutations = np.asarray(self.label_permutations, dtype=np.int64)
        if indices.ndim != 1:
            raise RuntimeError("operation_indices must be a one-dimensional index array")
        if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise RuntimeError(f"operation_indices must ascend without repeats: {indices.tolist()}")
        if rotations.shape != (len(indices), 3, 3):
            raise RuntimeError(
                f"rotations {rotations.shape} do not match {len(indices)} operation indices"
            )
        if not np.issubdtype(rotations.dtype, np.integer):
            raise RuntimeError("rotations must be integer matrices")
        if permutations.ndim != 2 or permutations.shape[0] != len(indices):
            raise RuntimeError(
                f"label_permutations {permutations.shape} do not match {len(indices)} operations"
            )
        expected = np.arange(permutations.shape[1], dtype=np.int64)
        for row, permutation in enumerate(permutations):
            if not np.array_equal(np.sort(permutation), expected):
                raise RuntimeError(
                    f"operation {int(indices[row])} with rotation {rotations[row].tolist()} "
                    f"does not permute the {len(expected)} grid labels: {permutation.tolist()}"
                )
        object.__setattr__(self, "operation_indices", indices)
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "label_permutations", permutations)


def reciprocal_grid_symmetry(
    supercell_matrix: object, symmetry: PrimitiveSymmetryOperations
) -> ReciprocalGridSymmetry:
    """Return the exact grid-preserving subgroup and its action on the full grid.

    Only operations whose integer rotation keeps the supercell lattice ``Z^3 S`` are
    kept, and every kept operation is then applied to every full-grid label: the image
    must be a label of the grid, so an incompatible operation raises instead of losing a
    point to a tolerance.  ``symmetry`` only has to expose its ``rotations``.
    """
    matrix = normalize_supercell_matrix(supercell_matrix)
    return _grid_symmetry(reciprocal_quotient_grid(matrix), matrix, symmetry)


def _grid_symmetry(
    grid: ReciprocalQuotientGrid, matrix: np.ndarray, symmetry: PrimitiveSymmetryOperations
) -> ReciprocalGridSymmetry:
    """Return the grid-preserving subgroup of ``symmetry`` for an existing grid."""
    rotations = np.asarray(symmetry.rotations)
    operation_indices = supercell_lattice_compatible_indices(matrix, rotations)
    selected = rotations[operation_indices]
    sorted_keys, order = _label_lookup(grid.labels, grid.denominator)
    permutations = np.empty((len(operation_indices), len(grid.labels)), dtype=np.int64)
    for row, rotation in enumerate(selected):
        moved = rotate_labels(grid.labels, rotation, grid.denominator)
        targets = _lookup_indices(sorted_keys, order, grid.denominator, moved)
        missing = np.flatnonzero(targets < 0)
        if len(missing):
            point = int(missing[0])
            raise RuntimeError(
                f"operation {int(operation_indices[row])} keeps the supercell lattice "
                f"S={matrix.tolist()} but maps label {grid.labels[point].tolist()} to "
                f"{moved[point].tolist()}, which is not a label of the denominator "
                f"{grid.denominator} grid"
            )
        permutations[row] = targets
    return ReciprocalGridSymmetry(operation_indices, selected, permutations)


@dataclass(frozen=True, slots=True)
class ReciprocalStar:
    """One orbit of the full q grid under the grid-preserving subgroup and time reversal.

    ``representative`` is the smallest full-grid index of the orbit, ``members`` are the
    full-grid indices ascending, ``operations`` holds for each member the primitive
    operation index that maps the representative onto it, and ``antiunitary`` says
    whether that map applies time reversal as well.

    The recorded operation is the smallest one that reaches the member, and a member a
    unitary operation reaches keeps its unitary map even when time reversal would reach
    it too, so ``antiunitary`` is ``True`` only where time reversal is genuinely needed.
    """

    representative: int
    members: np.ndarray
    operations: np.ndarray
    antiunitary: np.ndarray

    def __post_init__(self) -> None:
        representative = int(self.representative)
        members = np.asarray(self.members, dtype=np.int64)
        operations = np.asarray(self.operations, dtype=np.int64)
        antiunitary = np.asarray(self.antiunitary, dtype=bool)
        if members.ndim != 1 or members.size == 0:
            raise RuntimeError(f"star {representative} has no members")
        if np.any(np.diff(members) <= 0):
            raise RuntimeError(f"star {representative} members must ascend: {members.tolist()}")
        if int(members[0]) != representative:
            raise RuntimeError(
                f"star {representative} does not start at its smallest member {int(members[0])}"
            )
        if operations.shape != members.shape:
            raise RuntimeError(
                f"star {representative} has {len(operations)} operations for {len(members)} members"
            )
        if np.any(operations < 0):
            raise RuntimeError(f"star {representative} has a negative operation index")
        if antiunitary.shape != members.shape:
            raise RuntimeError(
                f"star {representative} has {len(antiunitary)} time reversal flags for "
                f"{len(members)} members"
            )
        object.__setattr__(self, "representative", representative)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "antiunitary", antiunitary)


@dataclass(frozen=True, slots=True)
class IrreducibleReciprocalGrid:
    """Exact irreducible wedge of one reciprocal grid.

    ``full_to_irreducible`` sends every full-grid index to the star that owns it,
    ``full_operations`` and ``full_antiunitary`` say how each full-grid point is reached
    from the representative of its star, ``weights`` are the star sizes and sum to the
    number of grid points ``N_q``, and ``little_groups[s]`` lists the grid-preserving
    operations that fix the representative of star ``s``.

    The little group is *operation-level*: it is indexed by primitive operations, not by
    the permutations they induce, so operations that coincide on every label -- the
    label action depends on the rotation only -- are all listed.  A later Fourier gauge
    needs the operations themselves, because their translation parts and phases differ
    even when their action on q does not.
    """

    full: ReciprocalQuotientGrid
    stars: tuple[ReciprocalStar, ...]
    representatives: np.ndarray
    full_to_irreducible: np.ndarray
    full_operations: np.ndarray
    full_antiunitary: np.ndarray
    weights: np.ndarray
    little_groups: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        _validate_irreducible_grid(self)

    @property
    def n_qpoints(self) -> int:
        """Number of q points of the full grid."""
        return len(self.full.labels)

    @property
    def n_irreducible(self) -> int:
        """Number of irreducible representative q points."""
        return len(self.representatives)

    @property
    def reduction_ratio(self) -> float:
        """How many full q points one representative stands for."""
        return self.n_qpoints / self.n_irreducible


def _validate_irreducible_grid(result: IrreducibleReciprocalGrid) -> None:
    """Raise ``RuntimeError`` unless the stars partition the grid with exact weights."""
    n_q = len(result.full.labels)
    denominator = result.full.denominator
    stars = result.stars
    star_count = len(stars)
    representatives = np.asarray(result.representatives, dtype=np.int64)
    weights = np.asarray(result.weights, dtype=np.int64)
    assignment = np.asarray(result.full_to_irreducible, dtype=np.int64)
    operations = np.asarray(result.full_operations, dtype=np.int64)
    antiunitary = np.asarray(result.full_antiunitary, dtype=bool)
    if representatives.shape != (star_count,):
        raise RuntimeError(
            f"representatives {representatives.tolist()} do not match {star_count} stars"
        )
    from_stars = np.asarray([star.representative for star in stars], dtype=np.int64)
    if not np.array_equal(representatives, from_stars) or np.any(np.diff(representatives) <= 0):
        raise RuntimeError(
            f"representatives {representatives.tolist()} must be the ascending star "
            f"representatives {from_stars.tolist()}"
        )
    sizes = np.asarray([len(star.members) for star in stars], dtype=np.int64)
    if weights.ndim != 1 or weights.shape != (star_count,):
        raise RuntimeError(f"star weights {weights.tolist()} do not match {star_count} stars")
    for name, values in (
        ("full_to_irreducible", assignment),
        ("full_operations", operations),
        ("full_antiunitary", antiunitary),
    ):
        if values.shape != (n_q,):
            raise RuntimeError(f"{name} {values.shape} does not cover the {n_q} grid points")
    members = np.concatenate([star.members for star in stars])
    if np.any(members < 0) or np.any(members >= n_q):
        raise RuntimeError(
            f"star members must index the {n_q} grid points of denominator "
            f"{denominator}, not {members.tolist()}"
        )
    if not np.array_equal(np.sort(members), np.arange(n_q, dtype=np.int64)):
        missing = np.setdiff1d(np.arange(n_q, dtype=np.int64), members)
        counts = np.unique(members, return_counts=True)[1]
        repeated = np.unique(members)[counts > 1]
        raise RuntimeError(
            f"stars of denominator {denominator} do not partition the {n_q} grid points: "
            f"missing {missing.tolist()}, repeated {repeated.tolist()}"
        )
    if not np.array_equal(weights, sizes):
        raise RuntimeError(
            f"star weights {weights.tolist()} are not the star sizes {sizes.tolist()}"
        )
    total = int(weights.sum())
    if total != n_q:
        raise RuntimeError(
            f"star weights {weights.tolist()} sum to {total}, not the {n_q} grid points "
            f"of denominator {denominator}"
        )
    for index, star in enumerate(stars):
        if not np.all(assignment[star.members] == index):
            raise RuntimeError(
                f"star {index} with representative {star.representative} does not own its "
                f"members {star.members.tolist()}: full_to_irreducible is "
                f"{assignment[star.members].tolist()}"
            )
        if not np.array_equal(operations[star.members], star.operations):
            raise RuntimeError(
                f"star {index} operations {star.operations.tolist()} disagree with "
                f"full_operations {operations[star.members].tolist()}"
            )
        if not np.array_equal(antiunitary[star.members], star.antiunitary):
            raise RuntimeError(
                f"star {index} time reversal flags {star.antiunitary.tolist()} disagree "
                f"with full_antiunitary {antiunitary[star.members].tolist()}"
            )
    if len(result.little_groups) != star_count:
        raise RuntimeError(
            f"{len(result.little_groups)} little groups do not match {star_count} stars"
        )
    for index, little_group in enumerate(result.little_groups):
        values = np.asarray(little_group)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError(
                f"little group of star {representatives[index]} is not an index array: "
                f"{values.tolist()}"
            )
        if values.size == 0:
            raise RuntimeError(
                f"little group of star {representatives[index]} is empty, but every "
                f"representative is fixed by at least the identity operation"
            )
        if np.any(values < 0) or np.any(np.diff(values) <= 0):
            raise RuntimeError(
                f"little group of star {representatives[index]} must ascend without "
                f"repeats: {values.tolist()}"
            )


def irreducible_reciprocal_grid(
    supercell_matrix: object,
    symmetry: PrimitiveSymmetryOperations,
    *,
    time_reversal: bool = True,
) -> IrreducibleReciprocalGrid:
    """Decompose the full q grid of one supercell into exact irreducible stars.

    The grid-preserving subgroup is exactly the set of primitive operations whose
    integer rotation keeps ``Z^3 S``, each star is one orbit of the full grid under that
    subgroup and, when ``time_reversal`` is true, under the antiunitary time reversal
    that maps the label ``l`` to the label of ``-q`` via
    :meth:`ReciprocalQuotientGrid.negative_label`.  Representatives are the smallest
    full-grid index of their orbit, members ascend, and each member records the smallest
    primitive operation that reaches it, so the result is independent of the order in
    which the symmetry operations were listed.

    Time reversal is an explicit switch so that future magnetic models can turn the
    antiunitary operation off; ``antiunitary`` records per member where it was used.
    """
    matrix = normalize_supercell_matrix(supercell_matrix)
    grid = reciprocal_quotient_grid(matrix)
    return _decompose(grid, _grid_symmetry(grid, matrix, symmetry), time_reversal=time_reversal)


def _decompose(
    grid: ReciprocalQuotientGrid, grid_symmetry: ReciprocalGridSymmetry, *, time_reversal: bool
) -> IrreducibleReciprocalGrid:
    """Return the star decomposition of ``grid`` under ``grid_symmetry``."""
    n_q = len(grid.labels)
    permutations = grid_symmetry.label_permutations
    operation_count = len(grid_symmetry.operation_indices)
    reversed_indices = None
    if time_reversal:
        sorted_keys, order = _label_lookup(grid.labels, grid.denominator)
        negated = np.mod(-grid.labels, grid.denominator)
        reversed_indices = _lookup_indices(sorted_keys, order, grid.denominator, negated)
        missing = np.flatnonzero(reversed_indices < 0)
        if len(missing):
            point = int(missing[0])
            raise RuntimeError(
                f"time reversal maps label {grid.labels[point].tolist()} to "
                f"{negated[point].tolist()}, which is not a label of the denominator "
                f"{grid.denominator} grid"
            )
    star_of = np.full(n_q, -1, dtype=np.int64)
    full_operations = np.zeros(n_q, dtype=np.int64)
    full_antiunitary = np.zeros(n_q, dtype=bool)
    stars: list[ReciprocalStar] = []
    for start in range(n_q):
        if star_of[start] >= 0:
            continue
        members = [start]
        seen = np.zeros(n_q, dtype=bool)
        seen[start] = True
        for member in members:
            for row in range(operation_count):
                target = int(permutations[row, member])
                if not seen[target]:
                    seen[target] = True
                    members.append(target)
            if reversed_indices is not None:
                target = int(reversed_indices[member])
                if not seen[target]:
                    seen[target] = True
                    members.append(target)
        members.sort()
        operations = np.empty(len(members), dtype=np.int64)
        antiunitary = np.zeros(len(members), dtype=bool)
        for position, member in enumerate(members):
            unitary = np.flatnonzero(permutations[:, start] == member)
            if len(unitary):
                operations[position] = grid_symmetry.operation_indices[unitary[0]]
                continue
            via_reversal = (
                np.flatnonzero(permutations[:, int(reversed_indices[start])] == member)
                if reversed_indices is not None
                else np.empty(0, dtype=np.int64)
            )
            if not len(via_reversal):
                raise RuntimeError(
                    f"no grid-preserving operation maps representative {start} "
                    f"(label {grid.labels[start].tolist()}) onto member {member} "
                    f"(label {grid.labels[member].tolist()}) of denominator "
                    f"{grid.denominator}"
                )
            operations[position] = grid_symmetry.operation_indices[via_reversal[0]]
            antiunitary[position] = True
        star_of[members] = len(stars)
        full_operations[members] = operations
        full_antiunitary[members] = antiunitary
        stars.append(
            ReciprocalStar(
                start,
                np.asarray(members, dtype=np.int64),
                operations,
                antiunitary,
            )
        )
    little_groups = tuple(
        grid_symmetry.operation_indices[permutations[:, star.representative] == star.representative]
        for star in stars
    )
    return IrreducibleReciprocalGrid(
        full=grid,
        stars=tuple(stars),
        representatives=np.asarray([star.representative for star in stars], dtype=np.int64),
        full_to_irreducible=star_of,
        full_operations=full_operations,
        full_antiunitary=full_antiunitary,
        weights=np.asarray([len(star.members) for star in stars], dtype=np.int64),
        little_groups=little_groups,
    )


__all__ = [
    "IrreducibleReciprocalGrid",
    "ReciprocalGridSymmetry",
    "ReciprocalQuotientGrid",
    "ReciprocalStar",
    "exact_modular_product",
    "irreducible_reciprocal_grid",
    "quotient_qpoints",
    "reciprocal_grid_symmetry",
    "reciprocal_quotient_grid",
    "rotate_label",
    "rotate_labels",
]
