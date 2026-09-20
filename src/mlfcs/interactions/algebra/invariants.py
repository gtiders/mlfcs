r"""Exact integer invariant-tensor kernels in the lattice (scaled) frame.

A spglib symmetry rotation is an integer matrix in lattice coordinates for every cell,
including the ones whose Cartesian rotations are irrational (an fcc primitive
$60^\circ$ cell, hexagonal and rhombohedral cells).  Every decision below therefore
happens on integers: the invariant dimension comes from a rank modulo a large prime, the
kernel basis is solved from an integer system and then *verified* against the constraint
Gram, and it is returned as the parameterization.  Nothing is discarded by a tolerance,
so the answer is exact whenever the integer checks pass.

The one numeric step in this stage is ``select_independent_rows``: the components a
finite-difference plan observes are Cartesian, and their independence is an algebraic
condition, so those rows are searched with a relative threshold while their *number* is
certified by the exact lattice dimension.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.linalg import qr

from mlfcs.interactions.algebra.exact import RANK_PRIMES, modular_echelon

# Rows of the Cartesian image are searched for the components a finite-difference plan
# observes; the count is certified by the exact lattice dimension, so this threshold only
# separates roundoff from $O(1)$ entries.  Independence in the Cartesian frame is an
# algebraic condition, which is why this single step is numeric.
_RELATIVE_EPS = 1e-9

# A ``float64`` mantissa holds 53 bits, so a rational with numerator and denominator
# below $2^{53/2}$ is the unique one within roundoff of its floating point value: that is
# the largest denominator a solve of an integer system can be reconstructed from, and it
# is derived here rather than assumed.
_RECONSTRUCTIBLE_DENOMINATOR = 1 << (53 // 2)


def label_symmetric_basis(cluster: tuple[int, ...]) -> np.ndarray:
    """Return the integer indicator basis of the label-symmetric classes.

    Columns are the equivalence classes of tensor components under permutations of
    atoms that carry the same label; entries are $0$ or $1$, so every later constraint
    matrix stays in the integers.
    """
    groups = []
    for atom in dict.fromkeys(cluster):
        positions = np.asarray(
            [index for index, value in enumerate(cluster) if value == atom], dtype=np.int32
        )
        if len(positions) > 1:
            groups.append(positions)
    size = 3 ** len(cluster)
    if not groups:
        return np.eye(size, dtype=np.int64)
    classes: dict[tuple[int, ...], list[int]] = {}
    for flat, component in enumerate(np.ndindex((3,) * len(cluster))):
        canonical = list(component)
        for positions in groups:
            ordered = sorted(canonical[int(position)] for position in positions)
            for position, direction in zip(positions, ordered, strict=True):
                canonical[int(position)] = direction
        classes.setdefault(tuple(canonical), []).append(flat)
    basis = np.zeros((size, len(classes)), dtype=np.int64)
    for column, members in enumerate(classes.values()):
        basis[members, column] = 1
    return basis


def invariant_kernel(
    label_basis: np.ndarray,
    actions,
    *,
    order: int,
) -> tuple[np.ndarray, int]:
    """Return an integer basis of the lattice invariant subspace and its dimension.

    ``label_basis`` is the integer label-symmetric indicator basis and ``actions`` the
    stabilizer actions of one orbit, all in the lattice frame.  The subspace is
    $\\{x : (S_a L - L)x = 0\\}$ over every stabilizer $a$, where $L$ is the label basis
    and $S_a$ the integer action matrix.  Returning fewer columns than the certified
    dimension is an error, never a silently smaller subspace.
    """
    label = np.asarray(label_basis, dtype=np.int64)
    size = 3**order
    if label.shape[0] != size:
        raise ValueError(f"label basis has {label.shape[0]} rows, expected {size}")
    columns = label.shape[1]
    gram = _constraint_gram(label, actions)
    if not gram.any():
        return np.eye(columns, dtype=np.int64), columns
    for prime in RANK_PRIMES:
        _, pivot_columns, pivot_rows = modular_echelon(gram, prime)
        dimension = columns - int(pivot_columns.size)
        if dimension <= 0:
            return np.empty((columns, 0), dtype=np.int64), 0
        try:
            basis = _solve_kernel(gram, pivot_columns, pivot_rows, dimension)
        except np.linalg.LinAlgError:
            # A rank-modulo-prime pivot block can still be singular over the rationals.
            continue
        except RuntimeError:
            # A different prime splits pivots differently, which can be reconstructible.
            continue
        if not (gram @ basis).any():
            return _primitive_columns(basis), dimension
    raise RuntimeError(
        "lattice-frame invariant kernel could not be verified over the integers "
        "(the constraint system has no solution within the reconstructed denominators)"
    )


def select_independent_rows(basis: np.ndarray, *, dimension: int) -> np.ndarray:
    """Return trailing rows of ``basis`` that determine the parameters.

    Rows are visited last to first and kept when they raise the rank, which fixes the
    components a finite-difference plan has to observe.  The number of rows is certified
    by the exact lattice-frame dimension of the same subspace, so a short selection is an
    error rather than a smaller parameter set; the rank test itself is numeric because
    the observed components are Cartesian, where independence is an algebraic rather than
    an integer condition.
    """
    matrix = np.asarray(basis, dtype=float)
    if dimension == 0:
        return np.empty(0, dtype=np.int32)
    if dimension > matrix.shape[1]:
        raise ValueError(f"cannot select {dimension} rows from {matrix.shape[1]} columns")
    threshold = _RELATIVE_EPS * max(float(np.max(np.abs(matrix))), 1.0)
    selected: list[int] = []
    for row in range(matrix.shape[0] - 1, -1, -1):
        trial = matrix[np.asarray((*selected, row), dtype=np.int64)]
        diagonal = np.abs(np.diag(qr(trial.T, mode="r", pivoting=False, check_finite=False)[0]))
        if int(np.count_nonzero(diagonal > threshold)) > len(selected):
            selected.append(row)
        if len(selected) == dimension:
            break
    if len(selected) != dimension:
        raise RuntimeError(f"selected {len(selected)} of {dimension} independent components")
    return np.asarray(sorted(selected), dtype=np.int32)


def _primitive_columns(basis: np.ndarray) -> np.ndarray:
    """Divide every column by the greatest common divisor of its entries.

    The solved columns carry an arbitrary integer scale from the reconstruction.  Removing
    it keeps the parameterization as small as the subspace allows, which is what the
    fitted parameter values and the conditioning of the downstream solves assume.
    """
    reduced = np.array(basis, dtype=np.int64, copy=True)
    for column in range(reduced.shape[1]):
        entries = reduced[:, column]
        divisor = int(np.gcd.reduce(np.abs(entries)))
        if divisor > 1:
            reduced[:, column] = entries // divisor
    return reduced


def _constraint_gram(label: np.ndarray, actions) -> np.ndarray:
    r"""Return the integer Gram of the stabilizer constraints.

    The kernel of $\sum_a (S_a L - L)^T (S_a L - L)$ is the kernel of every stabilizer
    residual, so the small $C \times C$ Gram replaces the stacked constraint matrix.
    Every entry is a sum of products of small integers, so the accumulation stays in
    ``int64`` and needs neither a rounding step nor a magnitude guard.
    """
    gram = np.zeros((label.shape[1], label.shape[1]), dtype=np.int64)
    for action in actions:
        residual = action.apply_scaled_columns(label) - label
        if residual.shape != label.shape:
            raise ValueError("stabilizer action did not preserve the label basis shape")
        gram += residual.T @ residual
    return gram


def _solve_kernel(
    rows: np.ndarray,
    pivot_columns: np.ndarray,
    pivot_rows: np.ndarray,
    dimension: int,
) -> np.ndarray:
    """Solve ``rows @ x = 0`` for integer columns, one per free component.

    Each column is assembled from the floating point solve and then checked against
    ``rows`` in integer arithmetic; a column that does not verify is reported instead of
    being returned.  The columns are independent by construction, because the entry of
    each free component is non-zero, so per-column verification covers the basis.
    """
    columns = rows.shape[1]
    free = np.setdiff1d(np.arange(columns, dtype=np.int64), pivot_columns, assume_unique=True)
    if free.size != dimension:
        raise RuntimeError("free component count disagrees with the certified dimension")
    block = rows[pivot_rows][:, pivot_columns].astype(float)
    solution = np.linalg.solve(block, -rows[pivot_rows][:, free].astype(float))
    basis = np.zeros((columns, dimension), dtype=np.int64)
    for local, row in enumerate(free):
        basis[:, local] = _kernel_column(rows, pivot_columns, int(row), solution[:, local])
    return basis


def _kernel_column(
    rows: np.ndarray,
    pivot_columns: np.ndarray,
    free_component: int,
    values: np.ndarray,
) -> np.ndarray:
    """Return one integer kernel column, verified against ``rows``.

    The solved column takes only a few distinct magnitudes, so each is reconstructed to
    an exact rational once.  The denominator bound is doubled until the assembled column
    satisfies ``rows @ column == 0`` exactly, which makes the bound a search rather than
    an assumption: a column that cannot be verified within the reconstructible
    denominator range is reported.  The entry of the free component is its scale, so the
    column is non-zero.
    """
    magnitudes = np.unique(np.abs(values))
    bound = 1
    while bound <= _RECONSTRUCTIBLE_DENOMINATOR:
        fractions = [
            Fraction(float(magnitude)).limit_denominator(bound) for magnitude in magnitudes
        ]
        scale = 1
        for fraction in fractions:
            denominator = fraction.denominator
            scale = scale * denominator // int(np.gcd(scale, denominator))
        magnitude_integers = np.asarray(
            [int(fraction * scale) for fraction in fractions], dtype=np.int64
        )
        positions = np.searchsorted(magnitudes, np.abs(values))
        column = np.zeros(rows.shape[1], dtype=np.int64)
        column[pivot_columns] = np.where(
            values < 0, -magnitude_integers[positions], magnitude_integers[positions]
        )
        column[free_component] = scale
        if not (rows @ column).any():
            return column
        bound *= 2
    raise RuntimeError(
        "solved kernel column could not be reconstructed as integers within the "
        "denominator range of a float64 mantissa"
    )


__all__ = [
    "invariant_kernel",
    "label_symmetric_basis",
    "select_independent_rows",
]
