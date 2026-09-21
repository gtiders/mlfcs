"""Exact integer invariant-tensor kernels in the lattice (scaled) frame.

A spglib symmetry rotation is an integer matrix in lattice coordinates for every cell,
including the cells whose Cartesian rotations are irrational (an fcc primitive
60 degree cell, hexagonal and rhombohedral cells).  Every decision here therefore
happens on integers:

* the label-symmetric basis is a 0/1 indicator basis, so the constraint matrix stays
  integral;
* the invariant dimension is ``columns - rank`` of the stacked stabilizer residuals,
  with the rank certified by the modular certificate of
  :mod:`mlfcs.interactions.algebra.exact`;
* the kernel basis is the *saturated* integer kernel of that constraint matrix, so the
  parameter columns span exactly the integer solutions and no scale of a solution is
  missing;
* the result is verified against the constraint matrix in exact integer arithmetic
  before it is returned.

Nothing is discarded by a tolerance, and no Gram matrix ``M^T M`` is formed: it squares
the integer magnitudes and can lose rank modulo a prime that the direct constraint
matrix still certifies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlfcs.interactions.algebra.actions import as_int64
from mlfcs.interactions.algebra.exact import certified_pivots, saturated_kernel, verify_kernel


def label_symmetric_basis(cluster: Sequence[int]) -> np.ndarray:
    """Return the integer indicator basis of the label-symmetric classes.

    Columns are the equivalence classes of tensor components under permutations of atoms
    carrying the same label, and entries are 0 or 1.  Keeping the basis integral is what
    lets every later constraint matrix stay in the integers.
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


def constraint_rows(label_basis: np.ndarray, actions) -> np.ndarray:
    r"""Stack the lattice residual constraints of every stabilizer action.

    One stabilizer $a$ contributes the rows $(S_a L - L)^T$, where $L$ is the label
    basis and $S_a$ its exact integer lattice action matrix.  The invariant parameters
    are the vectors $x$ with $(S_a L - L)x = 0$ for every stabilizer, so this matrix
    alone decides the invariant subspace; it stays in the integers.
    """
    label = as_int64(np.asarray(label_basis), context="label basis")
    if label.ndim != 2:
        raise ValueError(f"expected a 2D label basis, got shape {label.shape}")
    blocks = []
    for action in actions:
        rotated = action.apply_scaled_columns(label)
        if rotated.shape != label.shape:
            raise ValueError(
                f"stabilizer action returned shape {rotated.shape} for label basis {label.shape}"
            )
        blocks.append(rotated - label)
    if not blocks:
        return np.empty((0, label.shape[1]), dtype=np.int64)
    return np.ascontiguousarray(np.vstack(blocks))


def invariant_kernel(
    label_basis: np.ndarray,
    actions,
    *,
    order: int,
) -> tuple[np.ndarray, int]:
    """Return an integer basis of the lattice invariant subspace and its dimension.

    The dimension is certified exactly (a modular rank never overestimates it, and the
    Hadamard certificate settles deficient matrices), so a shorter basis can never be
    returned silently.
    """
    label = as_int64(np.asarray(label_basis), context="label basis")
    expected = 3**order
    if label.shape[0] != expected:
        raise ValueError(f"label basis has {label.shape[0]} rows, expected {expected}")
    columns = int(label.shape[1])
    rows = constraint_rows(label, actions)
    rank, pivot_rows, _ = certified_pivots(rows)
    dimension = columns - rank
    if dimension == 0:
        return np.empty((columns, 0), dtype=np.int64), 0
    basis = saturated_kernel(rows[pivot_rows])
    if basis.shape[1] != dimension:
        raise RuntimeError(
            f"the saturated kernel of the stabilizer constraints has {basis.shape[1]} "
            f"columns, but the certified invariant dimension is {dimension}"
        )
    basis = primitive_columns(basis)
    verify_kernel(rows, basis)
    return basis, dimension


def primitive_columns(basis: np.ndarray) -> np.ndarray:
    """Divide every column by the greatest common divisor of its entries.

    The solved columns carry an arbitrary integer scale from the kernel construction;
    removing it keeps the parameterization as small as the subspace allows, which the
    fitted parameter magnitudes and the downstream conditioning assume.  The lattice is
    unchanged: a saturated kernel contains every division of its own vectors by their
    content.
    """
    reduced = as_int64(np.asarray(basis), context="invariant basis").copy()
    for column in range(reduced.shape[1]):
        entries = reduced[:, column]
        divisor = int(np.gcd.reduce(np.abs(entries)))
        if divisor > 1:
            reduced[:, column] = entries // divisor
    return reduced


__all__ = [
    "constraint_rows",
    "invariant_kernel",
    "label_symmetric_basis",
    "primitive_columns",
]
