"""Finite reciprocal grids and their irreducible symmetry stars.

For a row-vector supercell matrix ``S``, define ``D = abs(det(S))``. An
integer label ``l`` belongs to its reciprocal grid when
``l @ S.T == 0 (mod D)`` and represents the fractional point ``q = l / D``.
Grid construction and symmetry actions operate on exact integer labels;
floating-point q coordinates are formed for Fourier calculations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import as_int64_array, require_allocation, require_bound
from mlfcs.foundation.integer import (
    adjugate_3x3,
    as_int64,
    determinant_3x3,
    to_python_rows,
)
from mlfcs.foundation.log import get_logger
from mlfcs.geometry.symmetry import PrimitiveSymmetry

logger = get_logger(__name__)


def _unimodular_inverse(matrix: object) -> np.ndarray:
    """Return the exact integer inverse of a 3 by 3 unimodular matrix.

    For determinant ``±1``, the inverse is ``det(matrix) * adjugate(matrix)``.
    """
    determinant = determinant_3x3(matrix)
    if abs(determinant) != 1:
        raise ValueError("matrix must be unimodular")
    return determinant * adjugate_3x3(matrix)


def _integer_matmul(left: object, right: object) -> np.ndarray:
    """Multiply integer matrices after proving every dot product fits int64."""
    a = as_int64(left, context="left operand")
    b = as_int64(right, context="right operand")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matrix shapes {a.shape} and {b.shape}")
    require_allocation("matrix product", (a.shape[0], b.shape[1]))
    _validate_integer_matmul(a, b)
    return _integer_matmul_kernel(a, b)


def _validate_integer_matmul(a: np.ndarray, b: np.ndarray) -> None:
    """Bound each integer dot product before the compiled matrix product."""
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            require_bound(
                "matrix product",
                sum(abs(int(a[i, k]) * int(b[k, j])) for k in range(a.shape[1])),
            )


@njit(cache=True)
def _integer_matmul_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the validated int64 matrix product used by reciprocal labels."""
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] += a[i, k] * b[k, j]
    return result


@njit(cache=True)
def _modular_matmul(a: np.ndarray, b: np.ndarray, modulus: int) -> np.ndarray:
    """Return the integer matrix product modulo ``modulus``.

    Operands and partial sums are reduced at each step to keep intermediates in
    signed int64. Callers validate dimensions and modulus before entry.
    """
    result = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                result[i, j] = (result[i, j] + (a[i, k] % modulus) * (b[k, j] % modulus)) % modulus
    return result


def rotate_q_labels(
    labels: object, rotation: object, *, denominator: int | None = None
) -> np.ndarray:
    """Apply a direct-lattice symmetry operation to reciprocal integer labels.

    If fractional row coordinates transform as ``x @ R.T``, reciprocal row
    labels transform by the dual action ``label @ R^-1``. When ``denominator``
    is supplied, reduce the result modulo it to obtain canonical labels on a
    finite reciprocal grid.
    """
    values = np.asarray(labels)
    if values.shape[-1:] != (3,):
        raise ValueError("reciprocal labels must end in shape (3,)")
    integer = as_int64_array(values, name="reciprocal labels")
    inverse = _unimodular_inverse(rotation)
    if denominator is None:
        return _integer_matmul(integer.reshape(-1, 3), inverse).reshape(integer.shape)
    if isinstance(denominator, (bool, np.bool_)) or not isinstance(denominator, (int, np.integer)):
        raise TypeError("denominator must be a positive integer")
    if denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    return _modular_matmul(integer.reshape(-1, 3), inverse, int(denominator)).reshape(integer.shape)


def _normalize_mesh_matrix(source: object) -> np.ndarray:
    """Normalize diagonal mesh sizes or an integer supercell matrix."""
    source = np.asarray(source, dtype=object)
    if source.shape == (3,):
        sizes = as_int64_array(source, name="mesh sizes")
        if np.any(sizes <= 0):
            raise ValueError("diagonal mesh sizes must be positive integers")
        source = np.diag(sizes)
    rows = to_python_rows(source)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("supercell matrix must be a 3 by 3 integer matrix")
    # The matrix algebra uses int64 inputs and Python-integer
    # intermediates. In particular, object-dtype integer matrix inputs are valid.
    values = np.asarray(rows, dtype=np.int64)
    if determinant_3x3(values) == 0:
        raise ValueError("supercell matrix must be nonsingular")
    values.setflags(write=False)
    return values


def as_qgrid(mesh: object) -> QGrid:
    """Return ``mesh`` as a ``QGrid``, constructing one when necessary."""
    return mesh if isinstance(mesh, QGrid) else QGrid(mesh)


def mass_preserving_symmetry(symmetry: PrimitiveSymmetry, masses: np.ndarray) -> PrimitiveSymmetry:
    """Return the symmetry subgroup that preserves the atomic mass assignment.

    Operations mapping any motif site to a site of different mass are removed.
    Geometric symmetry and force-constant parameterization are unchanged.
    """
    keep = np.all(masses[symmetry.site_permutations] == masses[None, :], axis=1)
    if np.all(keep):
        return symmetry
    return PrimitiveSymmetry(
        rotations=symmetry.rotations[keep],
        translations=symmetry.translations[keep],
        cartesian_rotations=symmetry.cartesian_rotations[keep],
        site_permutations=symmetry.site_permutations[keep],
        site_shifts=symmetry.site_shifts[keep],
        symbol=symmetry.symbol,
        symprec=symmetry.symprec,
    )


@dataclass(frozen=True, slots=True, init=False)
class QGrid:
    """Finite reciprocal grid associated with an integer supercell lattice.

    For supercell matrix ``S`` and ``D = abs(det(S))``, the grid contains exactly
    ``D`` integer labels ``l`` satisfying ``l @ S.T == 0 (mod D)``. Each label
    represents the fractional reciprocal point ``q = l / D``. Labels are
    canonical representatives in ``[0, D)`` ordered lexicographically. Grid
    construction uses integer quotient arithmetic rather than floating-point
    q-point matching.
    """

    matrix: np.ndarray
    labels: np.ndarray
    denominator: int

    def __init__(self, matrix: object) -> None:
        """Enumerate the reciprocal grid defined by a supercell matrix.

        ``matrix`` is either three positive diagonal mesh sizes or a nonsingular
        integer matrix with ``cell_super = matrix @ cell_primitive``. The grid
        contains ``abs(det(matrix))`` labels representing fractional reciprocal
        coordinates in the primitive cell basis.
        """
        values = _normalize_mesh_matrix(matrix)
        denominator = abs(determinant_3x3(values))
        adjugate = adjugate_3x3(values)
        generators = [tuple(int(value) % denominator for value in row) for row in adjugate.T]
        zero = (0, 0, 0)
        labels = {zero}
        pending = deque([zero])
        while pending:
            label = pending.popleft()
            for generator in generators:
                image = tuple(
                    (value + step) % denominator
                    for value, step in zip(label, generator, strict=True)
                )
                if image not in labels:
                    labels.add(image)
                    pending.append(image)
        if len(labels) != denominator:
            raise RuntimeError("reciprocal quotient has the wrong number of q points")
        ordered = np.asarray(sorted(labels), dtype=np.int64)
        products = ordered.astype(object) @ values.T.astype(object)
        if any(int(value) % denominator for value in products.flat):
            raise RuntimeError("reciprocal quotient contains a label outside the dual lattice")
        ordered.setflags(write=False)
        object.__setattr__(self, "matrix", values)
        object.__setattr__(self, "labels", ordered)
        object.__setattr__(self, "denominator", denominator)
        logger.info(
            "Reciprocal grid built: matrix=%s determinant=%d qpoints=%d",
            values.tolist(),
            denominator,
            self.size,
        )

    @property
    def points(self) -> np.ndarray:
        """Fractional q coordinates in the stored label order."""
        return self.labels.astype(np.float64) / self.denominator

    @property
    def size(self) -> int:
        """Number of reciprocal grid points, equal to ``abs(det(matrix))``."""
        return len(self.labels)


def _compatible(
    matrix: np.ndarray, adjugate: np.ndarray, rotation: np.ndarray, denominator: int
) -> bool:
    """Return whether a primitive rotation preserves the supercell lattice.

    Compatibility requires ``S @ R.T @ S^-1`` to have integer entries.
    """
    product = matrix.astype(object) @ rotation.T.astype(object) @ adjugate.astype(object)
    return all(int(value) % denominator == 0 for value in product.flat)


@dataclass(frozen=True, slots=True, init=False)
class QStars:
    """Irreducible symmetry stars of a finite reciprocal grid.

    Grid points related by supercell-compatible primitive symmetry operations,
    and optionally by time reversal, belong to a common star. Every full-grid
    member retains its irreducible representative, the symmetry operation
    mapping that representative to the member, and whether time reversal is
    used. ``weights`` counts the full-grid points in each star.
    """

    grid: QGrid
    symmetry: PrimitiveSymmetry
    representatives: np.ndarray
    star_of: np.ndarray
    operations: np.ndarray
    antiunitary: np.ndarray
    weights: np.ndarray
    time_reversal: bool

    def __init__(
        self, grid: QGrid, symmetry: PrimitiveSymmetry, *, time_reversal: bool = True
    ) -> None:
        """Partition a reciprocal grid into irreducible symmetry stars.

        Only primitive symmetry operations that preserve the supercell lattice
        participate. If ``time_reversal`` is true, ``q`` and ``-q`` are grouped
        in the same star. Each member stores a route from its representative
        through a symmetry operation and, when needed, time reversal.
        """
        if not isinstance(grid, QGrid) or not isinstance(symmetry, PrimitiveSymmetry):
            raise TypeError("grid and symmetry must be QGrid and PrimitiveSymmetry")
        n = grid.size
        labels = grid.labels
        denominator = grid.denominator
        lookup = {tuple(int(value) for value in label): index for index, label in enumerate(labels)}
        adjugate = adjugate_3x3(grid.matrix)
        identity = next(
            (
                index
                for index, rotation in enumerate(symmetry.rotations)
                if np.array_equal(rotation, np.eye(3, dtype=np.int32))
                and np.array_equal(symmetry.translations[index], np.zeros(3))
                and np.array_equal(
                    symmetry.site_permutations[index],
                    np.arange(symmetry.site_permutations.shape[1]),
                )
                and not np.any(symmetry.site_shifts[index])
            ),
            None,
        )
        if identity is None:
            raise ValueError("primitive symmetry does not contain the identity operation")
        compatible = [
            index
            for index, rotation in enumerate(symmetry.rotations)
            if _compatible(grid.matrix, adjugate, rotation, denominator)
        ]
        compatible.remove(identity)
        compatible.insert(0, identity)
        actions: dict[int, np.ndarray] = {}
        for operation in compatible:
            moved = rotate_q_labels(labels, symmetry.rotations[operation], denominator=denominator)
            try:
                targets = np.asarray(
                    [lookup[tuple(int(value) for value in label)] for label in moved],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise RuntimeError(
                    f"grid-preserving operation {operation} maps outside the reciprocal grid"
                ) from error
            if len(np.unique(targets)) != n:
                raise RuntimeError(f"operation {operation} does not permute the reciprocal grid")
            actions[operation] = targets
        negatives = np.asarray(
            [lookup[tuple(-int(value) % denominator for value in label)] for label in labels],
            dtype=np.int64,
        )

        star_of = np.full(n, -1, dtype=np.int64)
        operations = np.full(n, -1, dtype=np.int64)
        antiunitary = np.zeros(n, dtype=bool)
        representatives: list[int] = []
        weights: list[int] = []
        for representative in range(n):
            if star_of[representative] >= 0:
                continue
            routes: dict[int, tuple[int, bool]] = {}
            for operation, permutation in actions.items():
                routes.setdefault(int(permutation[representative]), (operation, False))
            if time_reversal:
                for operation, permutation in actions.items():
                    routes.setdefault(
                        int(negatives[permutation[representative]]), (operation, True)
                    )
            members = set(routes)
            if any(star_of[member] >= 0 for member in members):
                raise RuntimeError("primitive operations do not form closed reciprocal stars")
            if any(
                int(permutation[member]) not in members
                for member in members
                for permutation in actions.values()
            ) or (
                time_reversal and any(int(negatives[member]) not in members for member in members)
            ):
                raise RuntimeError("primitive operations do not form closed reciprocal stars")
            star = len(representatives)
            for member in members:
                star_of[member] = star
                operations[member], antiunitary[member] = routes[member]
            representatives.append(representative)
            weights.append(len(members))
        if np.any(star_of < 0) or sum(weights) != n:
            raise RuntimeError("reciprocal stars do not partition the grid")
        arrays = (
            np.asarray(representatives, dtype=np.int64),
            star_of,
            operations,
            antiunitary,
            np.asarray(weights, dtype=np.int64),
        )
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "symmetry", symmetry)
        for name, value in zip(
            ("representatives", "star_of", "operations", "antiunitary", "weights"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "time_reversal", bool(time_reversal))
        logger.info(
            "Reciprocal stars: grid=%d irreducible=%d reduction=%.3fx "
            "time_reversal=%s symmetry_operations=%d",
            n,
            len(self.representatives),
            n / len(self.representatives),
            time_reversal,
            len(compatible),
        )

    @property
    def points(self) -> np.ndarray:
        """Fractional q coordinates of the irreducible representatives."""
        return self.grid.points[self.representatives]

    def expand(self, values: object) -> np.ndarray:
        """Expand a star-invariant quantity from irreducible to full-grid order.

        ``values`` has one leading entry per star. Expansion uses star membership
        only, so it suits symmetry-invariant scalars such as eigenvalues.
        Matrix-valued quantities with a nontrivial symmetry or positional-gauge
        transformation require the stored symmetry routes instead.
        """
        array = np.asarray(values)
        if array.ndim == 0 or len(array) != len(self.representatives):
            raise ValueError("values must have one leading entry per irreducible q point")
        return array[self.star_of]

    def member_shift(self, member: int) -> tuple[int, int, int]:
        """Return the reciprocal-lattice shift associated with a star route.

        Let ``r`` be the representative label and ``m`` the stored member. For
        the route's rotation ``R`` and time-reversal sign ``s = ±1``, the
        returned integer vector ``G`` satisfies

            s * (r @ R^-1) = m + D * G,

        where ``D`` is the grid denominator. ``G`` determines the positional-
        gauge phase when transforming matrix-valued reciprocal quantities.
        """
        if not 0 <= member < self.grid.size:
            raise IndexError("member is outside the reciprocal grid")
        representative = self.grid.labels[self.representatives[self.star_of[member]]]
        operation = int(self.operations[member])
        moved = rotate_q_labels(representative, self.symmetry.rotations[operation])
        sign = -1 if self.antiunitary[member] else 1
        difference = [
            sign * int(value) - int(stored)
            for value, stored in zip(moved, self.grid.labels[member], strict=True)
        ]
        denominator = self.grid.denominator
        if any(value % denominator for value in difference):
            raise RuntimeError("star member has a nonintegral reciprocal lattice shift")
        return tuple(value // denominator for value in difference)


__all__ = ["QGrid", "QStars"]
