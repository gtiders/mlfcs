"""Certified machine-word exact algebra; arbitrary precision is used only for bounds.

Production kernels use int64. Matrices outside the admitted domain are rejected;
SymPy's former implementation lives in tests/oracles/exact_reference.py.
"""

from __future__ import annotations

import math
import operator

import numpy as np

from mlfcs._arrays import integer_array, require_allocation
from mlfcs.algebra import _congruence, _modular, _signed
from mlfcs.errors import RankCertificateError

RANK_PRIMES = _modular.RANK_PRIMES


def to_python_rows(matrix) -> list[list[int]]:
    """Rows of ``matrix`` as Python ints, exactly.

    Nested lists or tuples, NumPy integer arrays of any width and object arrays
    holding Python ints (arbitrary precision) are accepted. Floating-point values
    are refused even when integral: an exact integer fact has to be declared as
    such. Any other entry, and input that is not two-dimensional, raises
    ``ValueError``. The empty sequence reads as the 0x0 matrix.
    """
    return _integer_rows(matrix)[2]


def _integer_rows(matrix) -> tuple[int, int, list[list[int]]]:
    """Return ``(height, width, rows)`` of exact Python-int rows for any input."""
    array = matrix if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype=object)
    if array.ndim == 1 and array.size == 0:
        return 0, 0, []
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D integer matrix, got shape {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    if array.dtype.kind in "iub":
        return height, width, [[int(value) for value in row] for row in array.tolist()]
    rows = [
        [_exact_int(value, position=(index, column)) for column, value in enumerate(values)]
        for index, values in enumerate(array.tolist())
    ]
    return height, width, rows


def _exact_int(value, position: tuple[int, int]) -> int:
    """Convert one input entry to a Python int or reject it."""
    try:
        return operator.index(value)
    except TypeError:
        pass
    raise ValueError(f"matrix entry at {position} is not an integer: {value!r}")


def _matrix(matrix):
    array = np.asarray(matrix)
    if array.ndim == 1 and array.size == 0:
        array = np.empty((0, 0), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("expected a 2-D integer matrix")
    return integer_array(array, name="exact matrix")


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for base in (2, 3, 5, 7, 11):
        value = pow(base, d, n)
        if value in (1, n - 1):
            continue
        for _ in range(s - 1):
            value = value * value % n
            if value == n - 1:
                break
        else:
            return False
    return True


def prime_stream():
    yield from RANK_PRIMES
    candidate = RANK_PRIMES[-1] - 2
    while candidate >= 3:
        if _is_prime(candidate):
            yield candidate
        candidate -= 2
    yield 2


def hadamard_bound(matrix, size):
    if size <= 0:
        return 1
    rows = to_python_rows(matrix)
    norms = sorted((math.isqrt(sum(v * v for v in row)) + 1 for row in rows), reverse=True)
    return math.prod(norms[:size])


def certified_pivots(matrix):
    a = _matrix(matrix)
    target = min(a.shape)
    if target == 0:
        empty = np.empty(0, dtype=np.int64)
        return 0, empty, empty
    # Only a bound/certificate uses Python big integers; no integer elimination.
    bound = hadamard_bound(a, target)
    product, best = 1, 0
    rows = columns = np.empty(0, dtype=np.int64)
    for prime in prime_stream():
        _, picked_columns, picked_rows = _modular.echelon(a, prime)
        if len(picked_columns) > best:
            best, rows, columns = len(picked_columns), picked_rows, picked_columns
        product *= prime
        if best == target or product > bound:
            return best, rows, columns
    raise RankCertificateError("prime stream exhausted before exact rank certification")


def exact_rank(matrix):
    return certified_pivots(matrix)[0]


def verify_kernel(matrix, basis):
    a, b = _matrix(matrix), _matrix(basis)
    if a.shape[1] != b.shape[0]:
        raise ValueError("kernel basis has incompatible row count")
    maximum_a = max((abs(int(v)) for v in a.flat), default=0)
    maximum_b = max((abs(int(v)) for v in b.flat), default=0)
    bound = a.shape[1] * maximum_a * maximum_b
    product = 1
    for prime in prime_stream():
        if not _modular.product_is_zero_mod(a, b, prime):
            raise ValueError("matrix @ basis is nonzero (exact modular certificate)")
        product *= prime
        if product > bound:
            return
    raise RankCertificateError("prime stream exhausted before annihilation certification")


def exact_kernel(matrix, *, expected_nullity=None):
    """Return a readonly saturated int64 kernel basis after local validation."""
    a = _matrix(matrix)
    n = a.shape[1]
    require_allocation("signed component workspace", (n,))
    parent, signs, dead, columns, dimension = _signed.signed_components(a)
    if dimension >= 0:
        require_allocation("signed kernel basis", (n, dimension))
        signed = _signed.component_basis(parent, signs, dead, columns, dimension)
        if expected_nullity is not None and expected_nullity != dimension:
            raise ValueError("expected nullity differs from certified signed-kernel nullity")
        signed.setflags(write=False)
        return signed
    choices = [_modular.echelon(a, p) for p in RANK_PRIMES]
    chosen = max(choices, key=lambda choice: len(choice[1]))
    rank, pivots, rows = len(chosen[1]), chosen[1], chosen[2]
    if expected_nullity is not None and n - rank != expected_nullity:
        raise RankCertificateError("selected chart does not attain the expected exact rank")
    pivot_set = {int(j) for j in pivots}
    free = np.asarray([j for j in range(n) if j not in pivot_set], dtype=np.int64)
    order = np.concatenate((pivots, free))
    selected = np.ascontiguousarray(a[rows][:, order])
    charts = [_modular.chart(selected, rank, p) for p in RANK_PRIMES]
    if not all(ok for _, ok in charts):
        raise RankCertificateError("fixed chart pivot minor is singular at a reconstruction prime")
    d = len(free)
    if d == 0:
        result = np.empty((n, 0), dtype=np.int64)
        result.setflags(write=False)
        return result
    require_allocation("kernel basis", (n, d))
    require_allocation("congruence workspace", (d, d))
    for exponent in range(31):
        numerators, denominators, ok = _modular.reconstruct(
            charts[0][0], charts[1][0], 1 << exponent
        )
        if not ok:
            continue
        f, delta, ok = _modular.common_denominator(numerators, denominators)
        if not ok:
            continue
        chart_bound = max((abs(int(v)) for v in f.flat), default=0)
        accumulator = d * (chart_bound + 2 * delta + 1)
        if accumulator > np.iinfo(np.int64).max:
            continue
        rational = np.zeros((n, d), dtype=np.int64)
        rational[pivots] = -f
        rational[free] = delta * np.eye(d, dtype=np.int64)
        try:
            verify_kernel(a, rational)
        except ValueError:
            continue
        h = _congruence.congruence_preimage(f, delta)
        lifted = _congruence.lift_kernel(f, delta, h)
        result = np.empty((n, d), dtype=np.int64)
        result[order] = lifted
        verify_kernel(a, result)
        result.setflags(write=False)
        return result
    raise OverflowError("exact kernel chart cannot be certified in the bounded int64 domain")


__all__ = [
    "RANK_PRIMES",
    "certified_pivots",
    "exact_kernel",
    "exact_rank",
    "hadamard_bound",
    "prime_stream",
    "to_python_rows",
    "verify_kernel",
]
