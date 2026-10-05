"""Rational rank and lattice kernel bases using modular matrix operations."""

from __future__ import annotations

import math

import numpy as np

from mlfcs._arrays import as_int64_array, require_allocation
from mlfcs.algebra import _congruence, _modular, _signed
from mlfcs.algebra.matrix import to_python_rows
from mlfcs.errors import RankError

RANK_PRIMES = _modular.RANK_PRIMES


def _matrix(matrix):
    """Prepare a readonly, C-contiguous int64 matrix.

    ``matrix`` must have shape (m, n); an empty sequence becomes shape (0, 0).
    Entries follow as_int64_array's symmetric range, excluding INT64_MIN.
    Invalid shape or entry types raise ValueError; range failures raise
    OverflowError. Writable input storage is copied.
    """
    array = np.asarray(matrix)
    if array.ndim == 1 and array.size == 0:
        array = np.empty((0, 0), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("expected a 2-D integer matrix")
    return as_int64_array(array, name="matrix")


def _is_prime(n):
    """Test primality for candidates in prime_stream's range.

    ``n`` is an integer below 2**31. Return bool using small-prime division
    and Miller-Rabin bases 2, 3, 5, 7 and 11; values below two return False.
    """
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
    """Yield primes used for modular rank and kernel verification.

    Begin with RANK_PRIMES, then yield smaller odd primes in descending order,
    ending with two. Every yielded modulus is below 2**31.
    """
    yield from RANK_PRIMES
    candidate = RANK_PRIMES[-1] - 2
    while candidate >= 3:
        if _is_prime(candidate):
            yield candidate
        candidate -= 2
    yield 2


def hadamard_bound(matrix, size):
    """Bound absolute minors using Hadamard's determinant inequality.

    ``matrix`` is an (m, n) matrix accepted by to_python_rows; ``size`` is the
    requested minor order, at most min(m, n). Return the Python int product
    of the largest ``size`` row norms rounded to isqrt(sum(v*v)) + 1.
    Nonpositive sizes return one. Invalid matrix entries raise ValueError.
    """
    if size <= 0:
        return 1
    rows = to_python_rows(matrix)
    norms = sorted((math.isqrt(sum(v * v for v in row)) + 1 for row in rows), reverse=True)
    return math.prod(norms[:size])


def rank_pivots(matrix):
    """Return the rational rank and independent original row/column indices.

    Parameters
    ----------
    matrix : array_like, shape (m, n)
        Integer entries in [-INT64_MAX, INT64_MAX]. Floating inputs are rejected.

    Returns
    -------
    rank : int
        Dimension of the row space over the rationals.
    rows, columns : ndarray of int64, shape (rank,)
        Original indices defining a nonsingular minor at a tested prime.
        Output index buffers are writable.

    Raises
    ------
    ValueError
        Input is not a two-dimensional matrix of integers.
    OverflowError
        Input entries or storage exceed their supported ranges.
    RankError
        Available primes do not determine the rank.

    Notes
    -----
    Modular rank is a lower bound. A full-rank minor determines the result;
    otherwise, distinct primes whose product exceeds the Hadamard minor bound
    prove that all larger minors vanish. Primes are below 2**31.
    """
    a = _matrix(matrix)
    target = min(a.shape)
    if target == 0:
        empty = np.empty(0, dtype=np.int64)
        return 0, empty, empty
    bound = hadamard_bound(a, target)
    product, best = 1, 0
    rows = columns = np.empty(0, dtype=np.int64)
    for prime in prime_stream():
        _, picked_columns, picked_rows = _modular.echelon(a, prime)
        if len(picked_columns) > best:
            best, rows, columns = len(picked_columns), picked_rows, picked_columns
        product *= prime
        # Full rank has a nonzero modular witness. Otherwise every larger
        # minor vanishes at all tested primes, whose product exceeds its bound.
        if best == target or product > bound:
            return best, rows, columns
    raise RankError("prime stream exhausted before determining matrix rank")


def rank(matrix):
    """Return the dimension of a matrix's row space over the rationals.

    ``matrix`` has shape (m, n) with integer entries. Return an int in
    0..min(m, n), determined by rank_pivots without a floating-point tolerance.
    ValueError, OverflowError and RankError follow rank_pivots's input and
    modulus constraints. The matrix is not modified.
    """
    return rank_pivots(matrix)[0]


def verify_kernel(matrix, basis):
    """Verify the matrix identity A @ B == 0 by modular products.

    A has shape (m, n) and B (n, d), with int64 entries excluding INT64_MIN.
    A nonzero residue raises ValueError. Once the product of tested primes
    exceeds n*max(abs(A))*max(abs(B)), zero residues prove the identity.
    Return None; this checks annihilation, not independence or saturation.
    Incompatible shapes or entry types raise ValueError; input range failures
    raise OverflowError, and exhausted primes raise RankError. Inputs are
    not modified.
    """
    a, b = _matrix(matrix), _matrix(basis)
    if a.shape[1] != b.shape[0]:
        raise ValueError("kernel basis has incompatible row count")
    maximum_a = max((abs(int(v)) for v in a.flat), default=0)
    maximum_b = max((abs(int(v)) for v in b.flat), default=0)
    bound = a.shape[1] * maximum_a * maximum_b
    product = 1
    for prime in prime_stream():
        if not _modular.product_is_zero_mod(a, b, prime):
            raise ValueError("matrix @ basis is nonzero modulo the tested prime")
        product *= prime
        if product > bound:
            return
    raise RankError("prime stream exhausted before verifying matrix @ basis == 0")


def kernel_basis(matrix, *, expected_nullity=None):
    """Return a column basis generating all integer solutions of matrix @ x == 0.

    Parameters
    ----------
    matrix : array_like, shape (m, n)
        Integer entries in [-INT64_MAX, INT64_MAX], excluding INT64_MIN.
    expected_nullity : int, optional
        Required kernel dimension; mismatches reject the selected chart.

    Returns
    -------
    basis : ndarray of int64, shape (n, nullity)
        Readonly columns generate the saturated solution lattice. Generator
        orientation is not canonical.

    Raises
    ------
    ValueError
        Input or expected signed-kernel dimension is invalid.
    RankError
        The chosen modular chart fails its pivot/rank conditions.
    OverflowError
        Allocation or chart reconstruction exceeds the supported ranges.

    Notes
    -----
    Signed incidence rows use a component basis. Other matrices use two-prime
    pivot charts and rational reconstruction. Free coordinates z must
    satisfy F @ z == 0 modulo delta; a column-HNF preimage basis H yields
    (-F @ H / delta, H), with coordinates restored to original column order.
    Modular products verify annihilation. Denominator, scaled chart entries
    and quotient accumulators are validated before lattice saturation.
    There is no arbitrary-precision elimination fallback.
    """
    a = _matrix(matrix)
    n = a.shape[1]
    require_allocation("signed component workspace", (n,))
    parent, signs, dead, columns, dimension = _signed.signed_components(a)
    if dimension >= 0:
        require_allocation("signed kernel basis", (n, dimension))
        signed = _signed.component_basis(parent, signs, dead, columns, dimension)
        if expected_nullity is not None and expected_nullity != dimension:
            raise ValueError("expected nullity differs from signed-component dimension")
        signed.setflags(write=False)
        return signed
    choices = [_modular.echelon(a, p) for p in RANK_PRIMES]
    chosen = max(choices, key=lambda choice: len(choice[1]))
    rank, pivots, rows = len(chosen[1]), chosen[1], chosen[2]
    if expected_nullity is not None and n - rank != expected_nullity:
        raise RankError("selected chart does not attain the expected rank")
    pivot_set = {int(j) for j in pivots}
    free = np.asarray([j for j in range(n) if j not in pivot_set], dtype=np.int64)
    order = np.concatenate((pivots, free))
    # A single pivot chart must work at both reconstruction primes; choosing
    # separate pivot coordinates would make the residues incomparable.
    selected = np.ascontiguousarray(a[rows][:, order])
    charts = [_modular.chart(selected, rank, p) for p in RANK_PRIMES]
    if not all(ok for _, ok in charts):
        raise RankError("fixed chart pivot minor is singular at a reconstruction prime")
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
        # Clearing denominators alone gives a possibly proper sublattice.
        # The congruence preimage recovers every admissible integer free vector.
        h = _congruence.congruence_preimage(f, delta)
        lifted = _congruence.lift_kernel(f, delta, h)
        result = np.empty((n, d), dtype=np.int64)
        result[order] = lifted
        verify_kernel(a, result)
        result.setflags(write=False)
        return result
    raise OverflowError("kernel chart reconstruction exceeds the supported int64 domain")


__all__ = [
    "RANK_PRIMES",
    "hadamard_bound",
    "kernel_basis",
    "prime_stream",
    "rank",
    "rank_pivots",
    "verify_kernel",
]
