r"""Exact integer rank kernels for symmetry algebra.

A rank modulo a prime can only *underestimate* the rank over $\mathbb{Q}$, because a rank
deficiency over $\mathbb{Q}$ that survives reduction modulo $p$ is a deficiency modulo $p$
as well.  That direction is usable as a proof:

* if a prime gives the full column rank, the exact rank is settled immediately;
* otherwise the deficiency is certified by *adding primes until their product exceeds the
  Hadamard bound of the largest minors*: if every prime were defective, each of them would
  divide a non-zero maximal minor, so their product could not exceed that minor's absolute
  value, which is bounded by the Hadamard bound.  One prime therefore has to be good, and
  the largest modular rank seen is the exact rank.

Both directions are thus decided by the same arithmetic, and the fraction-free elimination
that used to settle the deficient verdict is no longer needed.  Everything stays in
``int64``: the primes are below $2^{31}$, so products of two residues fit.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import isqrt

import numpy as np

# Primes close to $2^{31}$: entries and products stay inside ``int64`` and a random
# matrix of small integers is deficient modulo either one only exceptionally.
RANK_PRIMES = (2147483647, 2147483629)


def modular_echelon(matrix: np.ndarray, prime: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a row echelon form modulo ``prime`` and its pivot columns and rows."""
    rows = np.array(matrix, dtype=np.int64, copy=True) % prime
    positions = np.arange(rows.shape[0], dtype=np.int64)
    pivot_columns: list[int] = []
    pivot_rows: list[int] = []
    rank = 0
    for column in range(rows.shape[1]):
        if rank == rows.shape[0]:
            break
        candidates = np.flatnonzero(rows[rank:, column])
        if candidates.size == 0:
            continue
        pivot = rank + int(candidates[0])
        rows[[rank, pivot]] = rows[[pivot, rank]]
        positions[[rank, pivot]] = positions[[pivot, rank]]
        rows[rank] = (rows[rank] * pow(int(rows[rank, column]), prime - 2, prime)) % prime
        below = rows[rank + 1 :]
        if below.shape[0]:
            factors = below[:, column].reshape(-1, 1)
            rows[rank + 1 :] = (below - factors * rows[rank]) % prime
        pivot_columns.append(column)
        pivot_rows.append(int(positions[rank]))
        rank += 1
    return (
        rows[:rank],
        np.asarray(pivot_columns, dtype=np.int64),
        np.asarray(pivot_rows, dtype=np.int64),
    )


def modular_rank(matrix: np.ndarray, prime: int) -> int:
    """Return the rank of an integer matrix modulo ``prime``."""
    if matrix.size == 0:
        return 0
    _, pivot_columns, _ = modular_echelon(matrix, prime)
    return int(pivot_columns.size)


def hadamard_bound(matrix: np.ndarray, size: int) -> int:
    """Return an exact upper bound of any ``size`` by ``size`` minor of ``matrix``.

    Each row norm is bounded by ``isqrt(sum of squares) + 1`` in exact integer arithmetic
    (a floating point norm loses precision for large entries, which would make the bound
    unsound), and the bound of a minor is the product of its row norms.
    """
    array = np.asarray(matrix, dtype=np.int64)
    if size <= 0 or array.ndim != 2 or not array.size:
        return 1
    squared = (array.astype(object) ** 2).sum(axis=1)
    norms = sorted((isqrt(int(value)) + 1 for value in squared), reverse=True)
    bound = 1
    for value in norms[:size]:
        bound *= value
    return bound * size


def certified_rank(rows: list[list[int]] | np.ndarray) -> int:
    """Return the exact rank of an integer matrix, proving it with modular arithmetic.

    A prime can never report more than the exact rank, so the running maximum only grows
    while it is still below it: a genuinely deficient matrix keeps its low rank no matter
    how many primes are added, while for a full-rank matrix one prime is enough.  The
    deficient case terminates because the primes keep accumulating until their product
    exceeds the Hadamard bound of the largest minors, at which point at least one of them
    must attain the exact rank.
    """
    array = np.asarray(rows, dtype=np.int64)
    if array.ndim != 2 or not array.size:
        return 0
    width = int(min(array.shape))
    bound = hadamard_bound(array, width)
    # Each prime contributes more than 30 bits, so the certificate cannot need more than
    # this many of them; the check below only guards the bound calculation itself.
    limit = bound.bit_length() // 30 + 2
    best = 0
    product = 1
    for used, prime in enumerate(prime_stream(), start=1):
        if used > limit:
            raise RuntimeError("rank certificate did not terminate below the Hadamard bound")
        best = max(best, modular_rank(array, prime))
        if best == width:
            return width
        product *= prime
        if product > bound:
            return best


def prime_stream() -> Iterator[int]:
    """Yield distinct primes below $2^{31}$ in descending order, lazily."""
    yield from RANK_PRIMES
    candidate = RANK_PRIMES[-1] - 2
    while candidate > 1:
        if _is_prime(candidate):
            yield candidate
        candidate -= 2


def _is_prime(value: int) -> bool:
    """Return whether ``value`` is prime, by trial division and Miller-Rabin."""
    if value < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % small == 0:
            return value == small
    shift = (value - 1) & -(value - 1)
    odd = (value - 1) // shift
    for base in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        power = pow(base, odd, value)
        if power in (1, value - 1):
            continue
        for _ in range(shift.bit_length() - 1):
            power = power * power % value
            if power == value - 1:
                break
        else:
            return False
    return True


__all__ = [
    "RANK_PRIMES",
    "certified_rank",
    "hadamard_bound",
    "modular_echelon",
    "modular_rank",
    "prime_stream",
]
