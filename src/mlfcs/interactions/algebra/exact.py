r"""Exact integer rank kernels for symmetry algebra.

A rank modulo a prime bounds the exact rank from below, because a rank deficiency over
$\mathbb{Q}$ that survives reduction modulo $p$ is a deficiency modulo $p$ as well.  A
full-rank verdict is therefore a proof, while a deficient verdict is only a suspicion and
is settled by fraction-free elimination over the integers.
"""

from __future__ import annotations

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


def integer_rank(rows: list[list[int]] | np.ndarray) -> int:
    """Return the exact rank of an integer matrix by fraction-free elimination."""
    matrix = [[int(value) for value in row] for row in np.atleast_2d(rows)]
    if not matrix or not matrix[0]:
        return 0
    n_rows = len(matrix)
    n_columns = len(matrix[0])
    rank = 0
    for column in range(n_columns):
        pivot_row = next((row for row in range(rank, n_rows) if matrix[row][column]), None)
        if pivot_row is None:
            continue
        matrix[rank], matrix[pivot_row] = matrix[pivot_row], matrix[rank]
        pivot = matrix[rank][column]
        for row in range(rank + 1, n_rows):
            value = matrix[row][column]
            if value:
                matrix[row] = [
                    pivot * entry - value * reference
                    for entry, reference in zip(matrix[row], matrix[rank], strict=True)
                ]
        rank += 1
        if rank == n_rows:
            break
    return rank


def certified_rank(rows: list[list[int]] | np.ndarray, prime: int | None = None) -> int:
    """Return the exact rank, proving a full-rank verdict modulo a large prime."""
    array = np.asarray(rows, dtype=np.int64)
    width = array.shape[1] if array.ndim == 2 else 0
    for candidate in RANK_PRIMES if prime is None else (prime,):
        if width and modular_rank(array, candidate) == width:
            return width
    return integer_rank(array)


__all__ = ["RANK_PRIMES", "certified_rank", "integer_rank", "modular_echelon", "modular_rank"]
