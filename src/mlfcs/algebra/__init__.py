"""Shared matrix operations, rational rank and lattice kernel bases."""

from mlfcs.algebra.linear import (
    RANK_PRIMES,
    hadamard_bound,
    kernel_basis,
    prime_stream,
    rank,
    rank_pivots,
    verify_kernel,
)
from mlfcs.algebra.matrix import (
    adjugate_3x3,
    as_int64,
    determinant_3x3,
    matmul,
    to_python_rows,
    unimodular_inverse,
)

__all__ = [
    "RANK_PRIMES",
    "adjugate_3x3",
    "as_int64",
    "determinant_3x3",
    "hadamard_bound",
    "kernel_basis",
    "matmul",
    "prime_stream",
    "rank",
    "rank_pivots",
    "to_python_rows",
    "unimodular_inverse",
    "verify_kernel",
]
