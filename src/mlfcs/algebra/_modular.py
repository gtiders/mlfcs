"""Bounded modular echelon, pivot charts and rational reconstruction."""

import numpy as np
from numba import njit

from mlfcs.algebra.integer import gcd

RANK_PRIMES = (2147483647, 2147483629)
CRT_MODULUS = RANK_PRIMES[0] * RANK_PRIMES[1]
CRT_INVERSE = pow(RANK_PRIMES[0], -1, RANK_PRIMES[1])
MODULAR_LIMIT = (1 << 31) - 1
INT64_LIMIT = (1 << 63) - 1


@njit(cache=True)
def modular_power(base, exponent, modulus):
    result = 1
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


@njit(cache=True)
def echelon(matrix, prime):
    a = matrix % prime
    height, width = a.shape
    order = np.arange(height, dtype=np.int64)
    columns = np.empty(min(height, width), dtype=np.int64)
    rows = np.empty(min(height, width), dtype=np.int64)
    rank = 0
    for column in range(width):
        if rank == height:
            break
        picked = -1
        for i in range(rank, height):
            if a[i, column]:
                picked = i
                break
        if picked < 0:
            continue
        for j in range(width):
            a[rank, j], a[picked, j] = a[picked, j], a[rank, j]
        order[rank], order[picked] = order[picked], order[rank]
        inverse = modular_power(a[rank, column], prime - 2, prime)
        for j in range(column, width):
            a[rank, j] = a[rank, j] * inverse % prime
        columns[rank], rows[rank] = column, order[rank]
        for i in range(rank + 1, height):
            factor = a[i, column]
            for j in range(column, width):
                a[i, j] = (a[i, j] - factor * a[rank, j]) % prime
        rank += 1
    return a, columns[:rank], rows[:rank]


@njit(cache=True)
def chart(matrix, rank, prime):
    a, pivots, _ = echelon(matrix, prime)
    if len(pivots) != rank:
        return a[:0, rank:], False
    for i in range(rank):
        if pivots[i] != i:
            return a[:0, rank:], False
    for i in range(rank - 1, -1, -1):
        for j in range(i):
            factor = a[j, i]
            for k in range(i, a.shape[1]):
                a[j, k] = (a[j, k] - factor * a[i, k]) % prime
    return a[:rank, rank:].copy(), True


@njit(cache=True)
def reconstruct(first, second, denominator_bound):
    p1, p2 = RANK_PRIMES
    modulus = CRT_MODULUS
    inverse = CRT_INVERSE
    numerator_bound = (modulus - 1) // (2 * denominator_bound)
    numerators = np.empty(first.shape, dtype=np.int64)
    denominators = np.empty(first.shape, dtype=np.int64)
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            step = ((second[i, j] - first[i, j]) % p2) * inverse % p2
            residue = first[i, j] + p1 * step
            r0, r1, t0, t1 = modulus, residue, 0, 1
            # Alternating coefficient signs imply |q*t1| <= |t_next| <= M.
            # This bounds the coefficient product independently of remainders.
            while r1 > numerator_bound:
                q = r0 // r1
                r0, r1 = r1, r0 - q * r1
                t0, t1 = t1, t0 - q * t1
            if t1 < 0:
                r1, t1 = -r1, -t1
            if not 0 < t1 <= denominator_bound or gcd(abs(r1), t1) != 1:
                return numerators, denominators, False
            if gcd(t1, modulus) != 1:
                return numerators, denominators, False
            numerators[i, j], denominators[i, j] = r1, t1
    return numerators, denominators, True


@njit(cache=True)
def common_denominator(numerators, denominators):
    delta = 1
    for value in denominators.flat:
        u = delta // gcd(delta, value)
        if u > MODULAR_LIMIT // value:
            return numerators, 0, False
        delta = u * value
    f = np.empty(numerators.shape, dtype=np.int64)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            scale = delta // denominators[i, j]
            if abs(numerators[i, j]) > INT64_LIMIT // scale:
                return f, delta, False
            f[i, j] = numerators[i, j] * scale
    return f, delta, True


@njit(cache=True)
def product_is_zero_mod(a, b, prime):
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            total = 0
            for k in range(a.shape[1]):
                total = (total + (a[i, k] % prime) * (b[k, j] % prime)) % prime
            if total:
                return False
    return True
