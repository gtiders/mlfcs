"""Composite congruence preimage and saturated integer lift."""

import numpy as np
from numba import njit

from mlfcs.algebra.integer import bezout


@njit(cache=True)
def reduce_column(h, column, delta):
    for i in range(column - 1, -1, -1):
        q = h[i, column] // h[i, i]
        h[i, column] %= h[i, i]
        for k in range(i):
            h[k, column] = (h[k, column] - q * h[k, i]) % delta


@njit(cache=True)
def row_preimage(w, delta):
    """Column HNF of {z: w*z == 0 mod delta}, including nonunit pivots.

    Prefix gcd g_j gives pivot g_{j-1}/g_j. The coefficient vector b
    represents g_{j-1} modulo delta; all coordinates are reduced immediately.
    Reduction modulo delta is valid because delta*Z**d lies in the preimage.
    """
    d = len(w)
    h = np.zeros((d, d), dtype=np.int64)
    b = np.zeros(d, dtype=np.int64)
    previous = delta
    for j in range(d):
        current, s, t = bezout(previous, w[j])
        pivot = previous // current
        target = (-w[j] * pivot) % delta
        multiple = target // previous
        h[j, j] = pivot
        for i in range(j):
            h[i, j] = b[i] * multiple % delta
        reduce_column(h, j, delta)
        for i in range(j):
            b[i] = (s * b[i]) % delta
        b[j] = t % delta
        previous = current
    return h


@njit(cache=True)
def congruence_preimage(f, delta):
    """Specialized Howell-style lift, with no characteristic-zero transform.

    Each row intersects H*Z**d with a single congruence. If T is its
    preimage HNF, H*T has diagonal <= delta because the intersection still
    contains delta*Z**d. Off-diagonals are multiplied modulo delta, then
    reduced by preceding columns. Every product is bounded by delta**2.
    """
    d = f.shape[1]
    h = np.eye(d, dtype=np.int64)
    if delta == 1:
        return h
    for row in range(f.shape[0]):
        w = np.zeros(d, dtype=np.int64)
        for j in range(d):
            for k in range(j + 1):
                w[j] = (w[j] + (f[row, k] % delta) * (h[k, j] % delta)) % delta
        if not np.any(w):
            continue
        t = row_preimage(w, delta)
        next_h = np.zeros((d, d), dtype=np.int64)
        for j in range(d):
            next_h[j, j] = h[j, j] * t[j, j]
            for i in range(j):
                for k in range(i, j + 1):
                    next_h[i, j] = (next_h[i, j] + (h[i, k] % delta) * (t[k, j] % delta)) % delta
            reduce_column(next_h, j, delta)
        h = next_h
    return h


@njit(cache=True)
def lift_kernel(f, delta, h):
    rank, d = f.shape
    result = np.zeros((rank + d, d), dtype=np.int64)
    result[rank:, :] = h
    for i in range(rank):
        for j in range(d):
            quotient, remainder = 0, 0
            for k in range(d):
                q, r = f[i, k] // delta, f[i, k] % delta
                product = r * h[k, j]
                quotient += q * h[k, j] + product // delta
                remainder += product % delta
                quotient += remainder // delta
                remainder %= delta
            if remainder:
                raise ValueError("congruence preimage lift is not integral")
            result[i, j] = -quotient
    return result
