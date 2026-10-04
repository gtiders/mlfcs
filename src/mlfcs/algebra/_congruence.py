"""Composite congruence preimage and saturated integer lift."""

import numpy as np
from numba import njit

from mlfcs.algebra.integer import bezout


@njit(cache=True)
def reduce_column(h, column, delta):
    """Reduce one upper-triangular column in place to column-HNF pivot ranges.

    The h columns generate a lattice containing delta*Z**d, so subtracting
    multiples of delta from entries preserves that lattice. Pivots are positive
    and at most delta; earlier columns are already reduced. With delta below
    2**31, quotient-times-entry updates fit the admitted machine-word bounds.
    """
    for i in range(column - 1, -1, -1):
        q = h[i, column] // h[i, i]
        h[i, column] %= h[i, i]
        for k in range(i):
            h[k, column] = (h[k, column] - q * h[k, i]) % delta


@njit(cache=True)
def row_preimage(w, delta):
    """Return a column-HNF basis of {z in Z**d: w @ z == 0 mod delta}.

    w has d entries reduced to [0, delta), with 1 <= delta < 2**31. Columns
    of the upper-triangular output generate the full congruence preimage.
    Prefix gcds give positive pivots dividing delta; entries above pivot i lie
    in [0, h[i,i]). Thus every stored entry is at most delta.
    Bezout coefficient vectors and off-diagonal products are reduced modulo
    delta. Neither input is modified; no full unimodular transform is tracked.
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
    """Return a column-HNF basis H of {z in Z**d: F @ z == 0 mod delta}.

    F is an int64 (rank, d) matrix and 1 <= delta < 2**31. Intersect the current
    lattice with each row congruence using row_preimage. Columns generate the
    integer preimage, not a vector-space kernel over a possibly composite ring.
    H is upper triangular with positive pivots <= delta and 0 <= H[i,j] <
    H[i,i] above the diagonal. Each intersection still contains delta*Z**d;
    this bounds pivots. Residue products and their one-step accumulations fit
    int64. The caller admits d*d storage; F is preserved.
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
    """Lift free-coordinate column basis H to (-F @ H / delta, H) exactly.

    F has shape (rank, d), H (d, d), with F @ H divisible by positive delta.
    H entries must be nonnegative and <= delta, delta < 2**31, and the caller
    must admit d*(max(abs(F)) + 2*delta + 1) <= INT64_MAX. Split F into quotient
    and remainder before multiplying to avoid forming a large F @ H temporary.
    Return an int64 (rank+d, d) basis in pivot-then-free coordinate order.
    Raise ValueError if an output entry is not integral; inputs are preserved.
    """
    rank, d = f.shape
    result = np.zeros((rank + d, d), dtype=np.int64)
    result[rank:, :] = h
    for i in range(rank):
        for j in range(d):
            # Track carries from reduced products so F @ H is never formed
            # as a potentially oversized characteristic-zero intermediate.
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
