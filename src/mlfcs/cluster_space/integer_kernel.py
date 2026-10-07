"""Construct saturated integer kernels for exact symmetry constraints.

For an integer constraint matrix ``A``, this module computes a basis of the
full solution lattice

    ker_Z(A) = {x in Z^n : A x = 0}.

Signed incidence-like systems are handled directly through equality components.
General systems are solved through modular rank detection, rational chart
reconstruction, and recovery of the full integer preimage lattice, avoiding
floating-point rank decisions.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mlfcs.foundation.arrays import require_allocation
from mlfcs.foundation.errors import RankError
from mlfcs.foundation.integer import RANK_PRIMES, as_int64_matrix, echelon, prime_stream

CRT_MODULUS = RANK_PRIMES[0] * RANK_PRIMES[1]
CRT_INVERSE = pow(RANK_PRIMES[0], -1, RANK_PRIMES[1])
INT64_LIMIT = (1 << 63) - 1
MODULAR_LIMIT = (1 << 31) - 1


@njit(cache=True)
def signed_components(a):
    """Analyze signed equality constraints using connected components.

    This fast path applies to integer systems whose nonzero rows contain at
    most two entries. A row with one nonzero coefficient imposes ``x_i = 0``,
    while a row with two coefficients of equal magnitude imposes
    ``x_i = +/- x_j``. These relations are represented by a signed
    disjoint-set structure. A component is marked dead when it is forced to
    zero, either directly or through an inconsistent sign cycle.

    Parameters
    ----------
    a
        Integer constraint matrix of shape ``(m, n)``.

    Returns
    -------
    parent
        Disjoint-set parent array of shape ``(n,)``.
    signs
        Relative signs to parent nodes, with values in ``{-1, +1}``.
    dead
        Boolean-like flags marking components forced to zero.
    columns
        Mapping from live component roots to kernel-basis columns.
    dimension
        Number of live signed components. ``-1`` indicates that the matrix
        does not satisfy the signed-equality structure required by this path.
    """
    n = a.shape[1]
    parent = np.arange(n, dtype=np.int64)
    signs = np.ones(n, dtype=np.int64)
    dead = np.zeros(n, dtype=np.uint8)
    for i in range(a.shape[0]):
        first, second, count = -1, -1, 0
        for j in range(n):
            if a[i, j]:
                if count == 0:
                    first = j
                elif count == 1:
                    second = j
                count += 1
        if count > 2:
            return parent, signs, dead, np.empty(0, dtype=np.int64), -1
        if count == 0:
            continue
        x, sx = first, 1
        while parent[x] != x:
            sx *= signs[x]
            x = parent[x]
        if count == 1:
            dead[x] = 1
            continue
        if abs(a[i, first]) != abs(a[i, second]):
            return parent, signs, dead, np.empty(0, dtype=np.int64), -1
        relation = -1 if a[i, first] == a[i, second] else 1
        y, sy = second, 1
        while parent[y] != y:
            sy *= signs[y]
            y = parent[y]
        if x == y:
            if sx != relation * sy:
                dead[x] = 1
        else:
            parent[y] = x
            signs[y] = relation * sx * sy
            dead[x] = max(dead[x], dead[y])
    columns = np.full(n, -1, dtype=np.int64)
    dimension = 0
    for j in range(n):
        root = j
        while parent[root] != root:
            root = parent[root]
        if not dead[root] and columns[root] < 0:
            columns[root] = dimension
            dimension += 1
    return parent, signs, dead, columns, dimension


@njit(cache=True)
def component_basis(parent, signs, dead, columns, dimension):
    """Construct the integer kernel basis of a signed-component system.

    Each live connected component contributes one independent basis vector.
    Entries within the same component differ only by their stored relative
    sign, while dead components contribute zero. The returned ``(n,
    dimension)`` matrix has entries in ``{-1, 0, +1}`` and generates the full
    integer solution lattice of the signed-component constraints. The caller
    admits output capacity and supplies a successful ``signed_components``
    result.
    """
    n = len(parent)
    result = np.zeros((n, dimension), dtype=np.int64)
    for j in range(n):
        root, sign = j, 1
        while parent[root] != root:
            sign *= signs[root]
            root = parent[root]
        if not dead[root]:
            result[j, columns[root]] = sign
    return result


@njit(cache=True)
def gcd(a, b):
    """Return the nonnegative greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def chart(matrix, rank, prime):
    """Construct the free-column block of a fixed modular pivot chart.

    The input columns are ordered so that the first ``rank`` columns are the
    prescribed pivots. Over the finite field ``F_prime``, successful reduction
    gives ``[I | C]``, where ``C`` has shape ``(rank, n-rank)``. Return
    ``(C, True)`` on success; ``(empty, False)`` if the prescribed pivot
    columns do not realize the requested rank modulo ``prime``.
    """
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
    """Reconstruct rational chart entries from residues at two primes.

    Residues modulo the two fixed reconstruction primes are first combined by
    the Chinese remainder theorem. Each resulting residue is then interpreted
    as a reduced rational number ``a / b`` satisfying the prescribed
    denominator bound and the corresponding uniqueness bound on ``|a|``.

    Parameters
    ----------
    first, second
        Residue matrices of identical shape over the two reconstruction primes.
    denominator_bound
        Positive upper bound on reconstructed denominators.

    Returns
    -------
    numerators, denominators
        Reconstructed reduced fractions entrywise.
    ok
        ``False`` if any entry has no unique admissible reconstruction.
    """
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
    """Convert reconstructed rational entries to a common denominator.

    Given entrywise reduced fractions ``N_ij / D_ij``, construct a positive
    common denominator ``delta`` and integer matrix ``F`` such that
    ``N_ij / D_ij = F_ij / delta`` for every entry.

    Returns
    -------
    F
        Integer scaled chart.
    delta
        Positive common denominator.
    ok
        ``False`` if the required denominator or scaled entries exceed the
        supported integer domain.
    """
    delta = 1
    for value in denominators.flat:
        # Divide before multiplying and compare by division: admission must
        # not overflow while constructing the quantity it intends to check.
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
    """Return whether ``a @ b`` is identically zero modulo ``prime``.

    The product is accumulated modulo ``prime`` entry by entry so that no
    characteristic-zero matrix product is formed.
    """
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            total = 0
            for k in range(a.shape[1]):
                total = (total + (a[i, k] % prime) * (b[k, j] % prime)) % prime
            if total:
                return False
    return True


@njit(cache=True)
def bezout(a, b):
    """Return ``(g, s, t)`` satisfying ``s*a + t*b = g = gcd(a, b)``.

    Callers provide ``0 < a < 2**31`` and ``0 <= b < 2**31``; the Euclidean
    coefficient updates stay within int64 for these inputs.
    """
    r0, r1, s0, s1, t0, t1 = a, b, 1, 0, 0, 1
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    return r0, s0, t0


@njit(cache=True)
def reduce_column(h, column, delta):
    """Reduce one triangular lattice-basis column to column-HNF residue form.

    Earlier pivot columns are assumed already reduced. The selected column is
    modified in place so that every entry above pivot ``i`` lies in
    ``0 <= h[i, column] < h[i, i]``. The lattice generated by the columns
    contains ``delta * Z**d``, so reduction modulo ``delta`` preserves it.
    """
    for i in range(column - 1, -1, -1):
        q = h[i, column] // h[i, i]
        h[i, column] %= h[i, i]
        for k in range(i):
            h[k, column] = (h[k, column] - q * h[k, i]) % delta


@njit(cache=True)
def row_preimage(w, delta):
    """Construct a column-HNF basis of one modular kernel preimage.

    The returned columns generate the integer lattice

        {z in Z**d : w @ z == 0 (mod delta)}.

    The basis is upper triangular with positive pivots and reduced
    above-diagonal entries, following column-Hermite-normal-form conventions.

    Parameters
    ----------
    w
        Integer row vector of length ``d``, interpreted modulo ``delta``.
    delta
        Positive modulus.

    Returns
    -------
    ndarray
        Integer ``(d, d)`` basis whose columns generate the full congruence
        preimage lattice.
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
    """Construct the integer preimage of a modular linear system.

    Return a column basis ``H`` of

        {z in Z**d : F z == 0 (mod delta)}.

    The lattice is built by intersecting the current basis successively with
    the congruence defined by each row of ``F``. This computes the full
    integer preimage rather than a vector-space kernel over the possibly
    composite ring ``Z/delta Z``. Every intersection still contains
    ``delta * Z**d``, which bounds the column-HNF pivots.

    Parameters
    ----------
    f
        Integer matrix of shape ``(r, d)``.
    delta
        Positive modulus.

    Returns
    -------
    ndarray
        Column-HNF basis ``H`` of shape ``(d, d)`` generating the full
        congruence preimage lattice.
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
    """Lift free-coordinate lattice vectors to exact kernel vectors.

    Suppose the reconstructed rational pivot chart is ``x_pivot = -(F /
    delta) z``, and ``H`` generates all integer free vectors ``z``
    satisfying ``F z == 0 (mod delta)``. Then each column of

        [ -F H / delta ]
        [       H       ]

    is an integer solution of the original chart equations.

    Parameters
    ----------
    f
        Integer numerator matrix of shape ``(rank, d)``.
    delta
        Positive common denominator.
    h
        Integer preimage basis of shape ``(d, d)``.

    Returns
    -------
    ndarray
        Integer basis of shape ``(rank+d, d)`` in pivot-then-free coordinates.
        The caller admits the quotient-accumulator bound before entry.

    Raises
    ------
    ValueError
        If ``F @ H`` is not exactly divisible by ``delta``.
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


def _verify_integer_kernel(matrix, basis):
    """Certify that ``matrix @ basis`` is exactly zero over the integers.

    The product is tested modulo successive primes. If all residues vanish and
    the product of tested primes exceeds a deterministic bound on every exact
    matrix-product entry, the integer identity ``matrix @ basis == 0`` is
    proven without forming the potentially overflowing characteristic-zero
    product. This verifies annihilation only; it does not prove independence
    or saturation of the basis.
    """
    a, b = as_int64_matrix(matrix), as_int64_matrix(basis, name="kernel basis")
    if a.shape[1] != b.shape[0]:
        raise ValueError("kernel basis has incompatible row count")
    maximum_a = max((abs(int(v)) for v in a.flat), default=0)
    maximum_b = max((abs(int(v)) for v in b.flat), default=0)
    bound = a.shape[1] * maximum_a * maximum_b
    product = 1
    for prime in prime_stream():
        if not product_is_zero_mod(a, b, prime):
            raise ValueError("matrix @ basis is nonzero modulo the tested prime")
        product *= prime
        if product > bound:
            return
    raise RankError("prime stream exhausted before verifying matrix @ basis == 0")


def integer_kernel_basis(matrix, *, expected_nullity=None):
    """Return a saturated basis of the integer kernel of a matrix.

    For an integer matrix ``A`` of shape ``(m, n)``, compute an integer matrix
    ``B`` whose columns generate exactly

        ker_Z(A) = {x in Z^n : A x = 0}.

    Thus every integer solution can be written as ``x = B c`` with ``c`` in
    ``Z**d``, where ``d`` is the nullity. The returned lattice is saturated:
    no integer kernel vector is omitted merely because a rational basis was
    cleared of denominators.

    Parameters
    ----------
    matrix
        Integer array of shape ``(m, n)`` with entries in the symmetric
        int64 domain, excluding ``INT64_MIN``.
    expected_nullity
        Optional required kernel dimension. A mismatch is treated as failure
        rather than changing the expected model dimension.

    Returns
    -------
    basis
        Readonly ``int64`` array of shape ``(n, d)`` whose columns generate
        the full integer kernel. Basis orientation is not canonical.

    Raises
    ------
    ValueError
        If the input is invalid or the signed-component fast path has a
        dimension inconsistent with ``expected_nullity``.
    RankError
        If the selected modular pivot chart does not realize the required rank
        at the reconstruction primes.
    OverflowError
        If exact reconstruction or lattice recovery exceeds the supported
        ``int64`` arithmetic domain.

    Notes
    -----
    Signed equality systems are solved directly through connected components.

    General systems are handled by:

    1. detecting a pivot chart over finite fields;
    2. reconstructing the rational free-variable chart using two-prime CRT
       and rational reconstruction;
    3. writing the chart as ``F / delta``;
    4. computing the full congruence preimage
       ``{z in Z**d : F z == 0 (mod delta)}``;
    5. lifting that lattice to exact kernel vectors ``(-F H / delta, H)``;
    6. restoring the original column order and certifying ``A @ B == 0`` by
       modular products.

    The congruence-preimage step is what converts a denominator-cleared
    rational kernel basis into the full saturated integer kernel.
    """
    a = as_int64_matrix(matrix)
    n = a.shape[1]
    require_allocation("signed component workspace", (n,))
    parent, signs, dead, columns, dimension = signed_components(a)
    if dimension >= 0:
        require_allocation("signed kernel basis", (n, dimension))
        signed = component_basis(parent, signs, dead, columns, dimension)
        if expected_nullity is not None and expected_nullity != dimension:
            raise ValueError("expected nullity differs from signed-component dimension")
        signed.setflags(write=False)
        return signed
    choices = [echelon(a, p) for p in RANK_PRIMES]
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
    charts = [chart(selected, rank, p) for p in RANK_PRIMES]
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
        numerators, denominators, ok = reconstruct(charts[0][0], charts[1][0], 1 << exponent)
        if not ok:
            continue
        f, delta, ok = common_denominator(numerators, denominators)
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
            _verify_integer_kernel(a, rational)
        except ValueError:
            continue
        # Clearing denominators alone gives a possibly proper sublattice.
        # The congruence preimage recovers every admissible integer free vector.
        h = congruence_preimage(f, delta)
        lifted = lift_kernel(f, delta, h)
        result = np.empty((n, d), dtype=np.int64)
        result[order] = lifted
        _verify_integer_kernel(a, result)
        result.setflags(write=False)
        return result
    raise OverflowError("kernel chart reconstruction exceeds the supported int64 domain")


__all__ = ["integer_kernel_basis"]
