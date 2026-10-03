"""Differential checks of the admitted exact backend, including lattice index."""

from pathlib import Path

import numpy as np
import pytest

from mlfcs.algebra._congruence import congruence_preimage
from mlfcs.algebra.exact import exact_kernel, exact_rank

pytest.importorskip("sympy")
from oracles.exact_reference import saturated_kernel as reference_kernel
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form


def hnf(values):
    return hermite_normal_form(Matrix(values.tolist()))


@pytest.mark.parametrize("delta", (1, 2, 4, 6, 12, 35, 101, 680581440))
def test_composite_preimage_is_the_full_canonical_integer_lattice(delta):
    rng = np.random.default_rng(42)
    for d in (1, 2, 4, 6):
        f = rng.integers(-30, 31, size=(3, d), dtype=np.int64)
        h = congruence_preimage(f, delta)
        augmented = np.concatenate((f, -delta * np.eye(3, dtype=np.int64)), axis=1)
        reference = reference_kernel(augmented)[:d]
        assert hnf(h) == hnf(reference)
        assert np.all(np.diag(h) > 0)
        assert np.all(delta % np.diag(h) == 0)
        assert np.all(np.tril(h, -1) == 0)
        for i in range(d):
            assert np.all((h[i, i + 1 :] >= 0) & (h[i, i + 1 :] < h[i, i]))


def test_saturated_kernel_matches_bigint_oracle_on_random_constraints():
    rng = np.random.default_rng(2718)
    cases = [
        np.array([[2, 1, 1]]),
        np.array([[1, -2, 1], [2, 1, 1]]),
        np.array([[1, -1, 0], [0, 1, 1]]),
        np.array([[1, 1], [1, -1]]),
    ]
    for n in range(2, 8):
        for _ in range(8):
            cases.append(rng.integers(-8, 9, (max(1, n - 2), n), dtype=np.int64))
    for a in cases:
        kernel = exact_kernel(a)
        reference = reference_kernel(a)
        assert hnf(kernel) == hnf(reference)
        assert a.shape[1] - kernel.shape[1] == exact_rank(a)
        assert kernel.dtype == np.int64
        assert not kernel.flags.writeable


def test_out_of_domain_inputs_are_rejected():
    with pytest.raises(OverflowError):
        exact_kernel([[2**70, 1]])


def test_empty_and_zero_constraints_keep_the_declared_width():
    for shape in ((0, 0), (0, 5), (4, 0), (3, 5)):
        a = np.zeros(shape, dtype=np.int64)
        kernel = exact_kernel(a)
        assert kernel.shape == (shape[1], shape[1])


def test_recorded_fc2_to_fc6_constraints_match_oracle_lattice():
    from sympy import ZZ
    from sympy.matrices.normalforms import smith_normal_form

    with np.load(Path(__file__).parent / "data" / "kernel_constraints.npz") as fixtures:
        assert len(fixtures["names"]) == 165
        assert set(fixtures["orders"]) == {2, 3, 4, 5, 6}
        for index, name in enumerate(fixtures["names"]):
            a = fixtures[f"a{index:03}"]
            kernel = exact_kernel(a)
            assert a.shape[1] - kernel.shape[1] == fixtures["ranks"][index], name
            # Full rank + annihilation + Smith invariant factors all 1 prove
            # equality to the saturated integer kernel without constructing
            # the oracle's potentially enormous unimodular transform.
            assert Matrix(a.tolist()) * Matrix(kernel.tolist()) == Matrix.zeros(
                a.shape[0], kernel.shape[1]
            ), name
            diagonal = smith_normal_form(Matrix(kernel.tolist()), domain=ZZ)
            assert all(abs(diagonal[i, i]) == 1 for i in range(kernel.shape[1])), name


def test_reconstruction_and_scaling_guard_machine_word_intermediates():
    import math

    from mlfcs.algebra._modular import common_denominator, reconstruct
    from mlfcs.algebra.exact import RANK_PRIMES

    modulus = math.prod(RANK_PRIMES)
    for denominator_bound in (1, 2, 65537, 1 << 30):
        numerator = (modulus - 1) // (2 * denominator_bound) - 17
        denominator = denominator_bound
        divisor = math.gcd(numerator, denominator)
        numerator //= divisor
        denominator //= divisor
        for sign in (-1, 1):
            residues = [
                np.array([[sign * numerator * pow(denominator, -1, prime) % prime]], dtype=np.int64)
                for prime in RANK_PRIMES
            ]
            n, d, ok = reconstruct(*residues, denominator_bound)
            assert ok
            assert (int(n[0, 0]), int(d[0, 0])) == (sign * numerator, denominator)
    # LCM is refused before multiplying, even though the individual values fit.
    _, _, ok = common_denominator(
        np.ones((1, 2), dtype=np.int64), np.array([[2147483646, 2147483647]], dtype=np.int64)
    )
    assert not ok
    # Numerator scaling is refused before multiplying, independently of the LCM.
    _, _, ok = common_denominator(
        np.array([[2**63 - 1, 1]], dtype=np.int64), np.array([[1, 2]], dtype=np.int64)
    )
    assert not ok
