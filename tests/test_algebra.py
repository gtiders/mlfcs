"""Matrix operations, rational reconstruction and lattice kernel bases."""

from pathlib import Path

import numpy as np
import pytest

from mlfcs._arrays import as_int64_array
from mlfcs.algebra._congruence import congruence_preimage
from mlfcs.algebra._modular import common_denominator, reconstruct
from mlfcs.algebra.linear import RANK_PRIMES, kernel_basis, rank, verify_kernel
from mlfcs.algebra.matrix import matmul


def _hnf(values):
    """Return the column HNF of an (n, d) integer basis using SymPy.

    The returned SymPy matrix canonically represents the generated lattice,
    allowing comparison of differently oriented generators. Missing SymPy
    skips the calling reference test.
    """
    sympy = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import hermite_normal_form

    return hermite_normal_form(sympy.Matrix(values.tolist()))


def _reference_kernel(matrix):
    """Return a saturated column basis from the independent bigint oracle.

    ``matrix`` is an (m, n) integer array. The result has shape (n, n-rank).
    Missing SymPy skips the calling reference test.
    """
    pytest.importorskip("sympy")
    from oracles.exact_reference import saturated_kernel

    return saturated_kernel(matrix)


def _random_cases():
    """Build reproducible small integer constraints with readable pytest IDs.

    Four fixed cases cover mixed coefficients and signed incidence. Seed 2718
    supplies eight matrices for each width from two through seven, with entries
    in [-8, 8]. Return pytest parameters containing each matrix.
    """
    fixed = ([[2, 1, 1]], [[1, -2, 1], [2, 1, 1]], [[1, -1, 0], [0, 1, 1]], [[1, 1], [1, -1]])
    cases = [
        pytest.param(np.array(a, dtype=np.int64), id=f"fixed-{i}") for i, a in enumerate(fixed)
    ]
    rng = np.random.default_rng(2718)
    for width in range(2, 8):
        for sample in range(8):
            matrix = rng.integers(-8, 9, (max(1, width - 2), width), dtype=np.int64)
            cases.append(pytest.param(matrix, id=f"width-{width}-sample-{sample}"))
    return cases


def _record_cases():
    """Return indices and readable IDs for all recorded FC2 through FC6 constraints."""
    with np.load(Path(__file__).parent / "data" / "kernel_constraints.npz") as records:
        return [
            pytest.param(index, id=f"FC{int(order)}-{name}")
            for index, (name, order) in enumerate(
                zip(records["names"], records["orders"], strict=True)
            )
        ]


@pytest.fixture(scope="module")
def constraint_records():
    """Open the recorded matrices and rational ranks for the duration of this module."""
    with np.load(Path(__file__).parent / "data" / "kernel_constraints.npz") as records:
        yield records


@pytest.mark.reference
@pytest.mark.parametrize("delta", (1, 2, 4, 6, 12, 35, 101, 680581440))
@pytest.mark.parametrize("dimension", (1, 2, 4, 6))
def test_congruence_preimage_matches_the_solution_lattice(delta, dimension):
    """Compare the composite-modulus preimage with an independent column HNF."""
    f = np.random.default_rng(42).integers(-30, 31, size=(3, dimension), dtype=np.int64)
    h = congruence_preimage(f, delta)
    augmented = np.concatenate((f, -delta * np.eye(3, dtype=np.int64)), axis=1)
    reference = _reference_kernel(augmented)[:dimension]
    assert _hnf(h) == _hnf(reference)
    assert np.all(np.diag(h) > 0)
    assert np.all(delta % np.diag(h) == 0)
    assert np.all(np.tril(h, -1) == 0)
    for row in range(dimension):
        assert np.all((h[row, row + 1 :] >= 0) & (h[row, row + 1 :] < h[row, row]))


@pytest.mark.reference
@pytest.mark.parametrize("matrix", _random_cases())
def test_kernel_basis_matches_oracle_lattice(matrix):
    """Compare the full integer solution lattice against the independent oracle."""
    basis = kernel_basis(matrix)
    assert _hnf(basis) == _hnf(_reference_kernel(matrix))
    assert matrix.shape[1] - basis.shape[1] == rank(matrix)
    assert basis.dtype == np.int64
    assert not basis.flags.writeable


@pytest.mark.parametrize(
    "shape", ((0, 0), (0, 5), (4, 0), (3, 5)), ids=("empty", "no-rows", "no-columns", "zero")
)
def test_zero_constraints_return_the_identity_basis(shape):
    """Verify that an unconstrained n-dimensional lattice has the identity basis."""
    matrix = np.zeros(shape, dtype=np.int64)
    np.testing.assert_array_equal(kernel_basis(matrix), np.eye(shape[1], dtype=np.int64))


@pytest.mark.reference
@pytest.mark.parametrize("record_index", _record_cases())
def test_recorded_kernel_has_the_expected_rank_and_saturation(record_index, constraint_records):
    """Check each recorded kernel's dimension, annihilation and Smith invariant factors."""
    sympy = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import smith_normal_form

    matrix = constraint_records[f"a{record_index:03}"]
    basis = kernel_basis(matrix)
    assert matrix.shape[1] - basis.shape[1] == constraint_records["ranks"][record_index]
    assert sympy.Matrix(matrix.tolist()) * sympy.Matrix(basis.tolist()) == sympy.Matrix.zeros(
        matrix.shape[0], basis.shape[1]
    )
    diagonal = smith_normal_form(sympy.Matrix(basis.tolist()), domain=sympy.ZZ)
    assert all(abs(diagonal[i, i]) == 1 for i in range(basis.shape[1]))


@pytest.mark.parametrize(
    "denominator_bound", (1, 2, 65537, 1 << 30), ids=("unit", "two", "65537", "large")
)
@pytest.mark.parametrize("sign", (-1, 1), ids=("negative", "positive"))
def test_rational_reconstruction_recovers_fractions(denominator_bound, sign):
    """Recover a reduced rational number from the two configured prime residues."""
    import math

    modulus = math.prod(RANK_PRIMES)
    numerator = (modulus - 1) // (2 * denominator_bound) - 17
    denominator = denominator_bound
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    residues = [
        np.array([[sign * numerator * pow(denominator, -1, prime) % prime]], dtype=np.int64)
        for prime in RANK_PRIMES
    ]
    n, d, ok = reconstruct(*residues, denominator_bound)
    assert ok
    assert (int(n[0, 0]), int(d[0, 0])) == (sign * numerator, denominator)


def test_common_denominator_preserves_fractions():
    """Represent 1/2 and -1/3 using a shared denominator of six."""
    f, delta, ok = common_denominator(
        np.array([[1, -1]], dtype=np.int64), np.array([[2, 3]], dtype=np.int64)
    )
    assert ok
    assert delta == 6
    np.testing.assert_array_equal(f, [[3, -2]])


def test_array_conversion_preserves_caller_storage():
    """Normalize strided input without changing its writability or retaining mutable aliases."""
    source = np.arange(8, dtype=np.int64)[::2]
    result = as_int64_array(source)
    assert source.flags.writeable
    assert result.flags.c_contiguous and (not result.flags.writeable)
    source[0] = 17
    np.testing.assert_array_equal(result, [0, 2, 4, 6])


def test_matmul_handles_zero_terms_and_large_operands():
    """Evaluate products with large operands whose actual nonzero terms remain small."""
    np.testing.assert_array_equal(matmul([[2**40, 0]], [[0], [2**40]]), [[0]])
    np.testing.assert_array_equal(matmul([[2, 3]], [[4], [5]]), [[23]])


def test_kernel_basis_matches_rational_nullity():
    """Verify the one-dimensional solution lattice of a hand-ranked 2 by 3 matrix."""
    matrix = np.array([[1, -2, 1], [2, 1, 1]], dtype=np.int64)
    basis = kernel_basis(matrix)
    assert rank(matrix) == 2
    assert basis.shape == (3, 1)
    np.testing.assert_array_equal(matrix @ basis, np.zeros((2, 1), dtype=np.int64))
    verify_kernel(matrix, basis)
