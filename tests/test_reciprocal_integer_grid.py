"""Exact integer identities of the reciprocal quotient grid.

The grid's labels are integers, and every acceptance test in this module is a congruence:
``l S^T = 0 (mod D)``.  A floating comparison would accept a grid that is not a lattice
quotient, and an int64 product that wraps would accept one that is not either, so both are
pinned here -- including matrices whose naive accumulation overflows and a shear large enough
that only Gamma survives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mlfcs.reciprocal.grid import (
    exact_modular_product,
    irreducible_reciprocal_grid,
    reciprocal_quotient_grid,
)
from mlfcs.structure.integer_lattice import exact_integer_product
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

MATRICES = {
    "diagonal": np.diag((2, 3, 4)),
    "negative-determinant": np.diag((-2, 3, 4)),
    "non-diagonal": np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64),
    "adjacent-shear": np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 2]], dtype=np.int64),
    "crossed-shear": np.asarray([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int64),
    "anisotropic": np.diag((1, 2, 8)),
}


@pytest.mark.parametrize("name", sorted(MATRICES))
def test_every_label_satisfies_the_integer_congruence(name: str) -> None:
    """Labels are certified by an integer congruence, not by a floating comparison."""
    matrix = MATRICES[name]
    grid = reciprocal_quotient_grid(matrix)
    denominator = grid.denominator
    assert denominator == abs(round(float(np.linalg.det(matrix))))
    assert len(grid.labels) == denominator
    assert np.all(exact_modular_product(grid.labels, matrix.T, denominator) == 0)
    # The floating coordinates are only a numerical view of the same labels.
    np.testing.assert_array_equal(grid.points, grid.labels / denominator)


def test_a_unit_determinant_shear_of_ten_to_the_twelfth_leaves_only_gamma() -> None:
    """A huge shear still has a well-defined quotient: the grid is exactly Gamma."""
    matrix = np.asarray([[1, 10**12, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    assert abs(round(float(np.linalg.det(matrix)))) == 1
    grid = reciprocal_quotient_grid(matrix)
    assert grid.denominator == 1
    np.testing.assert_array_equal(grid.labels, np.zeros((1, 3), dtype=np.int64))
    np.testing.assert_array_equal(grid.points, np.zeros((1, 3)))
    symmetry = PrimitiveSymmetryOperations.from_atoms(_cubic(), symprec=1e-5)
    decomposition = irreducible_reciprocal_grid(matrix, symmetry)
    assert decomposition.n_qpoints == 1 and decomposition.n_irreducible == 1


def test_an_overflow_prone_product_is_certified_exactly() -> None:
    """A product that wraps int64 is still certified exactly, and so is its residue."""
    big = 2**40
    left = np.diag((big, big, big)).astype(np.int64)
    right = np.diag((big, big, big)).astype(np.int64)
    naive = left @ right
    assert int(naive[0, 0]) == 0, "this product must wrap int64 for the test to mean anything"

    exact = exact_integer_product(left, right)
    assert int(exact[0, 0]) == big * big
    modulus = big * big - 1
    assert int(exact_modular_product(left, right, modulus)[0, 0]) == 1


def test_negative_determinant_supercells_use_the_absolute_determinant() -> None:
    """A mirrored supercell has the same quotient size, certified by the same congruence."""
    mirrored = np.diag((-2, 3, 4))
    grid = reciprocal_quotient_grid(mirrored)
    assert grid.denominator == 24 and len(grid.labels) == 24
    assert np.all(exact_modular_product(grid.labels, mirrored.T, grid.denominator) == 0)


def test_the_grid_module_has_no_floating_identity_checks() -> None:
    """The AST gate of the plan: no allclose/isclose anywhere in the grid module."""
    source = Path("src/mlfcs/reciprocal/grid.py").read_text()
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"allclose", "isclose"}:
            offenders.append((node.attr, node.lineno))
        if isinstance(node, ast.Name) and node.id in {"allclose", "isclose"}:
            offenders.append((node.id, node.lineno))
    assert not offenders, offenders


def _cubic():
    from ase import Atoms

    return Atoms("Ar", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
