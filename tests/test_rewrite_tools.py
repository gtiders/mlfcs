"""Focused contracts for optional structure tools."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms

from mlfcs.tools import build_supercell


def _primitive() -> Atoms:
    return Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


def test_build_supercell_returns_site_major_ase_atoms() -> None:
    result = build_supercell(_primitive(), (2, 1, 1))

    assert isinstance(result, Atoms)
    np.testing.assert_array_equal(result.numbers, [11, 11, 17, 17])
    np.testing.assert_allclose(
        result.get_scaled_positions(),
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
    )
    np.testing.assert_allclose(result.cell, np.diag([8.0, 4.0, 4.0]))


def test_build_supercell_accepts_general_integer_matrix_and_preserves_masses() -> None:
    primitive = _primitive()
    primitive.set_masses([22.5, 36.5])
    matrix = np.asarray([[2, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)

    result = build_supercell(primitive, matrix)

    assert len(result) == 8
    np.testing.assert_allclose(result.cell, matrix @ np.asarray(primitive.cell))
    np.testing.assert_array_equal(result.get_masses(), np.repeat([22.5, 36.5], 4))


@pytest.mark.parametrize(
    "matrix",
    (
        np.diag([2, 1, 1]),
        np.asarray([[2, 1, 0], [0, 2, 0], [0, 0, 1]]),
        np.asarray([[1, 1, 0], [0, 2, 0], [0, 0, 2]]),
    ),
)
def test_build_supercell_matches_phonopy_old_style_atom_order(matrix: np.ndarray) -> None:
    phonopy = pytest.importorskip("phonopy", reason="optional old-style ordering oracle")
    del phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    primitive = _primitive()
    primitive.set_masses([22.5, 36.5])
    expected = get_supercell(
        PhonopyAtoms(
            symbols=primitive.get_chemical_symbols(),
            cell=np.asarray(primitive.cell),
            scaled_positions=primitive.get_scaled_positions(wrap=True),
            masses=primitive.get_masses(),
        ),
        matrix.T,
        is_old_style=True,
        symprec=1e-5,
    )
    actual = build_supercell(primitive, matrix)

    np.testing.assert_array_equal(actual.numbers, expected.numbers)
    np.testing.assert_allclose(actual.cell, expected.cell, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        actual.get_scaled_positions(wrap=False), expected.scaled_positions, atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(actual.get_masses(), expected.masses, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("symprec", (0.0, -1.0, np.nan, np.inf))
def test_build_supercell_requires_physical_symprec(symprec: float) -> None:
    with pytest.raises(ValueError, match="finite positive length"):
        build_supercell(_primitive(), (2, 1, 1), symprec=symprec)


@pytest.mark.parametrize("matrix", ((2.0, 1, 1), ((2, 0, 0), (0, 1.0, 0), (0, 0, 1))))
def test_build_supercell_rejects_float_lattice_claims(matrix: object) -> None:
    with pytest.raises(TypeError, match="declared as integers"):
        build_supercell(_primitive(), matrix)


def test_build_supercell_rejects_non_positive_or_singular_matrices() -> None:
    with pytest.raises(ValueError, match="positive determinant"):
        build_supercell(_primitive(), np.diag([-1, 1, 1]))
    with pytest.raises(ValueError, match="positive determinant"):
        build_supercell(_primitive(), np.diag([1, 1, 0]))


def test_tools_builder_has_no_phonopy_runtime_or_reverse_dependency() -> None:
    root = Path(__file__).parents[1]
    source = (root / "src/mlfcs/tools/supercell.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )

    assert not any(name == "phonopy" or name.startswith("phonopy.") for name in imports)
    assert not any(name.startswith("mlfcs.") for name in imports)
    assert internal_dependencies("tools") == {"core", "force_constants", "supercell"}
    for package in (
        "core",
        "cluster_space",
        "supercell",
        "fitting",
        "finite_difference",
        "force_constants",
    ):
        assert "tools" not in internal_dependencies(package)
