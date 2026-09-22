from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlfcs.structure.integer_lattice import normalize_supercell_matrix
from mlfcs.tools.supercell import align_structures, build_supercell


def _primitive() -> Atoms:
    return Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


def test_builder_uses_phonopy_order_and_returns_plain_ase_atoms():
    reference = build_supercell(_primitive(), (2, 1, 1))
    assert isinstance(reference, Atoms)
    np.testing.assert_array_equal(reference.numbers, [11, 11, 17, 17])
    np.testing.assert_allclose(
        reference.get_scaled_positions(),
        [[0, 0, 0], [0.5, 0, 0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
    )


@pytest.mark.parametrize("symprec", (0.0, -1.0, np.nan, np.inf))
def test_builder_rejects_a_nonphysical_symprec(symprec):
    with pytest.raises(ValueError, match="finite positive length"):
        build_supercell(_primitive(), (2, 1, 1), symprec=symprec)


def test_builder_has_no_calculation_or_workflow_dependency():
    path = Path(__file__).parents[1] / "src/mlfcs/tools/supercell.py"
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        value.startswith(("mlfcs.fitting", "mlfcs.io", "mlfcs.phonon")) for value in imports
    )


def test_supercell_tool_has_no_phonopy_runtime_dependency():
    path = Path(__file__).parents[1] / "src/mlfcs/tools/supercell.py"
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(value == "phonopy" or value.startswith("phonopy.") for value in imports)


def test_project_builder_matches_phonopy_old_style_for_a_general_matrix():
    pytest.importorskip("phonopy", reason="phonopy is a reference-test oracle")
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    primitive = _primitive()
    primitive.set_masses([22.5, 36.5])
    matrix = normalize_supercell_matrix([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
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
    np.testing.assert_allclose(actual.cell, expected.cell, atol=1e-12)
    np.testing.assert_allclose(actual.get_scaled_positions(), expected.scaled_positions, atol=1e-12)
    np.testing.assert_allclose(actual.get_masses(), expected.masses, atol=1e-12)


def test_alignment_is_co_located_with_supercell_construction():
    assert align_structures.__module__ == "mlfcs.tools.supercell"
    assert not (Path(__file__).parents[1] / "src/mlfcs/tools/structure_alignment.py").exists()
