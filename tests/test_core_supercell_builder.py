from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs import build_supercell
from mlfcs.core.integer_lattice import normalize_supercell_matrix
from mlfcs.core.supercell import _fallback_phonopy_old_style, _from_phonopy


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


def test_builder_has_no_calculation_or_workflow_dependency():
    path = Path(__file__).parents[1] / "src/mlfcs/core/supercell.py"
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(value.startswith(("mlfcs.fitting", "mlfcs.io", "mlfcs.anharmonic")) for value in imports)


def test_fallback_is_identical_to_phonopy_old_style_for_a_general_matrix():
    primitive = _primitive()
    matrix = normalize_supercell_matrix([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    phonopy = _from_phonopy(primitive, matrix, symprec=1e-5)
    fallback = _fallback_phonopy_old_style(primitive, matrix, symprec=1e-5)

    np.testing.assert_array_equal(fallback.numbers, phonopy.numbers)
    np.testing.assert_allclose(fallback.cell, phonopy.cell, atol=1e-12)
    np.testing.assert_allclose(
        fallback.get_scaled_positions(), phonopy.get_scaled_positions(), atol=1e-12
    )
