from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs.tools import build_supercell


def _primitive() -> Atoms:
    return Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )


def test_tool_uses_phonopy_order_and_returns_plain_ase_atoms():
    reference = build_supercell(_primitive(), (2, 1, 1))
    assert isinstance(reference, Atoms)
    np.testing.assert_array_equal(reference.numbers, [11, 11, 17, 17])
    np.testing.assert_allclose(
        reference.get_scaled_positions(),
        [[0, 0, 0], [0.5, 0, 0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
    )


def test_tool_keeps_thirdorder_order_only_when_requested():
    reference = build_supercell(_primitive(), (2, 1, 1), ordering="thirdorder")
    np.testing.assert_array_equal(reference.numbers, [11, 17, 11, 17])


def test_tool_does_not_import_mlfcs_core():
    path = Path(__file__).parents[1] / "src/mlfcs/tools/supercell.py"
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(value.startswith("mlfcs.core") for value in imports)
