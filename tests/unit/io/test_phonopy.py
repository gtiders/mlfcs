from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.io.phonopy import write_phonopy
from mlfcs.model import ForceConstants


def test_phonopy_writer_expands_fc2_and_matches_reference_format(tmp_path):
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    supercell, index = make_supercell(primitive, (2, 1, 1))
    compact = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3) / 7
    target = tmp_path / "FORCE_CONSTANTS_2ND"
    write_phonopy(target, compact, supercell)

    lines = target.read_text().splitlines()
    assert lines[0] == "   4    4"
    assert len(lines) == 1 + 4 * 4 * 4

    # grouped order is (Na@0, Na@1, Cl@0, Cl@1). For the Na@1,Cl@0
    # pair the relative image is Cl@1, internal atom index 3.
    block = 1 + (1 * 4 + 2) * 4
    assert lines[block] == "2 3"
    expected = compact[0, 3]
    parsed = np.array(
        [[float(value) for value in lines[block + row + 1].split()] for row in range(3)]
    )
    np.testing.assert_allclose(parsed, expected, atol=1e-14)
    np.testing.assert_array_equal(index.grouped_permutation, [0, 2, 1, 3])


def test_force_constants_phonopy_format_requires_order_two(tmp_path):
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    supercell, _ = make_supercell(primitive, (1, 1, 1))
    fc = ForceConstants({3: np.zeros((1, 1, 1, 3, 3, 3))}, supercell)
    with pytest.raises(ValueError, match="only for order 2"):
        fc.write(tmp_path / "FORCE_CONSTANTS_2ND", format="phonopy")
