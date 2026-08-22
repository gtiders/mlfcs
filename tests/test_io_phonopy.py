from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from supercell_helpers import make_supercell

from mlfcs import write_force_constants
from mlfcs.force_constants.data import ForceConstants
from mlfcs.io.phonopy import write_phonopy


def test_phonopy_writer_expands_fc2_and_matches_reference_format(tmp_path):
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    compact = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3) / 7
    target = tmp_path / "FORCE_CONSTANTS_2ND"
    write_phonopy(target, compact, supercell)

    lines = target.read_text().splitlines()
    assert lines[0] == "   4    4"
    assert len(lines) == 1 + 4 * 4 * 4

    # The file retains phonopy's primitive-site-major reference order. For
    # the second reference atom as anchor and the third as tail, translational
    # anchoring selects primitive site 0 and reference atom 4.
    block = 1 + (1 * 4 + 2) * 4
    assert lines[block] == "2 3"
    expected = compact[0, 3]
    parsed = np.array(
        [[float(value) for value in lines[block + row + 1].split()] for row in range(3)]
    )
    np.testing.assert_allclose(parsed, expected, atol=1e-14)


def test_force_constants_phonopy_format_requires_order_two(tmp_path):
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    supercell, _ = make_supercell(primitive, (1, 1, 1))
    fc = ForceConstants({3: np.zeros((1, 1, 1, 3, 3, 3))}, supercell)
    with pytest.raises(ValueError, match="only for order 2"):
        write_force_constants(fc, tmp_path / "FORCE_CONSTANTS_2ND", format="phonopy")
