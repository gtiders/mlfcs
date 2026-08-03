import numpy as np
import pytest
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.io.shengbte import write_shengbte


def test_third_order_direction_and_block_order(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = np.arange(27, dtype=float).reshape((1, 1, 1, 3, 3, 3))
    output = tmp_path / "FORCE_CONSTANTS_3RD"
    write_shengbte(output, values, supercell, cutoff=1.0)
    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    assert lines[2] == "    1"
    assert lines[5] == "     1      1      1"
    assert lines[6] == " 1  1  1     0.0000000000e+00"
    assert lines[-1] == " 3  3  3     2.6000000000e+01"


def test_fourth_order_direction_and_block_order(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = np.arange(81, dtype=float).reshape((1, 1, 1, 1, 3, 3, 3, 3))
    output = tmp_path / "FORCE_CONSTANTS_4TH"
    write_shengbte(output, values, supercell, cutoff=1.0)
    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    assert lines[2] == "    1"
    assert lines[6] == "     1      1      1      1"
    assert lines[7] == " 1  1  1  1     0.0000000000e+00"
    assert lines[-1] == " 3  3  3  3     8.0000000000e+01"


def test_writer_rejects_orders_outside_shengbte_contract(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = np.arange(3**5, dtype=float).reshape((1,) * 5 + (3,) * 5)
    output = tmp_path / "FORCE_CONSTANTS_5TH"
    with pytest.raises(ValueError, match="third- and fourth-order"):
        write_shengbte(output, values, supercell, cutoff=1.0)
