import numpy as np
import pytest
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.io.shengbte import write_shengbte
from mlfcs.model import ForceConstants, SparseOrderForceConstants


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


def test_writer_never_discards_nonzero_tensor_during_geometry_filtering(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (2, 1, 1))
    values = np.zeros((1, 2, 2, 3, 3, 3))
    values[0, 1, 1, 0, 0, 0] = 7.0
    output = tmp_path / "FORCE_CONSTANTS_3RD"

    # The nonzero block lies outside this deliberately short output cutoff.
    # The dense tensor is authoritative and must survive serialization.
    write_shengbte(output, values, supercell, cutoff=1.0)

    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    assert " 1  1  1     7.0000000000e+00" in lines


def test_force_constants_defaults_to_closed_support_and_offers_thirdorder_mode(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (2, 1, 1))
    tensors = np.zeros((1, 3, 3, 3))
    tensors[0, 0, 0, 0] = 7.0
    sparse = SparseOrderForceConstants(
        order=3,
        n_primitive=1,
        n_supercell=2,
        clusters=np.asarray([[0, 1, 1]]),
        tensors=tensors,
    )
    result = ForceConstants(
        {},
        supercell,
        metadata={"cutoff_angstrom": 1.0},
        sparse={3: sparse},
    )

    faithful = tmp_path / "faithful"
    compatible = tmp_path / "compatible"
    result.write(faithful, format="shengbte")
    result.write(compatible, format="shengbte", compatibility="thirdorder")

    assert " 1  1  1     7.0000000000e+00" in faithful.read_text().splitlines()
    assert " 1  1  1     7.0000000000e+00" not in compatible.read_text().splitlines()


def test_shengbte_rejects_unknown_compatibility_mode(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = np.zeros((1, 1, 1, 3, 3, 3))
    with pytest.raises(ValueError, match="None or 'thirdorder'"):
        write_shengbte(
            tmp_path / "FORCE_CONSTANTS_3RD",
            values,
            supercell,
            cutoff=1.0,
            compatibility="unknown",
        )


def test_writer_rejects_orders_outside_shengbte_contract(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = np.arange(3**5, dtype=float).reshape((1,) * 5 + (3,) * 5)
    output = tmp_path / "FORCE_CONSTANTS_5TH"
    with pytest.raises(ValueError, match="third- and fourth-order"):
        write_shengbte(output, values, supercell, cutoff=1.0)
