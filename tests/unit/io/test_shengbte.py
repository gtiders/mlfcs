import numpy as np
import pytest
from ase import Atoms

from mlfcs.core.geometry import make_supercell
from mlfcs.io.shengbte import write_shengbte
from mlfcs.model import ForceConstants, SparseOrderForceConstants


def test_third_order_direction_and_block_order(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = SparseOrderForceConstants(
        3, 1, 1, np.asarray([[0, 0, 0]]), np.arange(27, dtype=float).reshape((1, 3, 3, 3))
    )
    output = tmp_path / "FORCE_CONSTANTS_3RD"
    write_shengbte(output, values, supercell)
    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    assert lines[2] == "    1"
    assert lines[5] == "     1      1      1"
    assert lines[6] == " 1  1  1     0.0000000000e+00"
    assert lines[-1] == " 3  3  3     2.6000000000e+01"


def test_fourth_order_direction_and_block_order(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = SparseOrderForceConstants(
        4,
        1,
        1,
        np.asarray([[0, 0, 0, 0]]),
        np.arange(81, dtype=float).reshape((1, 3, 3, 3, 3)),
    )
    output = tmp_path / "FORCE_CONSTANTS_4TH"
    write_shengbte(output, values, supercell)
    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    assert lines[2] == "    1"
    assert lines[6] == "     1      1      1      1"
    assert lines[7] == " 1  1  1  1     0.0000000000e+00"
    assert lines[-1] == " 3  3  3  3     8.0000000000e+01"


def test_writer_serializes_sparse_support_without_geometry_filtering(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (2, 1, 1))
    tensors = np.zeros((1, 3, 3, 3))
    tensors[0, 0, 0, 0] = 7.0
    values = SparseOrderForceConstants(3, 1, 2, np.asarray([[0, 1, 1]]), tensors)
    output = tmp_path / "FORCE_CONSTANTS_3RD"

    write_shengbte(output, values, supercell)

    lines = output.read_text().splitlines()
    assert lines[0] == "    2"
    assert " 1  1  1     7.0000000000e+00" in lines


def test_force_constants_writes_closed_sparse_support_without_materializing(tmp_path):
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
    result.write(faithful, format="shengbte")
    assert 3 not in result.arrays

    assert " 1  1  1     7.0000000000e+00" in faithful.read_text().splitlines()


def test_sparse_writer_uses_general_lattice_labels_for_reordered_nondiagonal_cells(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, index = make_supercell(primitive, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    supercell = supercell[[1, 0]]
    # Rebuild the index in the deliberately shuffled reference order.
    from mlfcs.core.geometry import PeriodicIndex

    index = PeriodicIndex(
        supercell.arrays["primitive_index"],
        supercell.arrays["cell_translation"],
        supercell.info["mlfcs_supercell_matrix"],
    )
    first = index.representative(0)
    tail = index.atom(0, [1, 0, 0])
    sparse = SparseOrderForceConstants(
        3, 1, 2, np.asarray([[first, tail, tail]]), np.ones((1, 3, 3, 3))
    )
    output = tmp_path / "FORCE_CONSTANTS_3RD"
    write_shengbte(output, sparse, supercell)

    lines = output.read_text().splitlines()
    assert lines[0] == "    2"
    assert lines[5] == "     1      1      1"
    assert "5.0000000000e+00" in lines[3]


@pytest.mark.parametrize("order", [3, 4])
def test_writer_resolves_quotient_labels_to_nearest_shengbte_images(tmp_path, order):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, index = make_supercell(primitive, (4, 1, 1))
    anchor = index.atom(0, [0, 0, 0])
    tail = index.atom(0, [3, 0, 0])
    sparse = SparseOrderForceConstants(
        order,
        1,
        4,
        np.asarray([[anchor, *([tail] * (order - 1))]]),
        np.ones((1,) + (3,) * order),
    )
    output = tmp_path / f"FORCE_CONSTANTS_{order}TH"

    write_shengbte(output, sparse, supercell)

    lines = output.read_text().splitlines()
    assert lines[0] == "    1"
    for line in lines[3 : 3 + order - 1]:
        assert line.split() == [
            "-5.0000000000e+00",
            "0.0000000000e+00",
            "0.0000000000e+00",
        ]


def test_writer_rejects_orders_outside_shengbte_contract(tmp_path):
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(atoms, (1, 1, 1))
    values = SparseOrderForceConstants(
        5, 1, 1, np.zeros((1, 5), dtype=np.int32), np.zeros((1,) + (3,) * 5)
    )
    output = tmp_path / "FORCE_CONSTANTS_5TH"
    with pytest.raises(ValueError, match="third- and fourth-order"):
        write_shengbte(output, values, supercell)
