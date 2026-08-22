import numpy as np
from ase import Atoms
from ase.build import bulk

from mlfcs import build_supercell
from mlfcs.structure.geometry import make_supercell


def test_supercell_index_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    assert len(supercell) == 16
    for atom in range(len(supercell)):
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom


def test_diagonal_supercell_keeps_the_legacy_cell_major_expansion_order():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    supercell, index = make_supercell(primitive, (2, 1, 1))

    np.testing.assert_array_equal(index.primitive, [0, 1, 0, 1])
    np.testing.assert_array_equal(
        index.translations,
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
    )
    np.testing.assert_allclose(supercell.positions[:2], primitive.positions)


def test_general_supercell_coset_enumeration_scales_with_determinant():
    primitive = Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 3, pbc=True)
    supercell, index = make_supercell(primitive, [[31, 7, 0], [0, 1, 0], [0, 0, 1]])

    assert len(supercell) == index.n_cells == 31
    assert len({index.residue(translation) for translation in index.translations}) == 31


def test_public_build_supercell_returns_metadata_bearing_ase_atoms():
    primitive = Atoms("He", positions=[[0, 0, 0]], cell=np.eye(3) * 3, pbc=True)
    supercell = build_supercell(primitive, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])

    assert isinstance(supercell, Atoms)
    assert len(supercell) == 2
    np.testing.assert_array_equal(
        supercell.info["mlfcs_supercell_matrix"], [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert {"primitive_index", "cell_translation", "primitive_scaled_position"} <= set(
        supercell.arrays
    )
