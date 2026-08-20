import numpy as np
from ase import Atoms
from ase.build import bulk
from supercell_helpers import make_supercell

from mlfcs import ForceConstantCalculation
from mlfcs.anharmonic.sscha import SSCHA
from mlfcs import build_supercell


def test_supercell_index_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    assert len(supercell) == 16
    for atom in range(len(supercell)):
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom


def test_diagonal_supercell_defaults_to_phonopy_primitive_site_major_order():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    supercell, index = make_supercell(primitive, (2, 1, 1))

    np.testing.assert_array_equal(index.primitive, [0, 0, 1, 1])
    np.testing.assert_array_equal(
        index.translations,
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
    )
    np.testing.assert_allclose(
        supercell.get_scaled_positions(),
        [[0, 0, 0], [0.5, 0, 0], [0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
    )


def test_phonopy_ordering_matches_the_old_style_non_diagonal_scan_order():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[[4, 0, 0], [0.3, 4, 0], [0.2, 0.1, 4]],
        pbc=True,
    )
    matrix = np.asarray([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    supercell, index = make_supercell(primitive, matrix)

    np.testing.assert_allclose(supercell.cell, matrix @ primitive.cell)
    np.testing.assert_array_equal(index.primitive, [0, 0, 1, 1])
    np.testing.assert_array_equal(
        index.translations,
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
    )


def test_internal_reference_construction_uses_phonopy_ordering():
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    reference = build_supercell(primitive, (2, 1, 1))
    calculation = ForceConstantCalculation(
        primitive, reference=reference, order=2, cutoff=3.0, verbose=False
    )
    sscha = SSCHA(primitive, reference=reference, snapshots=1, max_iterations=0)

    expected = np.asarray([0, 0, 1, 1])
    np.testing.assert_array_equal(calculation.index.primitive, expected)
    np.testing.assert_array_equal(sscha._index.primitive, expected)


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
    assert isinstance(supercell, Atoms)
