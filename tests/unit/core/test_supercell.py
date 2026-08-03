import numpy as np
from ase.build import bulk

from mlfcs.core.geometry import make_supercell


def test_supercell_index_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    assert len(supercell) == 16
    for atom in range(len(supercell)):
        assert index.atom(index.primitive[atom], index.translations[atom]) == atom


def test_grouped_permutation_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    expected = np.arange(16).reshape(8, 2).T.ravel()
    np.testing.assert_array_equal(index.grouped_permutation, expected)
    grouped = index.group_atoms(supercell)
    np.testing.assert_allclose(
        grouped.positions[index.internal_from_grouped],
        supercell.positions,
    )
