import numpy as np
from ase.build import bulk

from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.structure.geometry import make_supercell


def test_every_symmetry_operation_is_an_atom_permutation():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, _ = make_supercell(primitive, (2, 2, 2))
    operations = SymmetryOperations.from_atoms(primitive, supercell)
    assert operations.size > 1
    for permutation in operations.atom_permutations:
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(supercell)))
