from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from supercell_helpers import make_supercell

from mlfcs.structure.periodic_geometry import PeriodicGeometry
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations, SymmetryOperations


@pytest.mark.parametrize(("residual", "accepted"), ((0.5e-5, True), (5e-5, False)))
def test_primitive_symmetry_mapping_uses_exactly_symprec(monkeypatch, residual, accepted):
    primitive = bulk("Ar", "sc", a=1.0)
    dataset = SimpleNamespace(
        rotations=np.eye(3, dtype=np.int32)[None, :, :],
        translations=np.array([[residual, 0.0, 0.0]]),
        international="P1",
    )
    monkeypatch.setattr(
        "mlfcs.structure.symmetry.spglib.get_symmetry_dataset",
        lambda cell, symprec: dataset,
    )

    if accepted:
        operations = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
        assert operations.size == 1
    else:
        with pytest.raises(ValueError, match="nearest residual.*angstrom"):
            PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)


def test_every_symmetry_operation_is_an_atom_permutation():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, _ = make_supercell(primitive, (2, 2, 2))
    relation = StructureRelation.from_atoms(primitive, supercell, symprec=1e-5)
    primitive_operations = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    operations = SymmetryOperations.from_primitive_operations(primitive_operations, relation.index)
    assert operations.size > 1
    for permutation in operations.atom_permutations:
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(supercell)))


def test_exact_affine_permutations_match_cartesian_symmetry_positions():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, _ = make_supercell(primitive, [[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    supercell = supercell[np.random.default_rng(8).permutation(len(supercell))]
    relation = StructureRelation.from_atoms(primitive, supercell, symprec=1e-5)
    primitive_operations = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    operations = SymmetryOperations.from_primitive_operations(primitive_operations, relation.index)
    inverse = np.linalg.inv(np.asarray(primitive.cell))
    fractional = supercell.positions @ inverse
    for operation, (rotation, translation) in enumerate(
        zip(operations.rotations, operations.translations, strict=True)
    ):
        transformed = (fractional @ rotation.T + translation) @ primitive.cell
        expected = supercell.positions[operations.atom_permutations[operation]]
        delta, _ = PeriodicGeometry(supercell.cell).mic(transformed - expected)
        np.testing.assert_allclose(delta, 0.0, atol=1e-8, rtol=0.0)
