import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.fitting import FitDataset
from mlfcs.structure.relation import StructureRelation


def _with_forces(atoms, forces):
    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(atoms, forces=np.asarray(forces))
    return atoms


def test_reference_mapping_and_force_only_training_data():
    primitive = Atoms("Ar", cell=np.eye(3) * 3.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = _with_forces(primitive.repeat((2, 1, 1)), np.zeros((2, 3)))
    frame = reference.copy()
    frame.positions[0, 0] += 0.02
    frame.positions[1, 0] -= 0.02
    frame = _with_forces(frame, [[-0.1, 0, 0], [0.1, 0, 0]])

    geometry = StructureRelation.from_atoms(primitive, reference)
    dataset = FitDataset.from_atoms(geometry, [frame])

    np.testing.assert_array_equal(geometry.supercell_matrix, np.diag([2, 1, 1]))
    assert dataset.displacements.shape == (1, 2, 3)
    assert dataset.forces.shape == (1, 2, 3)
    np.testing.assert_allclose(dataset.displacements.sum(axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(dataset.forces.sum(axis=1), 0.0, atol=1e-15)


def test_reference_forces_are_optional_but_training_forces_are_required():
    primitive = Atoms("Ar", cell=np.eye(3) * 3.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    geometry = StructureRelation.from_atoms(primitive, reference)
    frame = _with_forces(reference, np.zeros((2, 3)))

    dataset = FitDataset.from_atoms(geometry, [frame])

    np.testing.assert_array_equal(dataset.reference_forces, np.zeros((2, 3)))


def test_training_data_are_diagnosed_without_silent_recentering():
    primitive = Atoms("Ar", cell=np.eye(3) * 3.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = _with_forces(primitive.repeat((2, 1, 1)), [[0.01, 0, 0], [-0.01, 0, 0]])
    frame = reference.copy()
    frame.positions += [0.02, 0, 0]
    frame = _with_forces(frame, [[0.04, 0, 0], [0.02, 0, 0]])

    dataset = FitDataset.from_atoms(StructureRelation.from_atoms(primitive, reference), [frame])

    np.testing.assert_allclose(dataset.displacements[0, :, 0], 0.02)
    np.testing.assert_allclose(dataset.displacements[0, :, 1:], 0.0)
    np.testing.assert_allclose(dataset.forces[0, :, 0], [0.03, 0.03])
    np.testing.assert_allclose(dataset.center_of_mass_displacements, [[0.02, 0, 0]])
    np.testing.assert_allclose(dataset.net_forces, [[0.06, 0, 0]])


@pytest.mark.parametrize(
    ("forces", "message"),
    [
        (np.zeros((2, 2)), "shape"),
        (np.asarray([[np.nan, 0, 0], [0, 0, 0]]), "NaN or infinite"),
    ],
)
def test_fit_dataset_rejects_invalid_force_arrays(forces, message):
    primitive = Atoms("Ar", cell=np.eye(3) * 3.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    frame = reference.copy()
    frame.new_array("forces", forces)

    with pytest.raises(ValueError, match=message):
        FitDataset.from_atoms(StructureRelation.from_atoms(primitive, reference), [frame])
