"""Physical dataset preparation and force-subtraction workflows."""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, ForceDataset


@pytest.fixture
def mapping():
    """Provide a small mapped cubic supercell."""
    atoms = bulk("Ar", "sc", a=2.0)
    return ClusterMap(ClusterSpace(atoms, cutoffs={2: 0.1}), atoms.repeat((2, 1, 1)))


def test_dataset_preserves_frames_and_periodic_displacements(mapping):
    """Collect stored forces in input order and unwrap periodic displacements."""
    frames = []
    for value in (0.02, -0.01):
        atoms = mapping.supercell_atoms
        atoms.positions[:, 0] += value
        atoms.positions[0] += mapping.cell[0]
        atoms.calc = SinglePointCalculator(atoms, forces=np.full((mapping.n_atoms, 3), value))
        frames.append(atoms)
    data = ForceDataset(mapping, iter(frames))
    np.testing.assert_allclose(
        data.displacements[:, :, 0], [[0.02, 0.02], [-0.01, -0.01]], atol=1e-15
    )
    np.testing.assert_allclose(data.displacements[:, :, 1:], 0, atol=1e-15)
    np.testing.assert_array_equal(data.forces[:, 0, 0], [0.02, -0.01])
    assert data.cluster_map is mapping
    np.testing.assert_array_equal(data.supercell_atoms.positions, mapping.supercell_atoms.positions)
    assert not data.forces.flags.writeable
    assert not data.displacements.flags.writeable


def test_single_supercell_force_subtraction_matches_per_frame(mapping):
    """Broadcast one force array without modifying the source dataset."""
    frames = []
    for value in (2.0, 3.0):
        atoms = mapping.supercell_atoms
        atoms.calc = SinglePointCalculator(atoms, forces=np.full((mapping.n_atoms, 3), value))
        frames.append(atoms)
    data = ForceDataset(mapping, frames)
    baseline = np.arange(mapping.n_atoms * 3).reshape(mapping.n_atoms, 3)
    broadcast = data.subtract_forces(baseline)
    repeated = data.subtract_forces(np.broadcast_to(baseline, data.forces.shape))
    np.testing.assert_array_equal(broadcast.forces, repeated.forces)
    np.testing.assert_array_equal(data.forces[:, 0, 0], [2, 3])
    assert broadcast.displacements is data.displacements
    assert broadcast.cluster_map is mapping
    for invalid in (0.0, np.zeros((1, mapping.n_atoms, 3)), np.full((mapping.n_atoms, 3), np.nan)):
        with pytest.raises(ValueError):
            data.subtract_forces(invalid)


def test_dataset_requires_existing_force_results(mapping):
    """Missing stored results fail rather than launch calculator evaluation."""
    with pytest.raises(ValueError, match="stored ASE forces"):
        ForceDataset(mapping, [mapping.supercell_atoms])
    with pytest.raises(ValueError, match="at least one"):
        ForceDataset(mapping, [])
    with pytest.raises(TypeError):
        ForceDataset(mapping, mapping.supercell_atoms)
