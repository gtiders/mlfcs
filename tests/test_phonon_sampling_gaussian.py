import numpy as np
import pytest
from ase import Atoms
from supercell_helpers import monoatomic_periodic

from mlfcs import perturb_structures
from mlfcs.reciprocal import perturb_structures as harmonic_perturb_structures


def test_gaussian_sampling_is_reproducible_and_centered():
    reference = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=np.eye(3) * 4, pbc=True)
    first = perturb_structures(reference, snapshots=4, displacement=0.02, random_seed=9)
    second = perturb_structures(reference, snapshots=4, displacement=0.02, random_seed=9)
    first_u = np.asarray([atoms.positions - reference.positions for atoms in first])
    second_u = np.asarray([atoms.positions - reference.positions for atoms in second])
    np.testing.assert_array_equal(first_u, second_u)
    np.testing.assert_allclose(first_u.mean(axis=1), 0.0, atol=1e-15)
    assert [atoms.info["mlfcs_configuration_id"] for atoms in first] == list(range(4))


def test_root_entry_point_only_generates_cartesian_gaussian_snapshots():
    """Harmonic sampling moved to `mlfcs.reciprocal`, and the root entry point says so."""
    reference = monoatomic_periodic()

    with pytest.raises(ValueError, match="mlfcs.reciprocal.perturb_structures"):
        perturb_structures(reference, snapshots=1, method="harmonic")
    # A harmonic-only keyword is not silently accepted either: the Gaussian entry point
    # has no temperature and says so through the argument name.
    with pytest.raises(TypeError, match="temperature"):
        perturb_structures(reference, snapshots=1, temperature=300)


def test_harmonic_sampling_validates_its_own_parameters():
    reference = monoatomic_periodic()

    with pytest.raises(ValueError, match="force_constants is required"):
        harmonic_perturb_structures(reference, snapshots=1, method="harmonic", temperature=300)
    with pytest.raises(ValueError, match="require harmonic"):
        harmonic_perturb_structures(reference, snapshots=1, method="gaussian", temperature=300)
