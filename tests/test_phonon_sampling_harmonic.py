import numpy as np
import pytest
from ase import Atoms, units
from supercell_helpers import make_supercell, monoatomic_periodic

from mlfcs.phonon.sampling.harmonic import HarmonicSampler


def _chain(spring=1.0):
    primitive = monoatomic_periodic("Al")
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    fc2 = np.zeros((1, 2, 3, 3))
    for axis in range(3):
        fc2[0, 0, axis, axis] = spring
        fc2[0, 1, axis, axis] = -spring
    return primitive, supercell, fc2


def test_classical_sampling_matches_analytic_variance():
    primitive, supercell, fc2 = _chain(spring=1.0)
    ensemble = HarmonicSampler(primitive, supercell, fc2, temperature=300, statistics="classical")
    displacement = ensemble.sample(80_000, random_seed=12)
    expected = units.kB * 300 / 4

    np.testing.assert_allclose(np.var(displacement[:, 0], axis=0), expected, rtol=0.02)
    np.testing.assert_allclose(displacement.sum(axis=1), 0.0, atol=1e-14)


def test_quantum_zero_temperature_has_zero_point_motion():
    primitive, supercell, fc2 = _chain(spring=1.0)
    ensemble = HarmonicSampler(primitive, supercell, fc2, temperature=0, statistics="quantum")
    displacement = ensemble.sample(60_000, random_seed=8)

    assert np.all(np.var(displacement[:, 0], axis=0) > 0)
    assert np.isfinite(ensemble.harmonic_free_energy())


def test_maximum_displacement_is_optional_and_reports_clipping():
    primitive, supercell, fc2 = _chain(spring=0.01)
    unrestricted = HarmonicSampler(
        primitive, supercell, fc2, temperature=1000, statistics="classical"
    )
    raw = unrestricted.sample(100, random_seed=4)
    assert np.max(np.linalg.norm(raw, axis=2)) > 0.05
    assert unrestricted.state.clipped_atoms == 0

    limited = HarmonicSampler(
        primitive,
        supercell,
        fc2,
        temperature=1000,
        statistics="classical",
        max_displacement=0.05,
    )
    clipped = limited.sample(100, random_seed=4)
    assert np.max(np.linalg.norm(clipped, axis=2)) <= 0.05 + 1e-14
    assert limited.state.clipped_atoms > 0
    assert limited.state.affected_snapshots > 0


def test_imaginary_mode_policy_is_explicit():
    primitive, supercell, fc2 = _chain(spring=-1.0)
    with pytest.raises(ValueError, match="imaginary harmonic modes"):
        HarmonicSampler(primitive, supercell, fc2, temperature=300)
    excluded = HarmonicSampler(
        primitive, supercell, fc2, temperature=300, imaginary_modes="exclude"
    )
    assert excluded.state.imaginary_modes == 3
    assert excluded.state.sampled_modes == 0


def test_sampling_supports_nondiagonal_reordered_reference_supercells():
    primitive = Atoms("Al", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    supercell, _ = make_supercell(primitive, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    supercell = supercell[[1, 0]]
    compact = np.zeros((1, 2, 3, 3))
    compact[0, :, :, :] = np.eye(3) / 2
    ensemble = HarmonicSampler(
        primitive, supercell, compact, temperature=300, statistics="classical"
    )

    samples = ensemble.sample(3, random_seed=4)
    assert samples.shape == (3, 2, 3)
    assert len(ensemble.qpoints) == 2
