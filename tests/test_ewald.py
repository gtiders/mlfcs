"""Screened three-dimensional dipole sums and symmetry-compatible local corrections."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import make_supercell
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, DipoleEwald, FitSystem, ForceDataset
from mlfcs.fitting.design import ForceDesign
from mlfcs.foundation.errors import ConstraintProjectionError
from mlfcs.phonon.ewald import _raw_fc2


@pytest.fixture(scope="module")
def polar_model():
    """Construct a P1 insulator with nonsymmetric neutral Born tensors."""
    atoms = Atoms(
        numbers=[1, 2, 3],
        scaled_positions=[[0, 0, 0], [0.15, 0.2, 0.1], [0.33, 0.27, 0.38]],
        cell=[[4, 0.2, 0.1], [0.1, 4.4, 0.3], [0.2, 0.1, 4.8]],
        pbc=True,
    )
    space = ClusterSpace(atoms, cutoffs={2: 3.0})
    mapping = ClusterMap(space, make_supercell(atoms, [[2, 1, 0], [0, 1, 0], [0, 0, 1]]))
    born = np.array(
        [[[1, 0.2, 0], [0.1, 0.8, 0], [0, 0.1, 1.2]], [[0.4, 0, 0.2], [0, 0.6, 0], [0.1, 0, 0.3]]]
    )
    born = np.concatenate((born, -born.sum(axis=0)[None]))
    epsilon = np.array([[2, 0.1, 0.2], [0.1, 3, 0.1], [0.2, 0.1, 2.5]])
    return mapping, born, epsilon


@pytest.fixture(scope="module")
def ewald(polar_model):
    """Build one converged locally corrected long-range model."""
    mapping, born, epsilon = polar_model
    return DipoleEwald(mapping, born_charges=born, dielectric=epsilon)


@pytest.fixture(scope="module")
def long_fc(ewald):
    """Compute reusable long-range tensors for the selected non-diagonal supercell."""
    return ewald.force_constants()


def test_local_correction_produces_a_symmetric_acoustic_hessian(ewald, long_fc):
    """Nontrivial low-symmetry correction enforces both exchange symmetry and ASR."""
    full = long_fc.full_array(2)
    assert ewald.correction_norm > 0.01
    np.testing.assert_allclose(full, full.transpose(1, 0, 3, 2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(full.sum(axis=1), 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        long_fc.harmonic_forces(np.ones((long_fc.cluster_map.n_atoms, 3))), 0, atol=1e-12
    )


def test_compact_force_contraction_and_energy_derivative(long_fc):
    """Check the shared force operator against its dense energy Hessian."""
    n = long_fc.cluster_map.n_atoms
    u = np.random.default_rng(4).normal(scale=0.01, size=(3, n, 3))
    full = long_fc.full_array(2)
    expected = -np.einsum("ijab,fjb->fia", full, u)
    np.testing.assert_allclose(long_fc.harmonic_forces(u), expected, atol=1e-14)
    np.testing.assert_allclose(long_fc.harmonic_forces(u[0]), expected[0], atol=1e-14)
    hessian = full.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    step = 1e-5
    vector = u[0].reshape(-1)
    direction = np.random.default_rng(5).normal(size=3 * n)
    plus = vector + step * direction
    minus = vector - step * direction
    derivative = (plus @ hessian @ plus - minus @ hessian @ minus) / (4 * step)
    np.testing.assert_allclose(derivative, -expected[0].reshape(-1) @ direction, atol=1e-12)


def test_full_ewald_is_independent_of_the_splitting_parameter(polar_model):
    """Real, reciprocal and self terms cancel dependence on Gaussian splitting."""
    mapping, born, epsilon = polar_model
    space = mapping.cluster_space
    results = [
        _raw_fc2(
            space.cell,
            space.cartesian_positions,
            np.arange(3),
            np.arange(3),
            born,
            epsilon,
            alpha,
            7.0,
        )
        for alpha in (0.45, 0.8)
    ]
    np.testing.assert_allclose(results[0], results[1], atol=1e-11, rtol=1e-11)


def test_folded_nonzero_translation_images_do_not_become_onsite(polar_model, ewald, long_fc):
    """A one-cell fold sums periodic images and retains the corrected primitive ASR."""
    mapping, born, epsilon = polar_model
    primitive_map = ClusterMap(mapping.cluster_space, mapping.cluster_space.primitive_atoms)
    primitive = DipoleEwald(primitive_map, born_charges=born, dielectric=epsilon).force_constants()
    folded = np.zeros_like(primitive.compact_array(2))
    for atom, site in enumerate(mapping.primitive_site_indices):
        folded[:, site] += long_fc.compact_array(2)[:, atom]
    np.testing.assert_allclose(folded, primitive.compact_array(2), atol=1e-11, rtol=1e-11)


def test_zero_born_charges_produce_zero_fc2_and_forces(polar_model):
    """A nonpolar input has no long-range contribution or local correction."""
    mapping, born, epsilon = polar_model
    zero = DipoleEwald(mapping, born_charges=np.zeros_like(born), dielectric=epsilon)
    np.testing.assert_array_equal(zero.force_constants().compact_array(2), 0)
    assert zero.correction_norm == 0


def test_insufficient_local_support_fails_without_fallback(polar_model):
    """An onsite-only model cannot cancel a nonzero antisymmetric dipole row sum."""
    mapping, born, epsilon = polar_model
    atoms = mapping.cluster_space.primitive_atoms
    small = ClusterMap(ClusterSpace(atoms, cutoffs={2: 0.01}), atoms)
    with pytest.raises(ConstraintProjectionError, match="insufficient"):
        DipoleEwald(small, born_charges=born, dielectric=epsilon)


def test_long_range_subtraction_restores_the_same_fitting_equations(long_fc):
    """Subtracting synthetic Ewald forces leaves the original short-range problem."""
    mapping = long_fc.cluster_map
    # Use the same mapped supercell for the synthetic forces and fitted equations.
    u = np.random.default_rng(6).normal(scale=0.01, size=(8, mapping.n_atoms, 3))
    design = ForceDesign(mapping)
    target = np.random.default_rng(7).normal(size=design.n_parameters)
    short = np.array([(design.matrix(frame) @ target).reshape(mapping.n_atoms, 3) for frame in u])
    frames = []
    long = long_fc.harmonic_forces(u)
    for displacement, force in zip(u, short + long, strict=True):
        atoms = mapping.supercell_atoms
        atoms.positions += displacement
        atoms.calc = SinglePointCalculator(atoms, forces=force)
        frames.append(atoms)
    data = ForceDataset(mapping, frames)
    residual = data.subtract_forces(long_fc.harmonic_forces(data.displacements))
    np.testing.assert_allclose(residual.forces, short, atol=1e-13)
    system = FitSystem(residual, representation="raw")
    assert system.rmse(target) < 1e-12


@pytest.mark.parametrize("invalid", ["charge", "dielectric", "tolerance"])
def test_invalid_physical_inputs_are_rejected(polar_model, invalid):
    """Reject nonneutral charges, an indefinite dielectric or invalid precision."""
    mapping, born, epsilon = polar_model
    born, epsilon = born.copy(), epsilon.copy()
    options = {}
    if invalid == "charge":
        born[0, 0, 0] += 0.1
    elif invalid == "dielectric":
        epsilon[0, 0] = -1
    else:
        options["rtol"] = 0
    with pytest.raises(ValueError):
        DipoleEwald(mapping, born_charges=born, dielectric=epsilon, **options)


def test_cubic_space_group_preserves_born_action_and_fc2():
    """Check a rocksalt dipole model in a high-symmetry primitive basis."""
    atoms = Atoms(
        numbers=[11, 17],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[[0, 2.8, 2.8], [2.8, 0, 2.8], [2.8, 2.8, 0]],
        pbc=True,
    )
    mapping = ClusterMap(ClusterSpace(atoms, cutoffs={2: 3.0}), atoms)
    born = np.array([np.eye(3), -np.eye(3)])
    model = DipoleEwald(mapping, born_charges=born, dielectric=2.5 * np.eye(3))
    full = model.force_constants().full_array(2)
    np.testing.assert_allclose(full.sum(axis=1), 0, atol=1e-12)
    for row_rotation, permutation in zip(
        mapping.cluster_space.symmetry.cartesian_rotations,
        mapping.cluster_space.symmetry.site_permutations,
        strict=True,
    ):
        rotated = np.einsum("ab,ijbc,cd->ijad", row_rotation.T, full, row_rotation)
        np.testing.assert_allclose(rotated, full[permutation][:, permutation], atol=1e-12)


def test_cartesian_rotation_preserves_the_local_correction(polar_model, ewald, long_fc):
    """A physical coordinate rotation rotates the final Hessian and force convention."""
    mapping, born, epsilon = polar_model
    rotation, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(3, 3)))
    atoms = mapping.cluster_space.primitive_atoms
    atoms.positions[:] = atoms.positions @ rotation
    atoms.cell[:] = atoms.cell.array @ rotation
    space = ClusterSpace(atoms, cutoffs={2: 3.0})
    supercell = mapping.supercell_atoms
    supercell.positions[:] = supercell.positions @ rotation
    supercell.cell[:] = supercell.cell.array @ rotation
    rotated_map = ClusterMap(space, supercell)
    rotated_born = np.einsum("ab,ibc,cd->iad", rotation.T, born, rotation)
    rotated_epsilon = rotation.T @ epsilon @ rotation
    # Roundoff normalization here belongs to the test's tensor construction.
    rotated_epsilon = (rotated_epsilon + rotated_epsilon.T) / 2
    model = DipoleEwald(rotated_map, born_charges=rotated_born, dielectric=rotated_epsilon)
    expected = np.einsum("ab,ijbc,cd->ijad", rotation.T, long_fc.full_array(2), rotation)
    np.testing.assert_allclose(model.force_constants().full_array(2), expected, atol=2e-12)
    np.testing.assert_allclose(model.correction_norm, ewald.correction_norm, atol=1e-12)


def test_dipole_kernel_matches_independent_green_function_derivatives(polar_model):
    """Check signs and dielectric factors with a finite-difference scalar Green sum."""
    from itertools import product

    from ase import units
    from scipy.special import erfc

    mapping, born, epsilon = polar_model
    space = mapping.cluster_space
    alpha = 0.65
    real = np.array(list(product(range(-6, 7), repeat=3))) @ space.cell
    reciprocal = np.array(list(product(range(-6, 7), repeat=3))) @ (
        2 * np.pi * np.linalg.inv(space.cell).T
    )
    reciprocal = reciprocal[np.any(reciprocal != 0, axis=1)]
    inverse = np.linalg.inv(epsilon)
    denominator = np.einsum("ij,jk,ik->i", reciprocal, epsilon, reciprocal)
    factors = (
        4
        * np.pi
        / abs(np.linalg.det(space.cell))
        * np.exp(-denominator / (4 * alpha**2))
        / denominator
    )

    def green(distance):
        """Evaluate the screened scalar periodic potential independently of the kernel."""
        vectors = distance + real
        rho = np.sqrt(np.einsum("ij,jk,ik->i", vectors, inverse, vectors))
        return np.sum(erfc(alpha * rho) / rho) / np.sqrt(np.linalg.det(epsilon)) + factors @ np.cos(
            reciprocal @ distance
        )

    distance = space.cartesian_positions[1] - space.cartesian_positions[0]
    step = 1e-4
    dipole = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            first, second = np.eye(3)[a] * step, np.eye(3)[b] * step
            dipole[a, b] = -(
                green(distance + first + second)
                - green(distance + first - second)
                - green(distance - first + second)
                + green(distance - first - second)
            ) / (4 * step**2)
    expected = units.Hartree * units.Bohr * born[0].T @ dipole @ born[1]
    result = _raw_fc2(
        space.cell, space.cartesian_positions, np.arange(3), np.arange(3), born, epsilon, alpha, 7.0
    )
    np.testing.assert_allclose(result[0, 1], expected, atol=1e-7, rtol=1e-6)


def test_correction_support_changes_partition_but_preserves_constraints(
    polar_model, ewald, long_fc
):
    """A different local cutoff changes the partition, not the required Hessian invariants."""
    mapping, born, epsilon = polar_model
    larger_space = ClusterSpace(mapping.cluster_space.primitive_atoms, cutoffs={2: 4.0})
    larger_map = ClusterMap(larger_space, mapping.supercell_atoms)
    larger = DipoleEwald(larger_map, born_charges=born, dielectric=epsilon)
    assert abs(larger.correction_norm - ewald.correction_norm) > 1e-4
    assert np.max(np.abs(larger.force_constants().full_array(2) - long_fc.full_array(2))) > 1e-4
    full = larger.force_constants().full_array(2)
    np.testing.assert_allclose(full, full.transpose(1, 0, 3, 2), atol=1e-12)
    np.testing.assert_allclose(full.sum(axis=1), 0, atol=1e-12)


def test_bound_mapping_force_api_reuses_the_same_hessian(ewald, polar_model, long_fc):
    """Direct mapped force calculation agrees with reuse of exported compact tensors."""
    mapping = polar_model[0]
    assert ewald.force_constants() is long_fc
    u = np.random.default_rng(9).normal(scale=0.01, size=(2, mapping.n_atoms, 3))
    np.testing.assert_allclose(ewald.forces(u), long_fc.harmonic_forces(u), atol=1e-14)
    reordered = ClusterMap(mapping.cluster_space, mapping.supercell_atoms[::-1])
    second = DipoleEwald(
        reordered, born_charges=ewald.born_charges, dielectric=ewald.dielectric
    ).force_constants()
    full = second.full_array(2)
    np.testing.assert_allclose(full, long_fc.full_array(2)[::-1, ::-1], atol=1e-12)
    np.testing.assert_allclose(
        second.harmonic_forces(u[:, ::-1]), long_fc.harmonic_forces(u)[:, ::-1], atol=1e-13
    )


def test_long_range_export_uses_the_same_compact_values(long_fc, tmp_path):
    """Phonopy export retains onsite and folded tails without requiring a short-range model."""
    import h5py

    file = long_fc.write(
        tmp_path / "fc2.hdf5", format="phonopy", order=2, storage="hdf5", threshold=0
    )
    with h5py.File(file) as handle:
        np.testing.assert_array_equal(handle["force_constants"][:], long_fc.full_array(2))


def test_finite_difference_extrapolation_after_long_range_subtraction():
    """Long-range subtraction preserves zero-step extrapolation of ordered force stencils."""
    from mlfcs import FiniteDifference

    atoms = Atoms(
        numbers=[11, 17],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[[0, 2.8, 2.8], [2.8, 0, 2.8], [2.8, 2.8, 0]],
        pbc=True,
    )
    space = ClusterSpace(atoms, cutoffs={2: 0.01})
    mapping = ClusterMap(space, atoms)
    long = DipoleEwald(
        mapping, born_charges=np.array([np.eye(3), -np.eye(3)]), dielectric=2.5 * np.eye(3)
    ).force_constants()
    fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))
    frames = []
    stiffness = 2.5
    for sample in fd.displacements():
        u = sample.positions - mapping.supercell_atoms.positions
        force = -stiffness * u - 7 * u**3 + long.harmonic_forces(u)
        sample.calc = SinglePointCalculator(sample, forces=force)
        frames.append(sample)
    data = ForceDataset(mapping, frames)
    residual = data.subtract_forces(long.harmonic_forces(data.displacements))
    model = fd.reconstruct(residual)
    np.testing.assert_allclose(model.coefficients[2], stiffness, atol=1e-11, rtol=0)
