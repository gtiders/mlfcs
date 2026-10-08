"""Acoustic coordinates across force fitting, derivative recovery and model IO."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, FitSystem, ForceConstants
from mlfcs.dataset import ForceDataset
from mlfcs.finite_difference import FiniteDifference
from mlfcs.fitting.design import ForceDesign


@pytest.fixture(scope="module")
def acoustic_mapping():
    """Prepare a nearest-neighbour Si model with independent FC2/3/4 ASR factors."""
    primitive = bulk("Si", "diamond", a=5.43)
    space = ClusterSpace(primitive, cutoffs={2: 3.0, 3: 3.0, 4: 3.0}, asr=True)
    return ClusterMap(space, primitive.repeat((3, 3, 3)))


def physical_vector(space, free):
    """Lift order-local free coordinates into the complete physical vector."""
    result, offset = [], 0
    for order in space.orders:
        coordinates = space.acoustic_coordinates(order)
        result.append(coordinates.lift(free[offset : offset + coordinates.dimension]))
        offset += coordinates.dimension
    return np.concatenate(result)


def test_sparse_lift_adjoint_and_block_design(acoustic_mapping):
    """Verify adjoint duality and row-block force-design equivalence."""
    mapping, rng = acoustic_mapping, np.random.default_rng(481)
    space = mapping.cluster_space
    assert space.n_parameters == 22
    assert space.n_free_parameters == 9
    for block in space.blocks:
        coordinates = space.acoustic_coordinates(block.order)
        free = rng.normal(size=coordinates.dimension)
        probe = rng.normal(size=coordinates.width)
        np.testing.assert_allclose(
            probe @ coordinates.lift(free),
            coordinates.adjoint(probe) @ free,
            rtol=1e-12,
            atol=1e-12,
        )
    design = ForceDesign(mapping)
    displacement = rng.normal(scale=0.02, size=(mapping.n_atoms, 3))
    canonical = design.matrix(displacement)
    reduced = np.vstack(
        [matrix for _, matrix in design.iter_fit_blocks(displacement, rows_per_block=31)]
    )
    lift = np.column_stack(
        [physical_vector(space, column) for column in np.eye(space.n_free_parameters)]
    )
    np.testing.assert_allclose(reduced, canonical @ lift, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("representation", ["raw", "normal"])
def test_reduced_fit_preserves_physical_model(acoustic_mapping, representation):
    """Fit exact forces through each public route and restore canonical coefficients."""
    mapping, rng = acoustic_mapping, np.random.default_rng(82)
    space = mapping.cluster_space
    free = rng.normal(size=space.n_free_parameters)
    theta = physical_vector(space, free)
    design = ForceDesign(mapping)
    structures = []
    for _ in range(16):
        displacement = rng.normal(scale=0.08, size=(mapping.n_atoms, 3))
        sample = mapping.supercell_atoms
        sample.positions += displacement
        sample.calc = SinglePointCalculator(
            sample, forces=(design.matrix(displacement) @ theta).reshape(-1, 3)
        )
        structures.append(sample)
    system = FitSystem(ForceDataset(mapping, structures), representation=representation)
    assert system.n_parameters == 9
    if representation == "raw":
        model = system.solve()
        np.testing.assert_allclose(
            system.to_normal().normal_matrix,
            system.design_matrix.T @ system.design_matrix,
            atol=1e-11,
        )
    else:
        model = system.solve(rtol=1e-12, maxiter=1000)
    np.testing.assert_allclose(model.parameters(), theta, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(system.force_constants(free).parameters(), theta)
    assert system.rmse(model) < 1e-7


@pytest.mark.parametrize("order", [2, 3, 4])
def test_extrapolated_finite_difference_uses_acoustic_subspace(acoustic_mapping, order):
    """Recover exact polynomial forces while keeping canonical stencil ordering."""
    mapping, rng = acoustic_mapping, np.random.default_rng(29 + order)
    space = mapping.cluster_space
    coordinates = space.acoustic_coordinates(order)
    expected = coordinates.lift(rng.normal(size=coordinates.dimension))
    theta = np.zeros(space.n_parameters)
    theta[space.block(order).parameters] = expected
    design = ForceDesign(mapping)
    sampling = FiniteDifference(mapping, order=order, disps=(0.03, 0.05))
    structures = []
    for sample in sampling.displacements():
        displacement = sample.positions - mapping.supercell_atoms.positions
        sample.calc = SinglePointCalculator(
            sample, forces=(design.matrix(displacement) @ theta).reshape(-1, 3)
        )
        structures.append(sample)
    model = sampling.reconstruct(ForceDataset(mapping, structures))
    np.testing.assert_allclose(model.coefficients[order], expected, rtol=1e-8, atol=1e-8)


def test_canonical_save_and_mass_copy(acoustic_mapping, tmp_path):
    """Keep masses and native physical storage independent of prepared fitting factors."""
    space = acoustic_mapping.cluster_space
    copied = space.with_masses(space.masses * 1.01)
    assert copied.asr and copied.n_free_parameters == 9
    theta = physical_vector(space, np.ones(space.n_free_parameters))
    model = ForceConstants(space, {block.order: theta[block.parameters] for block in space.blocks})
    path = tmp_path / "acoustic.mlfcs"
    model.save(path)
    restored = ForceConstants.load(path)
    np.testing.assert_array_equal(restored.parameters(), theta)
    assert not restored.cluster_space.asr


def test_model_outside_acoustic_subspace_is_not_silently_projected(acoustic_mapping):
    """Reject a canonical model that reduced equations cannot evaluate faithfully."""
    coordinates = acoustic_mapping.cluster_space.acoustic_coordinates(2)
    invalid = np.zeros(coordinates.width)
    invalid[0] = 1.0
    with pytest.raises(ValueError, match="acoustic subspace"):
        coordinates.extract(invalid)


@pytest.mark.parametrize("representation", ["raw", "normal"])
def test_zero_dimensional_acoustic_fit(representation):
    """An onsite-only model has no nonzero translation-invariant force constants."""
    atoms = bulk("Ar", "sc", a=3.0)
    space = ClusterSpace(atoms, cutoffs={2: 0.1}, asr=True)
    mapping = ClusterMap(space, atoms)
    snapshot = atoms.copy()
    snapshot.positions += 0.01
    snapshot.calc = SinglePointCalculator(snapshot, forces=np.ones((1, 3)))
    system = FitSystem(ForceDataset(mapping, [snapshot]), representation=representation)
    assert system.n_parameters == 0
    np.testing.assert_array_equal(system.solve().parameters(), np.zeros(space.n_parameters))
    assert system.rmse(system.solve()) == 1.0


def test_equal_orbit_coordinate_maps_share_storage():
    """Share identical physical maps without merging separate orbit parameters."""
    atoms = Atoms(
        "SiGe",
        scaled_positions=[[0, 0, 0], [0.23, 0.31, 0.41]],
        cell=[[2.9, 0.1, 0.2], [0.2, 3.2, 0.5], [0.3, 0.4, 3.4]],
        pbc=True,
    )
    space = ClusterSpace(atoms, cutoffs={2: 3.5}, asr=True)
    coordinates = space.acoustic_coordinates(2)
    frame = np.kron(space.cell.T, space.cell.T)
    maps = {}
    for orbit, physical_map in zip(space.orbits, coordinates.physical_maps, strict=True):
        key = (
            orbit.lattice_basis.shape,
            orbit.lattice_basis.tobytes(),
            orbit.observation_rows.tobytes(),
        )
        np.testing.assert_array_equal(
            physical_map, (frame @ orbit.lattice_basis)[orbit.observation_rows]
        )
        assert not physical_map.flags.writeable
        if key in maps:
            assert maps[key] is physical_map
        else:
            maps[key] = physical_map
    assert len(maps) < len(space.orbits)
    assert coordinates.width == space.n_parameters
