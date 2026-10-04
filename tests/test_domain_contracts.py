"""Admission, ownership, workspace and persistence contracts."""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from numba import get_num_threads, set_num_threads

from mlfcs import ClusterMap, ClusterSpace
from mlfcs._arrays import integer_array
from mlfcs.fitting.design import ForceDesign


def test_one_immutable_space_maps_to_multiple_supercells_without_prepared_wrappers():
    atoms = bulk("Ar", "sc", a=1)
    space = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 1.1}, max_body_orders={2: 2})
    small = ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large = ClusterMap(
        space, atoms.repeat((3,) * 3), supercell_matrix=3 * np.eye(3, dtype=np.int64)
    )
    assert small.cluster_space is large.cluster_space is space
    assert not hasattr(space, "supercell")
    assert small.rank_info().nullity > 0
    assert large.rank_info().full
    assert not space.scaled_positions.flags.writeable
    assert not space.symmetry.rotations.flags.writeable
    assert not large.image_atom_indices[0].flags.writeable
    assert not hasattr(space, "prepare")
    assert not hasattr(large, "prepare")


def test_normalization_rejects_overflow_before_cast_and_does_not_freeze_callers():
    for values in (
        [2**70],
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([-(2**63)], dtype=np.int64),
    ):
        with pytest.raises(OverflowError):
            integer_array(values)
    with pytest.raises(ValueError):
        integer_array([1.0])
    source = np.arange(8, dtype=np.int64)[::2]
    prepared = integer_array(source)
    assert source.flags.writeable
    assert prepared.flags.c_contiguous and not prepared.flags.writeable
    source[0] = 17
    assert prepared[0] == 0


def test_workspace_thread_admission_and_single_parallel_equivalence():
    atoms = bulk("Ar", "sc", a=1)
    space = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 1.1}, max_body_orders={2: 2})
    mapping = ClusterMap(
        space, atoms.repeat((3,) * 3), supercell_matrix=3 * np.eye(3, dtype=np.int64)
    )
    design = ForceDesign(mapping)
    previous = get_num_threads()
    try:
        set_num_threads(1)
        workspace = design.allocate_workspace()
        displacement = np.random.default_rng(81).normal(size=(27, 3))
        single = design.matrix(displacement, workspace=workspace)
        if previous >= 2:
            set_num_threads(2)
            with pytest.raises(ValueError, match="threads"):
                design.matrix(displacement, workspace=workspace)
            parallel = design.matrix(displacement)
            np.testing.assert_array_equal(single, parallel)
    finally:
        set_num_threads(previous)


def test_version_four_model_loads_into_the_validated_int64_domain(monkeypatch):
    from mlfcs import ForceConstants

    def no_neighbor_search(*args, **kwargs):
        pytest.fail("loading stored orbit data must not enumerate neighbors")

    monkeypatch.setattr("mlfcs.cluster_space.candidates.neighbors", no_neighbor_search)
    model = ForceConstants.load(Path(__file__).parent / "data" / "force_constants_v4.mlfcs")
    assert model.cluster_space.orbits[0].exact_lattice_basis.dtype == np.int64
    assert not model.cluster_space.orbits[0].exact_lattice_basis.flags.writeable
    np.testing.assert_array_equal(model.parameters(), 1.0)
    space = model.cluster_space
    mapping = ClusterMap(
        space,
        space.primitive_atoms.repeat((3,) * 3),
        supercell_matrix=3 * np.eye(3, dtype=np.int64),
    )
    design = ForceDesign(mapping)
    np.testing.assert_array_equal(design.matrix(np.zeros((len(mapping.supercell_atoms), 3))), 0.0)


def test_design_and_fitted_force_constants_match_original_python_snapshots():
    from ase.calculators.singlepoint import SinglePointCalculator

    from mlfcs import FitSystem

    atoms = bulk("Si", "diamond", a=5.43)
    space = ClusterSpace(
        atoms, symprec=1e-05, cutoffs={2: 4.0, 3: 4.0}, max_body_orders={2: 2, 3: 3}
    )
    reference_atoms = atoms.repeat((3,) * 3)
    mapping = ClusterMap(space, reference_atoms, supercell_matrix=3 * np.eye(3, dtype=np.int64))
    design = ForceDesign(mapping)
    workspace = design.allocate_workspace()
    with np.load(Path(__file__).parent / "data" / "force_design_reference.npz") as reference:
        # Different exact kernel generators yield different physical-component
        # parameter coordinates. Compare designs in the original coordinates.
        change = np.zeros((space.n_parameters, space.n_parameters))
        for index, orbit in enumerate(space.orbits):
            old_basis = reference[f"basis{index}"]
            new_basis = orbit.component_basis
            old_q, _ = np.linalg.qr(old_basis)
            new_q, _ = np.linalg.qr(new_basis)
            np.testing.assert_allclose(new_q @ new_q.T, old_q @ old_q.T, atol=2e-12)
            transform = np.linalg.lstsq(new_basis, old_basis, rcond=None)[0]
            np.testing.assert_allclose(new_basis @ transform, old_basis, atol=2e-12)
            start, stop = space.parameter_offsets[index : index + 2]
            change[start:stop, start:stop] = transform
        structures = []
        for displacement, forces, expected in zip(
            reference["displacements"], reference["forces"], reference["matrices"], strict=True
        ):
            np.testing.assert_allclose(
                design.matrix(displacement, workspace=workspace) @ change, expected, atol=1e-13
            )
            sample = reference_atoms.copy()
            sample.positions += displacement
            sample.calc = SinglePointCalculator(sample, forces=forces)
            structures.append(sample)
        system = FitSystem(mapping, structures)
        model = system.solve(rtol=1e-11, maxiter=5000)
        parameters = model.parameters()
        for index, orbit in enumerate(space.orbits):
            start, stop = space.parameter_offsets[index : index + 2]
            actual = orbit.component_basis @ parameters[start:stop]
            np.testing.assert_allclose(actual, reference[f"tensor{index}"], atol=2e-8)


def test_direct_domains_have_explicit_detached_ase_snapshots():
    import mlfcs

    atoms = bulk("Ar", "sc", a=1)
    atoms.set_masses([42.0])
    space = ClusterSpace(atoms, cutoffs={2: 0.1})
    explicit = atoms.repeat((2, 1, 1))
    mapping = ClusterMap(space, explicit, supercell_matrix=np.diag([2, 1, 1]))
    atoms.positions[:] = 4.0
    explicit.positions[:] = 5.0
    primitive_snapshot = space.primitive_atoms
    supercell_snapshot = mapping.supercell_atoms
    np.testing.assert_array_equal(primitive_snapshot.get_masses(), [42.0])
    np.testing.assert_array_equal(supercell_snapshot.get_masses(), [42.0, 42.0])
    primitive_snapshot.positions[:] = 7.0
    primitive_snapshot.set_masses([9.0])
    supercell_snapshot.positions[:] = 8.0
    supercell_snapshot.set_masses([10.0, 10.0])
    np.testing.assert_array_equal(space.primitive_atoms.get_masses(), [42.0])
    np.testing.assert_array_equal(mapping.supercell_atoms.get_masses(), [42.0, 42.0])
    assert not np.allclose(space.primitive_atoms.positions, 7.0)
    assert not np.allclose(mapping.supercell_atoms.positions, 8.0)
    for owner in (space, mapping):
        for name in ("atoms", "primitive", "supercell", "space", "build", "from_atoms", "map_to"):
            assert not hasattr(owner, name)
    for name in ("PrimitiveCell", "Supercell", "TaylorCalculator"):
        assert not hasattr(mlfcs, name)
    assert space.orders == (2,)
    assert space.block(2).max_body_order == 2
    assert mapping.cluster_space is space
    for value in (
        space.atomic_numbers,
        mapping.supercell_matrix,
        mapping.atomic_numbers,
        mapping.primitive_site_indices,
        mapping.lattice_translations,
        mapping.quotient_labels,
        *mapping.image_atom_indices,
    ):
        assert value.dtype == np.int64
        assert value.flags.c_contiguous and not value.flags.writeable


def test_native_header_rejects_legacy_models_before_decoding(tmp_path, monkeypatch):
    from mlfcs import ForceConstants

    def fail_decode(group):
        pytest.fail("legacy file must be rejected before model decoding")

    monkeypatch.setattr("mlfcs.force_constants._native._read_space", fail_decode)
    legacy = tmp_path / "legacy.mlfcs"
    legacy.write_bytes(b"MLFCS\x00\x03\n")
    with pytest.raises(ValueError, match="version 4"):
        ForceConstants.load(legacy)


def test_native_roundtrip_preserves_masses_and_revalidates_domain(tmp_path):
    from mlfcs import ForceConstants

    atoms = bulk("Ar", "sc", a=1)
    atoms.set_masses([41.0])
    space = ClusterSpace(atoms, cutoffs={2: 0.1})
    model = ForceConstants(space, {2: np.ones(space.n_parameters)})
    path = model.save(tmp_path / "model.mlfcs")
    restored = ForceConstants.load(path)
    np.testing.assert_array_equal(restored.cluster_space.primitive_atoms.get_masses(), [41.0])
    assert not restored.cluster_space.cell.flags.writeable


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("order", 1),
        ("order", 2.5),
        ("cutoff", 0.0),
        ("cutoff", float("inf")),
        ("cutoff", float("nan")),
        ("max_body_order", 0),
        ("max_body_order", 3),
    ),
)
def test_load_rejects_invalid_truncation_without_neighbor_search(
    tmp_path, monkeypatch, field, value
):
    import h5py
    from mlfcs import ForceConstants

    space = ClusterSpace(bulk("Ar", "sc", a=1), cutoffs={2: 0.1})
    model = ForceConstants(space, {2: np.ones(space.n_parameters)})
    path = model.save(tmp_path / "invalid.mlfcs")
    with h5py.File(path, "r+") as handle:
        handle["cluster_space/blocks/0"].attrs[field] = value

    def no_neighbor_search(*args, **kwargs):
        pytest.fail("restoration must validate stored inputs without neighbor search")

    monkeypatch.setattr("mlfcs.cluster_space.candidates.neighbors", no_neighbor_search)
    with pytest.raises(ValueError):
        ForceConstants.load(path)
