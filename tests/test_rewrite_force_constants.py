"""Small contracts for primitive force-constant models."""

from __future__ import annotations


import h5py
import numpy as np
import pytest
from ase.build import bulk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.mapping import ClusterMap


def assert_model_arrays_equal(left, right):
    """Compare the complete persisted numerical model without hashes or tolerances."""
    a, b = left.cluster_space, right.cluster_space
    for name in ("cell", "scaled_positions", "atomic_numbers", "masses"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert not getattr(b, name).flags.writeable
    assert a.symprec == b.symprec
    assert a.blocks == b.blocks
    assert a.symmetry.symbol == b.symmetry.symbol
    assert a.symmetry.symprec == b.symmetry.symprec
    for name in (
        "rotations",
        "translations",
        "cartesian_rotations",
        "site_permutations",
        "site_shifts",
    ):
        np.testing.assert_array_equal(getattr(a.symmetry, name), getattr(b.symmetry, name))
        assert not getattr(b.symmetry, name).flags.writeable
    for x, y in zip(a.orbits, b.orbits, strict=True):
        assert x.representative == y.representative
        assert x.clusters == y.clusters
        assert x.observation_condition == y.observation_condition
        for name in (
            "exact_lattice_basis",
            "component_basis",
            "observation_rows",
            "operations",
            "permutations",
        ):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
            assert not getattr(y, name).flags.writeable
    assert left.orders == right.orders
    for order in left.orders:
        np.testing.assert_array_equal(left.coefficients[order], right.coefficients[order])
        assert not right.coefficients[order].flags.writeable


def space():
    primitive = bulk("Ar", "sc", a=1.0)
    return ClusterSpace(
        primitive,
        cutoffs={2: 1.1, 3: 0.1},
        max_body_orders={2: 2, 3: 1},
        symprec=1e-05,
    )


def values(cluster_space, order: int, value: float) -> np.ndarray:
    block = cluster_space.block(order)
    return np.full(block.parameters.stop - block.parameters.start, value)


def test_force_constants_are_primitive_immutable_and_round_trip(tmp_path) -> None:
    cluster_space = space()
    source = values(cluster_space, 2, 2.0)
    model = ForceConstants(cluster_space, {2: source})
    source[:] = 7.0

    assert model.orders == (2,)
    assert not model.coefficients[2].flags.writeable
    np.testing.assert_array_equal(model.coefficients[2], 2.0)
    assert not hasattr(model, "supercell")
    with pytest.raises(TypeError):
        model.coefficients[3] = np.zeros(1)

    file = model.save(tmp_path / "fc.mlfcs")
    restored = ForceConstants.load(file)
    assert_model_arrays_equal(model, restored)


def test_hdf5_multi_order_roundtrip_without_model_reconstruction(tmp_path, monkeypatch):
    """Preserve all model buffers and custom masses without rebuilding any orbit."""
    cluster_space = space().with_masses([41.0])
    model = ForceConstants(cluster_space, {p: values(cluster_space, p, p) for p in (2, 3)})
    file = model.save(tmp_path / "multi.mlfcs")

    def forbid_rebuild(*args, **kwargs):
        """Fail if native restoration tries to rebuild derived model state."""
        pytest.fail("native loading must not reconstruct the cluster space")

    monkeypatch.setattr("mlfcs.cluster_space.builder._construct_space", forbid_rebuild)
    monkeypatch.setattr("mlfcs.cluster_space.candidates.neighbors", forbid_rebuild)
    assert_model_arrays_equal(model, ForceConstants.load(file))
    with h5py.File(file) as handle:
        assert handle.attrs["version"] == 4
        assert set(handle.attrs) == {"format", "version"}
        assert set(handle["coefficients"]) == {"2", "3"}


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_native_save_failure_preserves_target_and_cleans_temporary(
    tmp_path, monkeypatch, failure_stage
):
    """A failed write or replacement must leave the prior file intact with no scratch files."""
    from mlfcs.force_constants import _native

    cluster_space = space()
    model = ForceConstants(cluster_space, {2: values(cluster_space, 2, 1.0)})
    file = model.save(tmp_path / "prior.mlfcs")
    original = file.read_bytes()

    def fail(*args, **kwargs):
        """Inject an I/O failure at a specified stage of the atomic write."""
        raise OSError("injected failure")

    if failure_stage == "write":
        monkeypatch.setattr(_native, "_write_space", fail)
    else:
        monkeypatch.setattr(_native.os, "replace", fail)
    with pytest.raises(OSError, match="injected failure"):
        model.save(file)
    assert file.read_bytes() == original
    assert list(tmp_path.iterdir()) == [file]


@pytest.mark.parametrize(
    "corruption", ["version", "missing", "dtype", "shape", "mass", "position", "coefficient"]
)
def test_native_load_rejects_invalid_stored_models(tmp_path, corruption):
    """Reject malformed version, geometry, species and coefficient data before use."""
    cluster_space = space()
    model = ForceConstants(cluster_space, {2: values(cluster_space, 2, 1.0)})
    file = model.save(tmp_path / "invalid.mlfcs")
    with h5py.File(file, "r+") as handle:
        primitive = handle["cluster_space/primitive"]
        if corruption == "version":
            handle.attrs["version"] = 3
        elif corruption == "missing":
            del primitive["masses"]
        elif corruption == "dtype":
            del primitive["atomic_numbers"]
            primitive.create_dataset("atomic_numbers", data=np.array([18.0]))
        elif corruption == "shape":
            del primitive["cell"]
            primitive.create_dataset("cell", data=np.ones((2, 3)))
        elif corruption == "mass":
            primitive["masses"][0] = 0.0
        elif corruption == "position":
            primitive["scaled_positions"][0, 0] = 1.0
        else:
            handle["coefficients/2"][0] = np.nan
    with pytest.raises(ValueError):
        ForceConstants.load(file)


def test_force_constants_combine_disjoint_orders_only() -> None:
    cluster_space = space()
    second = ForceConstants(cluster_space, {2: values(cluster_space, 2, 2.0)})
    third = ForceConstants(cluster_space, {3: values(cluster_space, 3, 3.0)})

    combined = ForceConstants.combine([second, third])
    assert combined.orders == (2, 3)
    np.testing.assert_array_equal(
        combined.parameters(), np.concatenate([second.parameters(), third.parameters()])
    )
    with pytest.raises(ValueError, match="duplicated"):
        ForceConstants.combine([second, second])
    with pytest.raises(KeyError, match="do not contain"):
        second.parameters((3,))


def test_force_constants_reject_wrong_or_nonfinite_parameters() -> None:
    cluster_space = space()
    correct = values(cluster_space, 2, 0.0)
    with pytest.raises(ValueError, match="length"):
        ForceConstants(cluster_space, {2: correct[:-1]})
    correct[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ForceConstants(cluster_space, {2: correct})


def test_new_package_dependency_direction_is_explicit() -> None:
    from _architecture_helpers import internal_dependencies

    assert internal_dependencies("force_constants") == {
        "_arrays",
        "errors",
        "core",
        "cluster_space",
        "mapping",
    }
    assert internal_dependencies("finite_difference") == {"core", "force_constants", "mapping"}


def mapped_model(orders: tuple[int, ...], *, repeats: int = 2) -> tuple[ForceConstants, ClusterMap]:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = atoms
    cluster_space = ClusterSpace(
        primitive,
        cutoffs={order: 1.1 for order in orders},
        max_body_orders={order: order for order in orders},
        symprec=1e-05,
    )
    coefficients = {
        order: np.linspace(
            0.5e-8,
            2.0e-8,
            cluster_space.block(order).parameters.stop
            - cluster_space.block(order).parameters.start,
        )
        for order in orders
    }
    return ForceConstants(cluster_space, coefficients), ClusterMap(
        cluster_space,
        atoms.repeat((repeats, repeats, repeats)),
        supercell_matrix=np.diag([repeats, repeats, repeats]),
    )


def test_phonopy_writes_text_and_hdf5_with_one_threshold(tmp_path) -> None:
    model, mapping = mapped_model((2,))
    text = model.write(
        tmp_path / "FORCE_CONSTANTS",
        mapping,
        format="phonopy",
        order=2,
        storage="text",
    )
    assert text.read_text().splitlines()[0].split() == ["8", "8"]
    numbers = [float(value) for line in text.read_text().splitlines()[2:] for value in line.split()]
    assert all(value == 0.0 or abs(value) >= 1e-8 for value in numbers)

    binary = model.write(
        tmp_path / "force_constants.hdf5",
        mapping,
        format="phonopy",
        order=2,
        storage="hdf5",
    )
    with h5py.File(binary, "r") as handle:
        values = handle["force_constants"][:]
        assert values.shape == (8, 8, 3, 3)
        assert handle.attrs["mlfcs_threshold"] == 1e-8
        assert np.all((values == 0.0) | (np.abs(values) >= 1e-8))


def test_phono3py_is_fc3_hdf5_and_shengbte_is_filtered_text(tmp_path) -> None:
    model, mapping = mapped_model((3,), repeats=3)
    binary = model.write(tmp_path / "fc3.hdf5", mapping, format="phono3py", order=3)
    with h5py.File(binary, "r") as handle:
        values = handle["fc3"][:]
        assert values.shape == (27, 27, 27, 3, 3, 3)
        assert "force_constants" not in handle
        assert np.any(values)
        assert np.all((values == 0.0) | (np.abs(values) >= 1e-8))

    text = model.write(tmp_path / "FORCE_CONSTANTS_3RD", mapping, format="shengbte", order=3)
    assert int(text.read_text().splitlines()[0]) > 0
    with pytest.raises(ValueError, match="only order 3"):
        model.write(tmp_path / "wrong.hdf5", mapping, format="phono3py", order=2)
    with pytest.raises(ValueError, match="supported formats"):
        model.write(tmp_path / "legacy.xml", mapping, format="alamode", order=3)
