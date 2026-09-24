"""Small contracts for primitive force-constant models."""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest
from ase.build import bulk

from mlfcs.cluster_space import build_cluster_space
from mlfcs.core import PrimitiveCell
from mlfcs.force_constants import ForceConstants
from mlfcs.supercell import ClusterMap, Supercell


def space():
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    return build_cluster_space(
        primitive,
        cutoffs={2: 1.1, 3: 0.1},
        max_body_orders={2: 2, 3: 1},
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

    copied = pickle.loads(pickle.dumps(model))
    assert copied.fingerprint == model.fingerprint

    file = model.save(tmp_path / "fc.mlfcs")
    restored = ForceConstants.load(file)
    assert restored.fingerprint == model.fingerprint
    np.testing.assert_array_equal(restored.parameters(), model.parameters())


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

    assert internal_dependencies("force_constants") == {"cluster_space", "core", "supercell"}
    assert internal_dependencies("finite_difference") == {
        "core",
        "force_constants",
        "supercell",
    }


def mapped_model(orders: tuple[int, ...], *, repeats: int = 2) -> tuple[ForceConstants, ClusterMap]:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    cluster_space = build_cluster_space(
        primitive,
        cutoffs={order: 1.1 for order in orders},
        max_body_orders={order: order for order in orders},
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
    cell = Supercell.from_atoms(
        primitive,
        atoms.repeat((repeats, repeats, repeats)),
        matrix=np.diag([repeats, repeats, repeats]),
    )
    return ForceConstants(cluster_space, coefficients), ClusterMap.build(cluster_space, cell)


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
