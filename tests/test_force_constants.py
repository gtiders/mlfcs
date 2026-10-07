"""Numerical behavior of force constants."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from ase.build import bulk

from mlfcs import ClusterMap, ClusterSpace
from mlfcs.fitting.design import ForceDesign
from mlfcs.force_constants import ForceConstants


def test_native_reference_preserves_model_values():
    """Verify native reference preserves model values."""
    from mlfcs import ForceConstants

    model = ForceConstants.load(Path(__file__).parent / "data" / "force_constants_v5.mlfcs")
    assert model.cluster_space.orbits[0].lattice_basis.dtype == np.int64
    assert not model.cluster_space.orbits[0].lattice_basis.flags.writeable
    np.testing.assert_array_equal(model.parameters(), 1.0)
    space = model.cluster_space
    mapping = ClusterMap(
        space,
        space.primitive_atoms.repeat((3,) * 3),
        supercell_matrix=3 * np.eye(3, dtype=np.int64),
    )
    design = ForceDesign(mapping)
    np.testing.assert_array_equal(design.matrix(np.zeros((len(mapping.supercell_atoms), 3))), 0.0)


def assert_model_arrays_equal(left, right):
    """Compare the complete persisted numerical model without hashes or tolerances."""
    a, b = (left.cluster_space, right.cluster_space)
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
            "lattice_basis",
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
    """Construct a cubic Ar model with FC2 pair terms and on-site FC3 terms.

    The primitive cell has one atom and lattice length one angstrom. Cutoffs
    are 1.1 angstrom for FC2 and 0.1 angstrom for FC3."""
    primitive = bulk("Ar", "sc", a=1.0)
    return ClusterSpace(
        primitive, cutoffs={2: 1.1, 3: 0.1}, max_body_orders={2: 2, 3: 1}, symprec=1e-05
    )


def values(cluster_space, order: int, value: float) -> np.ndarray:
    """Return a constant physical parameter vector for one tensor order.

    Its length equals the order block parameter count. Entries equal value
    and are interpreted in eV/angstrom**order."""
    block = cluster_space.block(order)
    return np.full(block.parameters.stop - block.parameters.start, value)


def test_force_constants_are_primitive_immutable_and_round_trip(tmp_path) -> None:
    """Verify force constants are primitive immutable and round trip."""
    cluster_space = space()
    source = values(cluster_space, 2, 2.0)
    model = ForceConstants(cluster_space, {2: source})
    source[:] = 7.0
    assert model.orders == (2,)
    assert not model.coefficients[2].flags.writeable
    np.testing.assert_array_equal(model.coefficients[2], 2.0)
    file = model.save(tmp_path / "fc.mlfcs")
    restored = ForceConstants.load(file)
    assert_model_arrays_equal(model, restored)


def test_native_roundtrip_preserves_orders_and_masses(tmp_path):
    """Preserve all model arrays, physical coefficients and custom masses in a v5 roundtrip."""
    cluster_space = space().with_masses([41.0])
    model = ForceConstants(cluster_space, {p: values(cluster_space, p, p) for p in (2, 3)})
    file = model.save(tmp_path / "multi.mlfcs")
    assert_model_arrays_equal(model, ForceConstants.load(file))
    with h5py.File(file) as handle:
        assert handle.attrs["version"] == 5
        assert set(handle.attrs) == {"format", "version"}
        assert set(handle["coefficients"]) == {"2", "3"}
        for orbit in handle["cluster_space/orbits"].values():
            assert "lattice_basis" in orbit


def test_force_constants_combine_disjoint_orders_only() -> None:
    """Verify force constants combine disjoint orders only."""
    cluster_space = space()
    second = ForceConstants(cluster_space, {2: values(cluster_space, 2, 2.0)})
    third = ForceConstants(cluster_space, {3: values(cluster_space, 3, 3.0)})
    combined = ForceConstants.combine([second, third])
    assert combined.orders == (2, 3)
    np.testing.assert_array_equal(
        combined.parameters(), np.concatenate([second.parameters(), third.parameters()])
    )


def mapped_model(orders: tuple[int, ...], *, repeats: int = 2) -> tuple[ForceConstants, ClusterMap]:
    """Construct a cubic Ar force-constant model and repeated supercell map.

    orders selects tensor ranks, and repeats is the repetition on each axis.
    Coefficients span 0.5e-8 to 2e-8 in the corresponding physical units."""
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
            5e-09,
            2e-08,
            cluster_space.block(order).parameters.stop
            - cluster_space.block(order).parameters.start,
        )
        for order in orders
    }
    return (
        ForceConstants(cluster_space, coefficients),
        ClusterMap(
            cluster_space,
            atoms.repeat((repeats, repeats, repeats)),
            supercell_matrix=np.diag([repeats, repeats, repeats]),
        ),
    )


def test_phonopy_writes_text_and_hdf5_with_one_threshold(tmp_path) -> None:
    """Verify phonopy writes text and hdf5 with one threshold."""
    model, mapping = mapped_model((2,))
    text = model.write(
        tmp_path / "FORCE_CONSTANTS", mapping, format="phonopy", order=2, storage="text"
    )
    assert text.read_text().splitlines()[0].split() == ["8", "8"]
    numbers = [float(value) for line in text.read_text().splitlines()[2:] for value in line.split()]
    assert all(value == 0.0 or abs(value) >= 1e-08 for value in numbers)
    binary = model.write(
        tmp_path / "force_constants.hdf5", mapping, format="phonopy", order=2, storage="hdf5"
    )
    with h5py.File(binary, "r") as handle:
        values = handle["force_constants"][:]
        assert values.shape == (8, 8, 3, 3)
        assert handle.attrs["mlfcs_threshold"] == 1e-08
        assert np.all((values == 0.0) | (np.abs(values) >= 1e-08))


def test_phono3py_is_fc3_hdf5_and_shengbte_is_filtered_text(tmp_path) -> None:
    """Verify phono3py is fc3 hdf5 and shengbte is filtered text."""
    model, mapping = mapped_model((3,), repeats=3)
    binary = model.write(tmp_path / "fc3.hdf5", mapping, format="phono3py", order=3)
    with h5py.File(binary, "r") as handle:
        values = handle["fc3"][:]
        assert values.shape == (27, 27, 27, 3, 3, 3)
        assert "force_constants" not in handle
        assert np.any(values)
        assert np.all((values == 0.0) | (np.abs(values) >= 1e-08))
    text = model.write(tmp_path / "FORCE_CONSTANTS_3RD", mapping, format="shengbte", order=3)
    assert int(text.read_text().splitlines()[0]) > 0
