"""Periodic tensor composition, image information and streamed external export."""

import h5py
import numpy as np
import pytest
from ase.build import bulk, make_supercell

from mlfcs import ClusterMap, ClusterSpace, CompactForceConstants, ForceConstants
from mlfcs.force_constants.expansion import expand_lattice_tensors
from mlfcs.geometry.primitive import LatticeSite


@pytest.fixture(scope="module")
def mapped_tensors():
    """Prepare FC2/3/4 on a shuffled non-diagonal supercell with folded aliases."""
    atoms = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(atoms, cutoffs={2: 1.1, 3: 1.1, 4: 0.1})
    model = ForceConstants(
        space,
        {
            p: np.linspace(
                0.1, 0.9, space.block(p).parameters.stop - space.block(p).parameters.start
            )
            for p in (2, 3, 4)
        },
    )
    supercell = make_supercell(atoms, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])[::-1]
    mapping = ClusterMap(space, supercell)
    return model, mapping, CompactForceConstants(model, mapping)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_blocks_and_full_translation_preserve_image_sums(mapped_tensors, order):
    """All tile sizes produce the same folded aliases and relative-origin translation."""
    model, mapping, fc = mapped_tensors
    shape = (model.cluster_space.n_atoms,) + (mapping.n_atoms,) * (order - 1) + (3,) * order
    expected = np.zeros(shape)
    expanded = expand_lattice_tensors(model, order)
    for sites, translations, tensor in zip(expanded.sites, expanded.translations, expanded.tensors):
        tails = tuple(
            mapping.atom_index(LatticeSite(site, t)) for site, t in zip(sites[1:], translations)
        )
        expected[(sites[0], *tails)] += tensor
    actual = np.empty(shape)
    budget = 8 * 3**order * (order + 5) + 8 * (4 * order - 3)
    for slices, tensors in fc.compact_blocks(order, max_bytes=budget):
        actual[slices] = tensors
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(actual, fc.compact_array(order))
    full = fc.full_array(order)
    for atoms in np.ndindex((mapping.n_atoms,) * order):
        first = atoms[0]
        origin = mapping.lattice_translations[first]
        tails = tuple(
            mapping.atom_index(
                LatticeSite(
                    int(mapping.primitive_site_indices[atom]),
                    tuple(
                        int(t) - int(o) for t, o in zip(mapping.lattice_translations[atom], origin)
                    ),
                )
            )
            for atom in atoms[1:]
        )
        np.testing.assert_allclose(
            full[atoms], expected[(mapping.primitive_site_indices[first], *tails)], atol=1e-15
        )


def test_composition_sums_before_threshold_and_preserves_higher_orders(mapped_tensors, tmp_path):
    """Small FC2 contributions sum before cleanup, and absent higher orders add no contribution."""
    model, mapping, short = mapped_tensors
    compact = np.full((1, mapping.n_atoms, 3, 3), 6e-9)
    long = CompactForceConstants({2: compact}, mapping)
    compact[:] = 0
    combined = long + long
    file = combined.write(tmp_path / "fc2.hdf5", format="phonopy", order=2, storage="hdf5")
    with h5py.File(file) as handle:
        np.testing.assert_array_equal(handle["force_constants"][:], 1.2e-8)
        anchor = mapping.atom_index(LatticeSite(0, (0, 0, 0)))
        np.testing.assert_array_equal(handle["p2s_map"][:], [anchor])
    total = short + long
    assert total.orders == model.orders
    np.testing.assert_array_equal(total.compact_array(3), short.compact_array(3))
    with pytest.raises(ValueError, match="folded|support"):
        list(total.lattice_blocks(2))
    total.write(tmp_path / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)


def test_lattice_composition_keeps_individual_images(mapped_tensors):
    """Adding two parameter sources combines matching image labels without folding."""
    _, _, fc = mapped_tensors
    single = list(fc.lattice_blocks(2))
    doubled = list((fc + fc).lattice_blocks(2))
    for left, right in zip(single, doubled):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_allclose(right[2], 2 * left[2], rtol=0, atol=1e-15)


def test_force_and_export_do_not_materialize_arrays(mapped_tensors, tmp_path, monkeypatch):
    """Block consumers compute the same force operator without either array materializer."""
    _, mapping, fc = mapped_tensors
    full = fc.full_array(2)
    u = np.random.default_rng(1).normal(size=(3, mapping.n_atoms, 3))
    expected = -np.einsum("ijab,fjb->fia", full, u)

    def forbidden(*args, **kwargs):
        """Reject accidental whole-array materialization in block consumers."""
        raise AssertionError("a streaming consumer materialized an array")

    monkeypatch.setattr(CompactForceConstants, "full_array", forbidden)
    monkeypatch.setattr(CompactForceConstants, "compact_array", forbidden)
    np.testing.assert_allclose(fc.harmonic_forces(u), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(fc.harmonic_forces(u[0]), expected[0], rtol=1e-14, atol=1e-14)
    file = fc.write(tmp_path / "fc2.hdf5", format="phonopy", order=2, storage="hdf5", threshold=0)
    with h5py.File(file) as handle:
        np.testing.assert_array_equal(handle["force_constants"][:], full)
    fc.write(tmp_path / "FORCE_CONSTANTS", format="phonopy", order=2)


def test_block_budget_and_folded_input_shapes(mapped_tensors):
    """Require complete tensors and a declared compact layout instead of guessing a full array."""
    _, mapping, fc = mapped_tensors
    with pytest.raises(ValueError, match="working"):
        list(fc.compact_blocks(2, max_bytes=1))
    with pytest.raises(ValueError, match="shape"):
        CompactForceConstants({2: np.zeros((mapping.n_atoms, mapping.n_atoms, 3, 3))}, mapping)
    with pytest.raises(ValueError, match="contain order"):
        fc.compact_array(5)
    with pytest.raises(ValueError, match="shape"):
        fc.harmonic_forces(np.zeros((mapping.n_atoms, 2)))
