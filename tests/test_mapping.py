"""Numerical behavior of mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterMap, ClusterSpace
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.mapping.periodic import PeriodicIndex, map_labels


def test_cluster_space_maps_to_multiple_supercells():
    """Verify cluster space maps to multiple supercells."""
    atoms = bulk("Ar", "sc", a=1)
    space = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 1.1}, max_body_orders={2: 2})
    small = ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large = ClusterMap(
        space, atoms.repeat((3,) * 3), supercell_matrix=3 * np.eye(3, dtype=np.int64)
    )
    assert small.rank_info().nullity > 0
    assert large.rank_info().full
    assert not space.scaled_positions.flags.writeable
    assert not space.symmetry.rotations.flags.writeable
    assert not large.image_atom_indices[0].flags.writeable


def test_structure_snapshots_preserve_geometry_and_masses():
    """Verify structure snapshots preserve geometry and masses."""
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
    assert space.orders == (2,)
    assert space.block(2).max_body_order == 2
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
        assert value.flags.c_contiguous and (not value.flags.writeable)


def test_supercell_quotients_identify_periodic_sites() -> None:
    """Verify supercell quotients identify periodic sites."""
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
    supercell_atoms = atoms.repeat((2, 2, 2))
    cell = ClusterMap(primitive, supercell_atoms, supercell_matrix=np.diag([2, 2, 2]))
    assert cell.atom_index(LatticeSite(0, (2, -2, 0))) == cell.atom_index(LatticeSite(0))
    assert len({cell.quotient(tuple(value)) for value in cell.lattice_translations}) == 8
    assert len(cell.translation_representatives) == 8
    assert len({cell.quotient(value) for value in cell.translation_representatives}) == 8
    assert cell.translation_representatives == tuple(
        (
            tuple((int(value) for value in translation))
            for site, translation in zip(
                cell.primitive_site_indices, cell.lattice_translations, strict=True
            )
            if site == 0
        )
    )


def test_cluster_map_reports_supercell_aliasing_and_rank() -> None:
    """Verify cluster map reports supercell aliasing and rank."""
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = atoms
    space = ClusterSpace(primitive, cutoffs={2: 1.1}, max_body_orders={2: 2}, symprec=1e-05)
    small_map = ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large_map = ClusterMap(space, atoms.repeat((3, 3, 3)), supercell_matrix=np.diag([3, 3, 3]))
    assert small_map.rank_info(2).nullity > 0
    assert large_map.rank_info(2).full
    large_map.rank_info(2).require_full()
    assert len(small_map.aliases(2)) > len(large_map.aliases(2))


def test_mapped_labels_maps_large_cancelling_coordinates() -> None:
    """Verify mapped labels maps large cancelling coordinates."""
    adjugate = np.eye(3, dtype=np.int64)
    adjugate[1, 0] = 1
    prepared = PeriodicIndex(
        adjugate=adjugate,
        modulus=1,
        keys=np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        atom_indices=np.asarray([0], dtype=np.int64),
    )
    magnitude = 2**61
    labels = np.asarray(
        [[0, magnitude, magnitude, 0], [0, -magnitude, -magnitude, 0]], dtype=np.int64
    )
    translations = np.asarray(
        [[magnitude, -magnitude, 0], [-magnitude, magnitude, 0]], dtype=np.int64
    )
    np.testing.assert_array_equal(map_labels(labels, translations, prepared), 0)


def test_supercell_matrix_inference_handles_skewed_primitive_cell() -> None:
    """Verify supercell matrix inference handles skewed primitive cell."""
    primitive_atoms = bulk("Ar", "sc", a=1.0)
    primitive_atoms.set_cell([[1.0, 0.0, 0.0], [100.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    space = ClusterSpace(primitive_atoms, cutoffs={2: 0.001}, symprec=0.01)
    supercell_atoms = primitive_atoms.copy()
    cell = supercell_atoms.cell.array.copy()
    cell[0, 1] += 0.006
    supercell_atoms.set_cell(cell)
    mapped = ClusterMap(space, supercell_atoms)
    np.testing.assert_array_equal(mapped.supercell_matrix, np.eye(3, dtype=np.int64))


@pytest.mark.parametrize(
    "reference",
    json.loads((Path(__file__).parent / "data" / "symmetry_reference.json").read_text()),
    ids=lambda record: f"{record['material']}-FC{record['order']}",
)
def test_folded_rank_matches_saved_symmetry_reference(reference):
    """Verify folded rank matches saved symmetry reference."""
    material, order, cutoff = (reference["material"], reference["order"], reference["cutoff"])
    atoms = bulk("Si", "diamond", a=5.43) if material == "Si" else bulk("Mg", "hcp", a=3.2, c=5.2)
    primitive = atoms
    space = ClusterSpace(
        primitive, cutoffs={order: cutoff}, max_body_orders={order: order}, symprec=1e-05
    )
    for folded_reference in reference["folded_ranks"]:
        repeat = folded_reference["repeat"]
        mapping = ClusterMap(
            space, atoms.repeat((repeat,) * 3), supercell_matrix=np.eye(3, dtype=np.int64) * repeat
        )
        parameters, rank, aliases = folded_reference["values"]
        info = mapping.rank_info(order)
        assert (info.parameters, info.rank, info.aliases) == (parameters, rank, aliases)
