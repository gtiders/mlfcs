"""Numerical behavior of mapping."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterMap, ClusterSpace
from mlfcs.geometry.lattice import LatticeSite
from mlfcs.geometry.structure import PeriodicIndex


def test_cluster_space_rejects_a_supercell_that_cannot_distinguish_parameters():
    """Verify cluster space maps identifiable supercells and rejects folded ones."""
    atoms = bulk("Ar", "sc", a=1)
    space = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 1.1}, max_body_orders={2: 2})
    with pytest.raises(ValueError, match="cannot distinguish"):
        ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large = ClusterMap(
        space, atoms.repeat((3,) * 3), supercell_matrix=3 * np.eye(3, dtype=np.int64)
    )
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
    cell_snapshot = space.cell_atoms
    supercell_snapshot = mapping.supercell
    np.testing.assert_array_equal(cell_snapshot.get_masses(), [42.0])
    np.testing.assert_array_equal(supercell_snapshot.get_masses(), [42.0, 42.0])
    cell_snapshot.positions[:] = 7.0
    cell_snapshot.set_masses([9.0])
    supercell_snapshot.positions[:] = 8.0
    supercell_snapshot.set_masses([10.0, 10.0])
    np.testing.assert_array_equal(space.cell_atoms.get_masses(), [42.0])
    np.testing.assert_array_equal(mapping.supercell.get_masses(), [42.0, 42.0])
    assert not np.allclose(space.cell_atoms.positions, 7.0)
    assert not np.allclose(mapping.supercell.positions, 8.0)
    assert space.orders == (2,)
    assert space.block(2).max_body_order == 2
    for value in (
        space.atomic_numbers,
        mapping.supercell_matrix,
        mapping.atomic_numbers,
        mapping.cell_site_indices,
        mapping.lattice_translations,
        mapping.quotient_labels,
        *mapping.image_atom_indices,
    ):
        assert value.dtype == np.int64
        assert value.flags.c_contiguous and (not value.flags.writeable)


def test_supercell_quotients_identify_periodic_sites() -> None:
    """Verify supercell quotients identify periodic sites."""
    atoms = bulk("Ar", "sc", a=1.0)
    cell = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
    supercell = atoms.repeat((2, 2, 2))
    cell = ClusterMap(cell, supercell, supercell_matrix=np.diag([2, 2, 2]))
    assert cell.atom_index(LatticeSite(0, (2, -2, 0))) == cell.atom_index(LatticeSite(0))
    assert len({cell.quotient(tuple(value)) for value in cell.lattice_translations}) == 8
    assert len(cell.translation_representatives) == 8
    assert len({cell.quotient(value) for value in cell.translation_representatives}) == 8
    assert cell.translation_representatives == tuple(
        (
            tuple((int(value) for value in translation))
            for site, translation in zip(
                cell.cell_site_indices, cell.lattice_translations, strict=True
            )
            if site == 0
        )
    )


def test_cluster_map_rejection_reports_the_folded_directions() -> None:
    """Verify the aliasing rejection names the rank loss of a folded supercell."""
    atoms = bulk("Ar", "sc", a=1.0)
    cell = atoms
    space = ClusterSpace(cell, cutoffs={2: 1.1}, max_body_orders={2: 2}, symprec=1e-05)
    with pytest.raises(ValueError, match=r"rank \d+ for \d+ parameters"):
        ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large_map = ClusterMap(space, atoms.repeat((3, 3, 3)), supercell_matrix=np.diag([3, 3, 3]))
    assert large_map.n_atoms == 27


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
    np.testing.assert_array_equal(prepared.lookup(labels, translations), 0)


def test_supercell_matrix_inference_handles_skewed_cell() -> None:
    """Verify supercell matrix inference handles skewed cell."""
    cell_atoms = bulk("Ar", "sc", a=1.0)
    cell_atoms.set_cell([[1.0, 0.0, 0.0], [100.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    space = ClusterSpace(cell_atoms, cutoffs={2: 0.001}, symprec=0.01)
    supercell = cell_atoms.copy()
    cell = supercell.cell.array.copy()
    cell[0, 1] += 0.006
    supercell.set_cell(cell)
    mapped = ClusterMap(space, supercell)
    np.testing.assert_array_equal(mapped.supercell_matrix, np.eye(3, dtype=np.int64))
