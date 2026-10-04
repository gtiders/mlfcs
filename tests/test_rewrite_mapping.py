"""Focused contracts for the rewritten supercell data layer."""

from __future__ import annotations

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase.build import bulk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import LatticeSite
from mlfcs.errors import AliasingError
from mlfcs.mapping import ClusterMap
from mlfcs.mapping.geometry import _PeriodicIndex, mapped_labels


def test_supercell_is_only_structure_and_quotient_data() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
    supercell_atoms = atoms.repeat((2, 2, 2))
    cell = ClusterMap(primitive, supercell_atoms, supercell_matrix=np.diag([2, 2, 2]))

    assert internal_dependencies("mapping") == {
        "errors",
        "core",
        "_arrays",
        "cluster_space",
        "algebra",
    }
    assert not hasattr(cell, "orbits")
    assert not hasattr(cell, "cutoff")
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


def test_cluster_map_reports_supercell_aliasing_and_exact_rank() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = atoms
    space = ClusterSpace(primitive, cutoffs={2: 1.1}, max_body_orders={2: 2}, symprec=1e-05)
    small_map = ClusterMap(space, atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    large_map = ClusterMap(space, atoms.repeat((3, 3, 3)), supercell_matrix=np.diag([3, 3, 3]))

    assert small_map.rank_info(2).nullity > 0
    assert large_map.rank_info(2).full
    with pytest.raises(AliasingError, match="nullity"):
        small_map.rank_info(2).require_full()
    large_map.rank_info(2).require_full()
    assert len(small_map.aliases(2)) > len(large_map.aliases(2))
    assert len(small_map.fingerprint) == 64


def test_mapped_labels_bounds_the_sum_before_numba_addition() -> None:
    prepared = _PeriodicIndex(
        adjugate=np.eye(3, dtype=np.int64),
        modulus=3,
        keys=np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        atom_indices=np.asarray([0], dtype=np.int64),
    )
    labels = np.asarray([[0, 2**62, 0, 0]], dtype=np.int64)
    translations = np.asarray([[2**62, 0, 0]], dtype=np.int64)

    with pytest.raises(OverflowError, match="translated labels"):
        mapped_labels(labels, translations, prepared)


def test_mapped_labels_locally_checks_safe_cancelling_dot_products() -> None:
    adjugate = np.zeros((3, 3), dtype=np.int64)
    adjugate[:2, 0] = 1
    prepared = _PeriodicIndex(
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

    np.testing.assert_array_equal(mapped_labels(labels, translations, prepared), 0)


def test_supercell_matrix_inference_handles_skewed_primitive_cell() -> None:
    primitive_atoms = bulk("Ar", "sc", a=1.0)
    primitive_atoms.set_cell([[1.0, 0.0, 0.0], [100.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    space = ClusterSpace(
        primitive_atoms,
        cutoffs={2: 0.001},
        symprec=0.01,
    )
    supercell_atoms = primitive_atoms.copy()
    cell = supercell_atoms.cell.array.copy()
    cell[0, 1] += 0.006
    supercell_atoms.set_cell(cell)

    mapped = ClusterMap(space, supercell_atoms)

    np.testing.assert_array_equal(mapped.supercell_matrix, np.eye(3, dtype=np.int64))


@pytest.mark.parametrize(
    ("translation", "factors"),
    (
        ((2**62, 0, 0), (2, 0, 0)),
        ((2**62, 2**62, -(2**62)), (1, 1, 1)),
        ((-(2**62), -(2**62), 2**62), (1, 1, 1)),
    ),
)
def test_mapping_rejects_actual_product_or_partial_sum_overflow(translation, factors):
    adjugate = np.zeros((3, 3), dtype=np.int64)
    adjugate[:, 0] = factors
    index = _PeriodicIndex(
        adjugate=adjugate,
        modulus=1,
        keys=np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        atom_indices=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(OverflowError, match="periodic quotient intermediate"):
        mapped_labels(
            np.asarray([[0, *translation]], dtype=np.int64), np.zeros((1, 3), dtype=np.int64), index
        )


def test_periodic_matching_enumerates_all_images_inside_tolerance():
    from mlfcs.core.geometry import PeriodicGeometry

    geometry = PeriodicGeometry(np.eye(3))
    candidates, shifts = geometry.matching_images(
        np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]), tolerance=0.6
    )
    assert candidates.tolist() == [0, 0, 1]
    assert {tuple(row) for row in shifts[:2]} == {(-1, 0, 0), (0, 0, 0)}
    np.testing.assert_array_equal(shifts[2], [0, 0, 0])


def test_periodic_matching_uses_strict_radius_and_handles_no_candidates():
    from mlfcs.core.geometry import PeriodicGeometry

    geometry = PeriodicGeometry(np.eye(3))
    for vectors in (np.empty((0, 3)), np.asarray([[0.5, 0.0, 0.0]])):
        candidates, shifts = geometry.matching_images(vectors, tolerance=0.5)
        assert candidates.shape == (0,)
        assert shifts.shape == (0, 3)
