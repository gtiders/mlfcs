"""Focused contracts for the rewritten supercell data layer."""

from __future__ import annotations

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import LatticeSite, PrimitiveCell, PrimitiveSymmetry
from mlfcs.core.log_error import AliasingError
from mlfcs.supercell import ClusterMap, Supercell
from mlfcs.tools.supercell import build_supercell


def test_default_symprec_flows_into_inferred_supercell_mapping() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms)
    assert primitive.symprec == 1e-5
    assert PrimitiveSymmetry.from_primitive(primitive).symprec == primitive.symprec

    matrix = np.asarray([[2, 1, 0], [0, 2, 0], [0, 0, 1]])
    expanded = build_supercell(atoms, matrix)
    order = np.arange(len(expanded))[::-1]
    reordered = expanded[order]
    supercell = Supercell.from_atoms(primitive, reordered)
    assert supercell.pbc == primitive.pbc
    np.testing.assert_array_equal(supercell.matrix, matrix)
    np.testing.assert_array_equal(supercell.cell, reordered.cell)
    np.testing.assert_array_equal(supercell.sites, np.zeros(len(reordered), dtype=int))
    assert len({tuple(q) for q in supercell.quotients}) == len(reordered)

    slightly_changed = reordered.copy()
    slightly_changed.cell[0, 0] += 0.5e-5
    Supercell.from_atoms(primitive, slightly_changed)
    slightly_changed.cell[0, 0] += 1.5e-5
    with pytest.raises(ValueError, match="lattice residual.*symprec"):
        Supercell.from_atoms(primitive, slightly_changed)


def test_supercell_quotient_data() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    supercell_atoms = atoms.repeat((2, 2, 2))
    cell = Supercell.from_atoms(primitive, supercell_atoms)

    assert internal_dependencies("supercell") == {"cluster_space", "core"}
    assert cell.atom(LatticeSite(0, (2, -2, 0))) == cell.atom(LatticeSite(0))
    assert len({cell.quotient(tuple(value)) for value in cell.translations}) == 8
    assert len(cell.cell_translations) == 8
    assert len({cell.quotient(value) for value in cell.cell_translations}) == 8
    assert cell.cell_translations == tuple(
        tuple(int(value) for value in translation)
        for site, translation in zip(cell.sites, cell.translations, strict=True)
        if site == 0
    )


def test_supercell_inherits_pbc_without_reinterpreting_ase_flags() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms)
    expanded = atoms.repeat((2, 1, 1))
    expanded.pbc = (True, False, False)

    mapped = Supercell.from_atoms(primitive, expanded)
    assert mapped.pbc == primitive.pbc
    np.testing.assert_array_equal(mapped.matrix, np.diag([2, 1, 1]))


def test_sheared_supercell_mapping_uses_the_cartesian_nearest_image() -> None:
    cell = np.asarray([[1.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=0.1)
    shifted = atoms.copy()
    shifted.positions[0, 1] = 0.055

    supercell = Supercell.from_atoms(primitive, shifted)
    np.testing.assert_array_equal(supercell.translations, [[0, 0, 0]])


def test_sheared_supercell_inference_accepts_cartesian_close_cell() -> None:
    cell = np.asarray([[1.0, 0.0, 0.0], [100.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=0.01)
    expanded = atoms.repeat((2, 1, 1))
    expanded.cell[0, 1] += 0.006
    expected = np.diag([2, 1, 1])

    assert np.max(np.linalg.norm(expected @ cell - expanded.cell.array, axis=1)) < primitive.symprec
    mapped = Supercell.from_atoms(primitive, expanded)
    np.testing.assert_array_equal(mapped.matrix, expected)


def test_cluster_map_reports_supercell_aliasing_and_exact_rank() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = ClusterSpace(
        primitive.to_atoms(),
        symprec=primitive.symprec,
        cutoffs={2: 1.1},
        max_body_orders={2: 2},
    )
    assert space.pbc == primitive.pbc
    small = Supercell.from_atoms(primitive, atoms)
    large = Supercell.from_atoms(
        primitive,
        atoms.repeat((3, 3, 3)),
    )
    small_map = ClusterMap.build(space, small)
    large_map = ClusterMap.build(space, large)

    assert small_map.rank_info(2).nullity > 0
    assert large_map.rank_info(2).full
    with pytest.raises(AliasingError, match="nullity"):
        small_map.rank_info(2).require_full()
    large_map.rank_info(2).require_full()
    assert len(small_map.aliases(2)) > len(large_map.aliases(2))
    assert len(small_map.fingerprint) == 64
