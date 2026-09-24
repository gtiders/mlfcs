"""Focused contracts for the rewritten supercell data layer."""

from __future__ import annotations

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase.build import bulk

from mlfcs.cluster_space import build_cluster_space
from mlfcs.core import LatticeSite, PrimitiveCell
from mlfcs.core.errors import AliasingError
from mlfcs.supercell import ClusterMap, Supercell


def test_supercell_is_only_structure_and_quotient_data() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    supercell_atoms = atoms.repeat((2, 2, 2))
    cell = Supercell.from_atoms(primitive, supercell_atoms, matrix=np.diag([2, 2, 2]))

    assert internal_dependencies("supercell") == {"cluster_space", "core"}
    assert not hasattr(cell, "orbits")
    assert not hasattr(cell, "cutoff")
    assert cell.atom(LatticeSite(0, (2, -2, 0))) == cell.atom(LatticeSite(0))
    assert len({cell.quotient(tuple(value)) for value in cell.translations}) == 8
    assert len(cell.cell_translations) == 8
    assert len({cell.quotient(value) for value in cell.cell_translations}) == 8
    assert cell.cell_translations == tuple(
        tuple(int(value) for value in translation)
        for site, translation in zip(cell.sites, cell.translations, strict=True)
        if site == 0
    )


def test_cluster_map_reports_supercell_aliasing_and_exact_rank() -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={2: 1.1},
        max_body_orders={2: 2},
    )
    small = Supercell.from_atoms(primitive, atoms, matrix=np.eye(3, dtype=np.int64))
    large = Supercell.from_atoms(
        primitive,
        atoms.repeat((3, 3, 3)),
        matrix=np.diag([3, 3, 3]),
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
