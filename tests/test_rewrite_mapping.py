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
