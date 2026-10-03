"""Differential checks against frozen pre-migration symmetry outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterMap, ClusterSpace


@pytest.mark.parametrize(
    "reference",
    json.loads((Path(__file__).parent / "data" / "symmetry_reference.json").read_text()),
)
def test_reference_orbits_integer_lattice_and_folded_rank(reference):
    sympy = pytest.importorskip("sympy")
    from sympy.matrices.normalforms import hermite_normal_form

    material, order, cutoff = reference["material"], reference["order"], reference["cutoff"]
    atoms = bulk("Si", "diamond", a=5.43) if material == "Si" else bulk("Mg", "hcp", a=3.2, c=5.2)
    primitive = atoms
    space = ClusterSpace(
        primitive,
        cutoffs={order: cutoff},
        max_body_orders={order: order},
        symprec=1e-05,
    )
    assert len(space.orbits) == len(reference["orbits"])
    assert space.n_parameters == reference["n_parameters"]
    for index, orbit in enumerate(space.orbits):
        old = reference["orbits"][index]
        np.testing.assert_array_equal(orbit.representative.labels, old["representative"])
        np.testing.assert_array_equal([c.labels for c in orbit.clusters], old["images"])
        np.testing.assert_array_equal(orbit.operations, old["operations"])
        np.testing.assert_array_equal(orbit.permutations, old["permutations"])
        assert hermite_normal_form(
            sympy.Matrix(orbit.exact_lattice_basis.tolist())
        ) == hermite_normal_form(sympy.Matrix(old["lattice_basis"]))
        old_basis = np.asarray(old["component_basis"], dtype=float)
        q1, _ = np.linalg.qr(orbit.component_basis)
        q2, _ = np.linalg.qr(old_basis)
        np.testing.assert_allclose(q1 @ q1.T, q2 @ q2.T, atol=2e-12)
    for folded_reference in reference["folded_ranks"]:
        repeat = folded_reference["repeat"]
        mapping = ClusterMap(
            space, atoms.repeat((repeat,) * 3), supercell_matrix=np.eye(3, dtype=np.int64) * repeat
        )
        parameters, rank, aliases = folded_reference["values"]
        info = mapping.rank_info(order)
        assert (info.parameters, info.rank, info.aliases) == (parameters, rank, aliases)
