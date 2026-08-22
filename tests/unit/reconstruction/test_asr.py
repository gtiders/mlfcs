import numpy as np
import pytest
from ase.build import bulk

from mlfcs.core.expansion import expand_orbit_parameters
from mlfcs.core.geometry import PeriodicGeometry, make_supercell, resolve_cutoff
from mlfcs.core.orbits import build_orbit_space
from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.reconstruction.asr import project_sum_rules
from mlfcs.reconstruction.solver import reconstruct_sparse


@pytest.mark.parametrize("order", [3, 4])
def test_acoustic_sum_rule_projection_is_strict(order):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=order,
        cutoff=resolve_cutoff(supercell, index, -1),
    )
    rng = np.random.default_rng(4)
    derivatives = {key: rng.normal(size=(len(supercell), 3)) for key in space.displacement_keys}
    raw = reconstruct_sparse(space, index, derivatives, enforce_asr=False).to_dense()
    projected = reconstruct_sparse(space, index, derivatives, enforce_asr=True).to_dense()
    raw_residual = np.linalg.norm(raw.sum(axis=order - 1))
    projected_residual = np.linalg.norm(projected.sum(axis=order - 1))
    assert projected_residual < raw_residual
    assert projected_residual < 1e-10


def test_asr_reports_phonopy_style_maximum_drift():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=3,
        cutoff=resolve_cutoff(supercell, index, -1),
    )
    rng = np.random.default_rng(8)
    derivatives = {key: rng.normal(size=(len(supercell), 3)) for key in space.displacement_keys}
    messages = []

    reconstruct_sparse(space, index, derivatives, enforce_asr=True, report=messages.append)

    assert len(messages) == 2
    assert messages[0].startswith("- Max drift of fc3: ")
    assert " -> " in messages[0]
    assert messages[0].endswith(" eV/angstrom^3")
    assert messages[1].startswith("- ASR parameter correction: maximum=")
    assert "relative L2=" in messages[1]


def test_harmonic_translational_and_rotational_rules_are_projected_together():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=2,
        cutoff=resolve_cutoff(supercell, index, -2),
    )
    rng = np.random.default_rng(9)
    pivots = [rng.normal(size=orbit.dimension) for orbit in space.orbits]

    projected, drifts = project_sum_rules(
        space,
        pivots,
        supercell=supercell,
        index=index,
        acoustic=True,
        rotational=True,
    )

    assert drifts["translational"][1] < 1e-8
    assert drifts["rotational"][1] < 1e-8

    sparse_fc = expand_orbit_parameters(
        space,
        np.concatenate(projected),
        n_primitive=index.n_primitive,
        n_supercell=len(supercell),
        index=index,
    )
    dense = sparse_fc.to_dense(primitive_index=index.primitive)
    geometry = PeriodicGeometry(supercell.cell, supercell.pbc)
    axes = np.eye(3)
    residual = []
    for site in range(index.n_primitive):
        first = index.representative(site)
        vectors, _ = geometry.mic(supercell.positions - supercell[first].position)
        rigid = np.cross(axes[:, None, :], vectors[None, :, :])
        residual.append(np.einsum("jab,wjb->aw", dense[site], rigid))
    assert np.max(np.abs(residual)) < 1e-8
