import numpy as np
import pytest
from ase.build import bulk
from supercell_helpers import make_supercell

from mlfcs.constraints.solver import reconstruct_sparse
from mlfcs.core.real_space import build_primitive_interaction_space, realize_orbit_space


@pytest.mark.parametrize("order", [3, 4])
def test_acoustic_sum_rule_projection_is_strict(order):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    primitive_space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=-1,
        max_body_order=None,
        symprec=1e-5,
    )
    space = realize_orbit_space(primitive_space, index)
    rng = np.random.default_rng(4)
    derivatives = {key: rng.normal(size=(len(supercell), 3)) for key in space.displacement_keys}
    raw = reconstruct_sparse(
        space, index, derivatives, enforce_asr=False, primitive_interaction_space=primitive_space
    ).to_dense()
    projected = reconstruct_sparse(
        space, index, derivatives, enforce_asr=True, primitive_interaction_space=primitive_space
    ).to_dense()
    raw_residual = np.linalg.norm(raw.sum(axis=order - 1))
    projected_residual = np.linalg.norm(projected.sum(axis=order - 1))
    assert projected_residual < raw_residual
    assert projected_residual < 2e-9


def test_asr_reports_phonopy_style_maximum_drift():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    primitive_space = build_primitive_interaction_space(
        primitive,
        order=3,
        cutoff=-1,
        max_body_order=None,
        symprec=1e-5,
    )
    space = realize_orbit_space(primitive_space, index)
    rng = np.random.default_rng(8)
    derivatives = {key: rng.normal(size=(len(supercell), 3)) for key in space.displacement_keys}
    messages = []

    reconstruct_sparse(
        space,
        index,
        derivatives,
        enforce_asr=True,
        report=messages.append,
        primitive_interaction_space=primitive_space,
    )

    assert len(messages) == 2
    assert messages[0].startswith("- Max drift of fc3: ")
    assert " -> " in messages[0]
    assert messages[0].endswith(" eV/angstrom^3")
    assert messages[1].startswith("- ASR parameter correction: maximum=")
    assert "relative L2=" in messages[1]
