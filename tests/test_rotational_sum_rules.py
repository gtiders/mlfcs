import numpy as np
from ase import Atoms
from supercell_helpers import make_supercell

from mlfcs import FiniteDifferenceCalculation, enforce_rotational_sum_rules
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants


def test_strict_harmonic_projection_uses_tied_images_and_keeps_higher_orders():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 3.0, pbc=True)
    calculation = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=make_supercell(primitive, (2, 1, 1))[0],
        cutoff=3.1,
        verbose=False,
    )
    fc2 = SparseOrderForceConstants(
        2,
        np.array([[0, 0], [0, 0]], dtype=np.int32),
        np.array([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.int32),
        np.array([np.eye(3), -np.eye(3)]),
    )
    fc3 = SparseOrderForceConstants(
        3,
        np.array([[0, 0, 0]], dtype=np.int32),
        np.array([[[1, 0, 0], [1, 0, 0]]], dtype=np.int32),
        np.ones((1, 3, 3, 3)),
    )
    fc4 = SparseOrderForceConstants(
        4,
        np.array([[0, 0, 0, 0]], dtype=np.int32),
        np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=np.int32),
        np.ones((1, 3, 3, 3, 3)),
    )
    result = ForceConstants(
        {},
        calculation.supercell,
        sparse={2: fc2, 3: fc3, 4: fc4},
        relation=calculation.interaction_space.relation,
    )
    constrained = enforce_rotational_sum_rules(result, born_huang=True, huang=True)
    diagnostics = constrained.diagnostics
    assert diagnostics.strength == 1.0
    assert diagnostics.huang_before is not None and diagnostics.huang_before > 1.0
    assert diagnostics.huang_after is not None and diagnostics.huang_after < 1e-10
    assert diagnostics.acoustic_after < 1e-10
    np.testing.assert_array_equal(constrained.force_constants.sparse[3].tensors, fc3.tensors)
    np.testing.assert_array_equal(constrained.force_constants.sparse[4].tensors, fc4.tensors)


def test_strength_zero_only_enforces_asr():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 3.0, pbc=True)
    calculation = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=make_supercell(primitive, (2, 1, 1))[0],
        cutoff=3.1,
        verbose=False,
    )
    fc2 = SparseOrderForceConstants(
        2,
        np.array([[0, 0], [0, 0]], dtype=np.int32),
        np.array([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.int32),
        np.array([np.eye(3), np.eye(3)]),
    )
    result = ForceConstants(
        {}, calculation.supercell, sparse={2: fc2}, relation=calculation.interaction_space.relation
    )
    constrained = enforce_rotational_sum_rules(result, huang=True, strength=0.0)
    assert constrained.diagnostics.acoustic_after < 1e-10
    assert constrained.diagnostics.huang_after is not None
    assert constrained.diagnostics.huang_after <= constrained.diagnostics.huang_before
