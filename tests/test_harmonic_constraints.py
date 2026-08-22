import numpy as np
from ase import Atoms

from mlfcs import ForceConstantCalculation
from mlfcs.model import ForceConstants, SparseOrderForceConstants


def test_strict_harmonic_projection_uses_tied_images_and_keeps_higher_orders():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 3.0, pbc=True)
    calculation = ForceConstantCalculation(
        primitive, order=2, supercell=(2, 1, 1), cutoff=None, verbose=False
    )
    fc2 = SparseOrderForceConstants(
        2,
        1,
        2,
        np.array([[0, 0], [0, 1]], dtype=np.int32),
        np.array([np.eye(3), -np.eye(3)]),
        sites=np.array([[0, 0], [0, 0]], dtype=np.int32),
        translation_representatives=np.array([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.int32),
    )
    fc3 = SparseOrderForceConstants(
        3,
        1,
        2,
        np.array([[0, 1, 1]], dtype=np.int32),
        np.ones((1, 3, 3, 3)),
        sites=np.array([[0, 0, 0]], dtype=np.int32),
        translation_representatives=np.array([[[1, 0, 0], [1, 0, 0]]], dtype=np.int32),
    )
    fc4 = SparseOrderForceConstants(
        4,
        1,
        2,
        np.array([[0, 1, 1, 1]], dtype=np.int32),
        np.ones((1, 3, 3, 3, 3)),
        sites=np.array([[0, 0, 0, 0]], dtype=np.int32),
        translation_representatives=np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=np.int32),
    )
    result = ForceConstants(
        {},
        calculation.supercell,
        sparse={2: fc2, 3: fc3, 4: fc4},
        relation=calculation.interaction_space.relation,
    )
    constrained = result.enforce_harmonic_constraints(born_huang=True, huang=True)
    diagnostics = constrained.diagnostics
    assert diagnostics.strength == 1.0
    assert diagnostics.huang_before is not None and diagnostics.huang_before > 1.0
    assert diagnostics.huang_after is not None and diagnostics.huang_after < 1e-10
    assert diagnostics.acoustic_after < 1e-10
    np.testing.assert_array_equal(constrained.force_constants.sparse[3].tensors, fc3.tensors)
    np.testing.assert_array_equal(constrained.force_constants.sparse[4].tensors, fc4.tensors)


def test_strength_zero_only_enforces_asr():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 3.0, pbc=True)
    calculation = ForceConstantCalculation(
        primitive, order=2, supercell=(2, 1, 1), cutoff=None, verbose=False
    )
    fc2 = SparseOrderForceConstants(
        2,
        1,
        2,
        np.array([[0, 0], [0, 1]], dtype=np.int32),
        np.array([np.eye(3), np.eye(3)]),
        sites=np.array([[0, 0], [0, 0]], dtype=np.int32),
        translation_representatives=np.array([[[0, 0, 0]], [[1, 0, 0]]], dtype=np.int32),
    )
    result = ForceConstants(
        {}, calculation.supercell, sparse={2: fc2}, relation=calculation.interaction_space.relation
    )
    constrained = result.enforce_harmonic_constraints(huang=True, strength=0.0)
    assert constrained.diagnostics.acoustic_after < 1e-10
    assert constrained.diagnostics.huang_after is not None
    assert constrained.diagnostics.huang_after > 1e-3
