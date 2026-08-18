from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfcs.anharmonic.scph import LoopSCPH, harmonic_frequencies
from mlfcs.core.geometry import StructureRelation
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.io.hdf5 import read_hdf5


def _force_constants(*, cell=4.0):
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * cell, pbc=True)
    relation = StructureRelation.from_atoms(primitive, primitive)
    fc2_tensor = np.eye(3)[None, ...]
    quartic = np.zeros((1, 3, 3, 3, 3))
    for axis in range(3):
        quartic[0, axis, axis, axis, axis] = 1.0
    fc2 = SparseOrderForceConstants(
        2,
        1,
        1,
        np.array([[0, 0]]),
        fc2_tensor,
        np.array([[0, 0]]),
        np.zeros((1, 1, 3), dtype=int),
    )
    fc4 = SparseOrderForceConstants(
        4,
        1,
        1,
        np.array([[0, 0, 0, 0]]),
        quartic,
        np.array([[0, 0, 0, 0]]),
        np.zeros((1, 3, 3), dtype=int),
    )
    return (
        ForceConstants({}, primitive, sparse={2: fc2}, relation=relation),
        ForceConstants({}, primitive, sparse={4: fc4}, relation=relation),
    )


def test_loop_scph_accepts_independent_fc2_and_fc4_and_writes_effective_fc2(tmp_path):
    fc2, fc4 = _force_constants()
    result = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=300,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        tolerance=1e-12,
        max_iterations=2,
    ).run()
    base = fc2.materialize(2)
    effective = result.effective_force_constants.materialize(2)
    assert np.all(np.diag(effective[0, 0]) > np.diag(base[0, 0]))
    target = tmp_path / "effective.h5"
    result.write(target, format="hdf5")
    assert read_hdf5(target).orders == (2,)
    assert result.history[0].frequency_change_thz >= 0.0


def test_loop_scph_uses_alamode_rms_frequency_stopping_metric():
    fc2, fc4 = _force_constants()
    _, initial = harmonic_frequencies(fc2, (1, 1, 1))
    result = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=300,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
    ).run()
    expected = np.sqrt(np.mean((result.frequencies - initial) ** 2))
    assert result.history[0].frequency_change_thz == pytest.approx(expected)


def test_loop_scph_rejects_incompatible_force_constant_frames():
    fc2, _ = _force_constants(cell=4.0)
    _, fc4 = _force_constants(cell=5.0)
    with pytest.raises(ValueError, match="supercell matrices|reference supercells"):
        LoopSCPH(
            fc2=fc2,
            fc4=fc4,
            temperature=300,
            interpolation_mesh=(1, 1, 1),
            scph_mesh=(1, 1, 1),
        )
