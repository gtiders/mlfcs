from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlfcs.anharmonic.common.thermodynamics import mode_sigma
from mlfcs.anharmonic.scph import LoopSCPH, _fourier_terms, harmonic_frequencies
from mlfcs.core.geometry import StructureRelation
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.io.hdf5 import read_hdf5
from mlfcs.tools import build_supercell


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


def test_loop_correction_has_quartic_one_half_factor():
    fc2, fc4 = _force_constants()
    result = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=300,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
        verbose=False,
    ).run()
    mass = fc2.relation.primitive.get_masses()[0]
    sigma2 = mode_sigma(np.ones(3) / mass, temperature=300, statistics="quantum") ** 2 / mass
    correction = result.loop_correction.materialize(2)[0, 0]
    np.testing.assert_allclose(correction, np.diag(0.5 * sigma2), rtol=1e-12, atol=1e-12)


def test_loop_scph_qpoint_workers_preserve_covariance():
    fc2, fc4 = _force_constants()
    serial = LoopSCPH(
        fc2=fc2, fc4=fc4, temperature=300,
        interpolation_mesh=(1, 1, 1), scph_mesh=(2, 1, 1),
        mixing=1.0, max_iterations=1, verbose=False, qpoint_workers=1,
    ).run()
    parallel = LoopSCPH(
        fc2=fc2, fc4=fc4, temperature=300,
        interpolation_mesh=(1, 1, 1), scph_mesh=(2, 1, 1),
        mixing=1.0, max_iterations=1, verbose=False, qpoint_workers=2,
    ).run()
    np.testing.assert_allclose(
        serial.frequencies, parallel.frequencies, rtol=1e-12, atol=1e-12
    )


def test_loop_scph_temperature_series_uses_previous_effective_fc2():
    fc2, fc4 = _force_constants()
    calculation = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=0.0,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
        verbose=False,
    )
    series = calculation.run_temperature_series([0, 300])
    assert tuple(result.temperature for result in series) == (0.0, 300.0)
    direct = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=300,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
        verbose=False,
        warm_start=series[0].effective_force_constants,
    ).run()
    np.testing.assert_allclose(series[1].frequencies, direct.frequencies)


def test_loop_scph_temperature_range_runs_continuation():
    fc2, fc4 = _force_constants()
    results = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=range(300, 901, 300),
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
        verbose=False,
    ).run()
    assert isinstance(results, tuple)
    assert tuple(result.temperature for result in results) == (300.0, 600.0, 900.0)


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


def test_loop_scph_fourier_terms_keep_reference_translations_for_incompatible_mesh():
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3) * 2.0, pbc=True)
    reference = build_supercell(primitive, np.diag([2, 2, 3]), ordering="phonopy")
    relation = StructureRelation.from_atoms(primitive, reference)
    compact = np.zeros((1, len(reference), 3, 3))
    terms = _fourier_terms(compact, relation)
    for atom, (_, _, fractional, _) in enumerate(terms):
        expected = relation.index.translations[atom] - relation.index.translations[0]
        np.testing.assert_allclose(fractional, expected, atol=1e-12)


def test_loop_scph_keeps_fc4_induced_pair_support():
    primitive = Atoms(
        "H2",
        positions=[[0, 0, 0], [1, 0, 0]],
        cell=np.diag([2.0, 2.0, 2.0]),
        pbc=True,
    )
    relation = StructureRelation.from_atoms(primitive, primitive)
    fc2 = SparseOrderForceConstants(
        2,
        2,
        2,
        np.array([[0, 0]]),
        np.eye(3)[None, ...],
        np.array([[0, 0]]),
        np.zeros((1, 1, 3), dtype=int),
    )
    tensor = np.zeros((1, 3, 3, 3, 3))
    tensor[0, 0, 0, 0, 0] = 1.0
    fc4 = SparseOrderForceConstants(
        4,
        2,
        2,
        np.array([[0, 1, 0, 0]]),
        tensor,
        np.array([[0, 1, 0, 0]]),
        np.zeros((1, 3, 3), dtype=int),
    )
    result = LoopSCPH(
        fc2=ForceConstants({}, primitive, sparse={2: fc2}, relation=relation),
        fc4=ForceConstants({}, primitive, sparse={4: fc4}, relation=relation),
        temperature=300,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        max_iterations=1,
    ).run()
    pairs = {
        (int(sites[0]), int(sites[1]), tuple(map(int, translations[0])))
        for sites, translations in zip(
            result.effective_force_constants.sparse[2].sites,
            result.effective_force_constants.sparse[2].translation_representatives,
            strict=True,
        )
    }
    assert (0, 1, (0, 0, 0)) in pairs


def test_loop_scph_convergence_does_not_require_positive_frequencies():
    fc2, fc4 = _force_constants()
    fc2.sparse[2].tensors *= -1.0
    fc4.sparse[4].tensors *= 0.0
    result = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=0.0,
        interpolation_mesh=(1, 1, 1),
        scph_mesh=(1, 1, 1),
        mixing=1.0,
        tolerance=1e-12,
        max_iterations=1,
    ).run()
    assert result.converged
    assert np.min(result.frequencies) < 0.0
