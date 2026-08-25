from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy.file_IO import parse_FORCE_CONSTANTS

from mlfcs import (
    ForceConstantFitter,
    MLFCSCalculator,
    build_supercell,
    read_hdf5,
    write_force_constants,
)
from mlfcs.force_constants.dense import expand_compact_fc2
from mlfcs.force_constants.periodic_fc2 import SupercellHessianSpace


def _model():
    primitive = bulk("NaCl", "rocksalt", a=5.64)
    reference = build_supercell(primitive, (2, 2, 2))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
        periodic_fc2_completion=True,
    )
    return fitter, SupercellHessianSpace.build(fitter.calculations[0])


def test_periodic_fc2_space_is_hessian_asr_and_exact_complement():
    fitter, space = _model()
    full = space.full_completion_basis().T.reshape((-1, 16, 16, 3, 3))
    assert space.rank_report.asr_dimension == 11
    assert space.rank_report.exact_rank == 2
    assert space.rank_report.completion_dimension == 9
    assert np.max(np.abs(full - full.transpose(0, 2, 1, 4, 3))) < 1e-12
    assert np.max(np.abs(full.sum(axis=2))) < 1e-12

    exact = space.compact_basis @ space.exact_map
    assert np.linalg.norm(exact.T @ space.completion_basis) < 1e-12
    hybrid = np.column_stack((exact, space.completion_basis))
    assert np.linalg.matrix_rank(hybrid) == space.rank_report.asr_dimension
    exact_parameters = np.arange(1, space.exact_map.shape[1] + 1, dtype=float)
    exact_target = exact @ exact_parameters
    recovered = np.linalg.lstsq(hybrid, exact_target, rcond=None)[0]
    assert np.allclose(recovered[: len(exact_parameters)], exact_parameters, atol=1e-12)
    assert np.linalg.norm(recovered[len(exact_parameters) :]) < 1e-12
    assert fitter.periodic_fc2_completion


def test_hybrid_recovers_a_random_periodic_hessian_and_hdf5(tmp_path):
    fitter, space = _model()
    rng = np.random.default_rng(42)
    coordinates = rng.normal(size=space.compact_basis.shape[1])
    compact_shape = (len(fitter.primitive), len(fitter.reference), 3, 3)
    target = (space.compact_basis @ coordinates).reshape(compact_shape)
    full = expand_compact_fc2(target, fitter.reference)
    structures = []
    for _ in range(8):
        displacement = rng.normal(scale=0.01, size=(len(fitter.reference), 3))
        displacement -= displacement.mean(axis=0)
        atoms = fitter.reference.copy()
        atoms.positions += displacement
        atoms.calc = SinglePointCalculator(
            atoms,
            forces=-np.einsum("ijab,jb->ia", full, displacement),
        )
        structures.append(atoms)

    result = fitter.fit(
        structures,
        validation_split=0,
        batch_size=2,
        tolerance=1e-11,
        max_iterations=5000,
    )
    recovered = result.force_constants.materialize(2)
    assert result.periodic_fc2_completion is not None
    assert result.training_force_rmse < 1e-9
    assert np.linalg.norm(recovered - target) / np.linalg.norm(target) < 1e-9

    output = tmp_path / "hybrid.h5"
    write_force_constants(result.force_constants, output, format="hdf5")
    restored = read_hdf5(output)
    assert restored.periodic_fc2_completion is not None
    assert np.allclose(restored.materialize(2), recovered, atol=1e-12)

    evaluated = structures[0].copy()
    evaluated.calc = MLFCSCalculator(restored, reference=fitter.reference)
    assert np.allclose(evaluated.get_forces(), structures[0].get_forces(), atol=1e-9)

    phonopy_output = tmp_path / "FORCE_CONSTANTS"
    write_force_constants(restored, phonopy_output, format="phonopy")
    exported = parse_FORCE_CONSTANTS(filename=phonopy_output)
    assert np.allclose(exported, expand_compact_fc2(recovered, fitter.reference), atol=1e-10)

    other = build_supercell(fitter.primitive, (3, 3, 3))
    with pytest.raises(ValueError, match="different source supercell"):
        result.periodic_fc2_completion.full_hessian(other)


def test_periodic_completion_is_default_off_and_requires_asr():
    primitive = bulk("NaCl", "rocksalt", a=5.64)
    reference = build_supercell(primitive, (2, 2, 2))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
    )
    assert not fitter.periodic_fc2_completion

    enabled = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
        periodic_fc2_completion=True,
    )
    atoms = enabled.reference.copy()
    atoms.calc = SinglePointCalculator(atoms, forces=np.zeros((len(atoms), 3)))
    with pytest.raises(ValueError, match="requires acoustic_sum_rule=True"):
        enabled.fit([atoms], validation_split=0, acoustic_sum_rule=False)
