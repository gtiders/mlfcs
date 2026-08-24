"""Canonical Taylor potential and ASE Calculator regression tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ForceConstantFitter, MLFCSCalculator, build_supercell
from mlfcs.calculators.taylor import TaylorPotential
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.io.hdf5 import write_hdf5
from mlfcs.structure.relation import StructureRelation


def _model(*, orders=(2, 3, 4, 5), reference=None):
    primitive = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    if reference is None:
        reference = build_supercell(primitive, (2, 1, 1))
    relation = StructureRelation.from_atoms(primitive, reference)
    sparse = {}
    rng = np.random.default_rng(91)
    for order in orders:
        vector = rng.normal(scale=0.3, size=3)
        tensor = vector
        for _ in range(order - 1):
            tensor = np.multiply.outer(tensor, vector)
        sparse[order] = SparseOrderForceConstants(
            order,
            np.zeros((1, order), dtype=np.int32),
            np.zeros((1, order - 1, 3), dtype=np.int32),
            tensor[None],
        )
    return ForceConstants(
        {},
        reference.copy(),
        metadata={"force_constants_basis": "taylor"},
        sparse=sparse,
        relation=relation,
    )


def test_reference_energy_and_forces_are_zero_for_all_orders():
    model = _model()
    calculator = MLFCSCalculator(model)
    atoms = model.relation.reference.copy()
    atoms.calc = calculator
    assert atoms.get_potential_energy() == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(atoms.get_forces(), 0.0, atol=1e-15)


def test_calculator_matches_taylor_fitting_design_prediction():
    primitive = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = build_supercell(primitive, (2, 1, 1))
    structures = []
    for value in (-0.03, -0.01, 0.01, 0.03):
        atoms = reference.copy()
        atoms.positions[0, 0] += value
        forces = np.zeros((len(reference), 3))
        forces[0, 0] = -value
        forces[1, 0] = value
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)
    result = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 4.1},
        fitting_basis="taylor",
    ).fit(structures, validation_split=0.0, acoustic_sum_rule=False)
    calculator = MLFCSCalculator(result.force_constants)
    predicted = []
    observed = []
    for atoms in structures:
        evaluation = atoms.copy()
        evaluation.calc = calculator
        predicted.append(evaluation.get_forces())
        observed.append(atoms.get_forces())
    rmse = float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(observed)) ** 2)))
    assert rmse == pytest.approx(result.training_force_rmse, abs=1e-12)


def test_force_is_negative_energy_gradient():
    model = _model()
    potential = TaylorPotential(model)
    displacement = np.array([[0.011, -0.007, 0.004], [-0.003, 0.006, -0.009]])
    _, forces = potential.evaluate_displacement(displacement)
    step = 1e-6
    numerical = np.empty_like(forces)
    for atom in range(len(displacement)):
        for axis in range(3):
            plus = displacement.copy()
            minus = displacement.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical[atom, axis] = -(
                potential.evaluate_displacement(plus)[0]
                - potential.evaluate_displacement(minus)[0]
            ) / (2 * step)
    np.testing.assert_allclose(forces, numerical, atol=2e-11, rtol=2e-9)


def test_hdf5_reload_and_larger_target_are_supported(tmp_path):
    model = _model(orders=(2, 3))
    path = tmp_path / "mlfcs.h5"
    write_hdf5(path, model)
    target = build_supercell(model.relation.primitive, (3, 1, 1))
    calculator = MLFCSCalculator.from_hdf5(path, reference=target)
    atoms = target.copy()
    atoms.positions[0, 0] += 0.02
    atoms.calc = calculator
    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))


def test_non_diagonal_target_supercell_is_supported():
    primitive_model = _model(orders=(2, 3))
    matrix = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=int)
    target = build_supercell(primitive_model.relation.primitive, matrix)
    calculator = MLFCSCalculator(primitive_model, reference=target)
    atoms = target.copy()
    atoms.positions[1] += [0.004, -0.006, 0.008]
    atoms.calc = calculator
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (len(target), 3)


def test_wrapped_and_unwrapped_positions_are_equivalent():
    model = _model(orders=(2,))
    potential = TaylorPotential(model)
    displaced = model.relation.reference.copy()
    displaced.positions[0] += [0.013, -0.008, 0.004]
    wrapped = displaced.copy()
    wrapped.positions[0] += wrapped.cell[0]
    left = potential.evaluate(displaced)
    right = potential.evaluate(wrapped)
    assert left[0] == pytest.approx(right[0], abs=1e-14)
    np.testing.assert_allclose(left[1], right[1], atol=1e-14)


def test_incompatible_structure_and_properties_are_rejected():
    model = _model(orders=(2,))
    calculator = MLFCSCalculator(model)
    wrong = model.relation.reference.copy()
    wrong.numbers[0] = 2
    with pytest.raises(ValueError, match="atom order"):
        calculator.calculate(wrong)
    with pytest.raises(NotImplementedError, match="stress"):
        calculator.calculate(model.relation.reference, properties=["stress"])


def test_non_taylor_metadata_is_rejected():
    model = _model(orders=(2,))
    model.metadata["force_constants_basis"] = "wick"
    with pytest.raises(ValueError, match="Taylor"):
        MLFCSCalculator(model)


def test_displacement_warning_does_not_clip(caplog):
    model = _model(orders=(2,))
    atoms = model.relation.reference.copy()
    atoms.positions[0, 0] += 0.02
    calculator = MLFCSCalculator(model, maximum_displacement=0.01)
    with caplog.at_level(logging.WARNING, logger="mlfcs.calculators.ase"):
        calculator.calculate(atoms)
    assert "continues without clipping" in caplog.text
    direct = TaylorPotential(model).evaluate(atoms)
    np.testing.assert_allclose(calculator.results["forces"], direct[1])
