"""Small physical and ASE-only contracts for fixed-centroid SSCHA."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import PrimitiveCell, PrimitiveSymmetry
from mlfcs.force_constants import ForceConstants
from mlfcs.reciprocal import SSCHA, QGrid, QStars, SSCHAContinuationError
from mlfcs.reciprocal.ensemble import HarmonicEnsemble, UnstableTrialError, _expand_vectors
from mlfcs.reciprocal.stars import StarPlan
from mlfcs.supercell import ClusterMap, Supercell
from mlfcs.tools.taylor import TaylorCalculator


def _system():
    primitive = PrimitiveCell.from_atoms(bulk("Ar", "sc", a=1.0), symprec=1e-5)
    supercell = Supercell.from_atoms(primitive, primitive.to_atoms().repeat((2, 2, 2)))
    space = ClusterSpace(
        primitive.to_atoms(), symprec=primitive.symprec, cutoffs={2: 1.1}, max_body_orders={2: 2}
    )
    mapping = ClusterMap.build(space, supercell)
    coefficients = []
    for orbit in space.orbits:
        tensor = (
            6.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else -1.0
        ) * np.eye(3)
        coefficients.extend(
            np.linalg.lstsq(orbit.component_basis, tensor.reshape(-1), rcond=None)[0]
        )
    model = (
        ForceConstants(space, {2: np.asarray(coefficients)})
        .enforce_asr(orders=(2,))
        .force_constants
    )
    return mapping, model


def test_reduced_ensemble_is_real_paired_and_has_the_expected_covariance() -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell.matrix, initial=model, seed=7)
    ensemble = HarmonicEnsemble(model, mapping.supercell, solver.stars, 300, statistics="quantum")
    first = ensemble.sample(4000, seed=7, iteration=1)
    np.testing.assert_array_equal(first, ensemble.sample(4000, seed=7, iteration=1))
    assert first.shape == (4000, 8, 3)
    assert np.all(np.isfinite(first))
    masses = mapping.supercell.primitive.masses[mapping.supercell.sites]
    np.testing.assert_allclose(np.einsum("a,pad->pd", masses, first), 0.0, atol=1e-12)
    assert ensemble.minimum_frequency_thz > 0.0
    theoretical = 0.0
    for member in range(ensemble.stars.grid.size):
        star = int(ensemble.stars.star_of[member])
        values = ensemble.eigenvalues[star]
        vectors = ensemble.vectors[star]
        covariance = (vectors * ensemble._variances(values)) @ vectors.conj().T
        theoretical += float(ensemble.plan.matrix(member, covariance)[0, 0].real)
    theoretical /= ensemble.stars.grid.size * masses[0]
    assert np.var(first[:, 0, 0]) == pytest.approx(theoretical, rel=0.08)


def test_non_self_conjugate_q_pair_keeps_the_full_random_variance() -> None:
    _, model = _system()
    primitive = model.space.primitive
    supercell = Supercell.from_atoms(primitive, primitive.to_atoms().repeat((3, 1, 1)))
    stars = QStars.from_symmetry(QGrid.from_matrix(supercell.matrix), model.space.symmetry)
    ensemble = HarmonicEnsemble(model, supercell, stars, 300, statistics="classical")
    samples = ensemble.sample(4000, seed=17, iteration=1)
    theoretical = 0.0
    for member in range(stars.grid.size):
        star = int(stars.star_of[member])
        values = ensemble.eigenvalues[star]
        vectors = ensemble.vectors[star]
        covariance = (vectors * ensemble._variances(values)) @ vectors.conj().T
        theoretical += float(ensemble.plan.matrix(member, covariance)[0, 0].real)
    theoretical /= stars.grid.size * primitive.masses[0]
    assert np.var(samples[:, 0, 0]) == pytest.approx(theoretical, rel=0.08)
    np.testing.assert_allclose(np.sum(samples, axis=1), 0.0, atol=1e-12)


def test_only_star_representatives_are_diagonalized(monkeypatch) -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=4)
    calls = []
    original = np.linalg.eigh

    def tracked(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", tracked)
    ensemble = HarmonicEnsemble(model, mapping.supercell, solver.stars, 300, statistics="quantum")
    assert len(calls) == len(solver.stars.representatives)
    assert len(calls) < solver.stars.grid.size
    ensemble.sample(8, seed=4, iteration=1)
    assert len(calls) == len(solver.stars.representatives)


def test_member_modal_vectors_include_the_positional_gauge() -> None:
    primitive = PrimitiveCell.from_atoms(
        Atoms(
            "NaCl",
            scaled_positions=[[0, 0, 0], [0.25, 0, 0]],
            cell=np.diag([2.0, 3.0, 4.0]),
            pbc=True,
        ),
        symprec=1e-5,
    )
    stars = QStars.from_symmetry(
        QGrid.from_matrix(np.diag([3, 1, 1])), PrimitiveSymmetry.from_primitive(primitive)
    )
    plan = StarPlan.from_stars(stars, primitive)
    member = next(index for index in range(stars.grid.size) if stars.antiunitary[index])
    source = np.arange(18, dtype=float).reshape(6, 3) + 1j * np.arange(18, 36).reshape(6, 3)
    actual = _expand_vectors(plan, member, source)
    expected = plan.matrix(member, source @ source.conj().T)
    np.testing.assert_allclose(actual @ actual.conj().T, expected, atol=1e-12)
    assert not np.allclose(actual, source.conj())


def test_two_site_sample_covariance_uses_the_star_gauge() -> None:
    primitive = PrimitiveCell.from_atoms(
        Atoms(
            "NaCl",
            scaled_positions=[[0, 0, 0], [0.25, 0, 0]],
            cell=np.diag([2.0, 3.0, 4.0]),
            pbc=True,
        ),
        symprec=1e-5,
    )
    space = ClusterSpace(
        primitive.to_atoms(), symprec=primitive.symprec, cutoffs={2: 0.6}, max_body_orders={2: 2}
    )
    coefficients = []
    for orbit in space.orbits:
        strength = 4.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else -1.0
        coefficients.extend(
            np.linalg.lstsq(orbit.component_basis, (strength * np.eye(3)).reshape(-1), rcond=None)[
                0
            ]
        )
    model = ForceConstants(space, {2: np.asarray(coefficients)})
    supercell = Supercell.from_atoms(primitive, primitive.to_atoms().repeat((3, 1, 1)))
    stars = QStars.from_symmetry(QGrid.from_matrix(supercell.matrix), space.symmetry)
    ensemble = HarmonicEnsemble(model, supercell, stars, 300, statistics="quantum")
    samples = ensemble.sample(6000, seed=5, iteration=1)
    first, second = 0, 1
    actual = float(np.mean(samples[:, first, 0] * samples[:, second, 0]))
    expected = 0.0
    positions = supercell.translations + primitive.scaled_positions[supercell.sites]
    for member in range(stars.grid.size):
        star = int(stars.star_of[member])
        values = ensemble.eigenvalues[star]
        vectors = ensemble.vectors[star]
        covariance = (vectors * ensemble._variances(values)) @ vectors.conj().T
        expanded = ensemble.plan.matrix(member, covariance)
        phase = np.exp(
            2j * np.pi * (stars.grid.points[member] @ (positions[first] - positions[second]))
        )
        expected += float(
            (phase * expanded[3 * supercell.sites[first], 3 * supercell.sites[second]]).real
        )
    expected /= stars.grid.size * np.sqrt(
        primitive.masses[supercell.sites[first]] * primitive.masses[supercell.sites[second]]
    )
    assert actual == pytest.approx(expected, rel=0.12, abs=1e-5)


def test_ase_calculator_path_fits_harmonic_fc2_and_reports_energy() -> None:
    mapping, model = _system()
    calculator = TaylorCalculator(model, mapping)
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=9)
    result = solver.run(300, calculator, pairs=12, max_iterations=2, tol_thz=1e-5, energy=True)
    assert result.converged
    assert len(result.history) == 1
    np.testing.assert_allclose(result.fc2.parameters(), model.parameters(), rtol=1e-8, atol=1e-8)
    assert result.history[0].free_energy is not None
    assert result.history[0].free_energy_error is not None
    assert result.history[0].mean_force_norm < 1e-10
    trial = HarmonicEnsemble(model, mapping.supercell, solver.stars, 300, statistics="quantum")
    assert result.history[0].free_energy == pytest.approx(trial.free_energy(), abs=1e-10)


def test_zero_temperature_quantum_sampling_is_defined_but_classical_is_not() -> None:
    mapping, model = _system()
    quantum = SSCHA(mapping, (2, 2, 2), initial=model, seed=3)
    ensemble = HarmonicEnsemble(model, mapping.supercell, quantum.stars, 0, statistics="quantum")
    assert np.any(ensemble.sample(2, seed=3, iteration=1))
    classical = SSCHA(mapping, (2, 2, 2), initial=model, statistics="classical", seed=3)
    with pytest.raises(ValueError, match="zero temperature"):
        classical.run(0, TaylorCalculator(model, mapping), pairs=2)


def test_multiorder_map_cannot_be_silently_truncated_to_fc2() -> None:
    mapping, _ = _system()
    primitive = mapping.space.primitive
    space = ClusterSpace(
        primitive.to_atoms(),
        symprec=primitive.symprec,
        cutoffs={2: 1.1, 4: 0.1},
        max_body_orders={2: 2, 4: 1},
    )
    mixed = ClusterMap.build(space, mapping.supercell)
    with pytest.raises(ValueError, match="FC2 only"):
        SSCHA(mixed, (2, 2, 2))


def test_unstable_trial_is_not_silently_sampled() -> None:
    mapping, model = _system()
    unstable = ForceConstants(model.space, {2: -model.coefficients[2]})
    solver = SSCHA(mapping, mapping.supercell.matrix, initial=unstable, seed=1)
    with pytest.raises(UnstableTrialError, match="positive Gaussian"):
        solver.run(300, TaylorCalculator(model, mapping), pairs=4, max_iterations=1)

    bootstrapped = SSCHA(
        mapping,
        (2, 2, 2),
        initial=unstable,
        bootstrap_displacement=0.05,
        seed=1,
    ).run(300, TaylorCalculator(model, mapping), pairs=12, max_iterations=2)
    assert bootstrapped.converged


def test_unstable_fitted_update_is_reported_without_modifying_the_trial() -> None:
    mapping, model = _system()
    unstable = ForceConstants(model.space, {2: -10.0 * model.coefficients[2]})
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=1)
    result = solver.run(
        300,
        TaylorCalculator(unstable, mapping),
        pairs=12,
        mixing=1.0,
        max_backtracks=0,
        max_iterations=1,
    )
    assert result.status == "unstable"
    assert result.message is not None and "representative" in result.message
    np.testing.assert_array_equal(result.fc2.coefficients[2], model.coefficients[2])


class _Counting(Calculator):
    implemented_properties: ClassVar[list[str]] = ["forces"]

    def __init__(self, model: ForceConstants, mapping: ClusterMap):
        super().__init__()
        self.inner = TaylorCalculator(model, mapping)
        self.calls = 0

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        copy = atoms.copy()
        copy.calc = self.inner
        self.results = {"forces": copy.get_forces()}


def test_only_ase_calculator_is_accepted_and_forces_are_recomputed() -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell.matrix, initial=model, seed=2)
    with pytest.raises(TypeError, match="ASE Calculator"):
        solver.run(300, object(), pairs=4)
    calculator = _Counting(model, mapping)
    result = solver.run(300, calculator, pairs=6, max_iterations=1, tol_thz=1e-5)
    assert calculator.calls == 12
    assert result.history[0].free_energy is None
    with pytest.raises(Exception, match="energy"):
        solver.run(300, calculator, pairs=4, max_iterations=1, energy=True)


def test_gaussian_tail_cannot_be_silently_folded_in_the_fit() -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=2)
    calculator = _Counting(model, mapping)
    displacement = np.zeros((1, len(mapping.supercell.numbers), 3))
    displacement[0, 0, 0] = mapping.supercell.cell[0, 0]
    with pytest.raises(ValueError, match="minimum-image boundary"):
        solver._evaluate(displacement, calculator, energy=False)
    assert calculator.calls == 0


def test_cartesian_bootstrap_is_separate_from_canonical_iterations() -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell.matrix, seed=11)
    result = solver.run(300, TaylorCalculator(model, mapping), pairs=12, max_iterations=2)
    assert result.converged
    assert len(result.history) == 1
    assert result.bootstrap_displacement == 0.01
    assert result.history[0].phase == "canonical"
    np.testing.assert_allclose(result.fc2.parameters(), model.parameters(), atol=1e-8)


def test_iteration_and_sampling_limits_are_not_reported_as_convergence(monkeypatch) -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=3)
    calculator = TaylorCalculator(model, mapping)
    monkeypatch.setattr(HarmonicEnsemble, "frequency_change_thz", lambda _self, _other: 1.0)
    limited = solver.run(300, calculator, pairs=12, max_iterations=1, tol_thz=0.1)
    assert limited.status == "max_iterations"
    assert not limited.converged

    calls = 0

    def changes(_self, _other):
        nonlocal calls
        calls += 1
        return 1.0 if calls == 1 else 0.0

    monkeypatch.setattr(HarmonicEnsemble, "frequency_change_thz", changes)
    noisy = solver.run(300, calculator, pairs=12, max_iterations=1, tol_thz=0.1)
    assert noisy.status == "insufficient_samples"
    assert noisy.message is not None and "frequency uncertainty" in noisy.message


def test_temperature_schedule_descends_and_warm_starts_only_after_success(monkeypatch) -> None:
    mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell.matrix, initial=model, seed=12)
    calculator = TaylorCalculator(model, mapping)
    calls = []
    original = solver.run

    def tracked(temperature, calculator, *, start=None, **kwargs):
        calls.append((temperature, start))
        return original(temperature, calculator, start=start, **kwargs)

    monkeypatch.setattr(solver, "run", tracked)
    results = solver.run_many([0, 100], calculator, pairs=12, max_iterations=2)
    assert [result.temperature for result in results] == [0.0, 100.0]
    assert calls[0] == (100.0, None)
    assert calls[1][0] == 0.0
    assert calls[1][1] is results[1].fc2

    def failed(temperature, calculator, *, start=None, **kwargs):
        result = original(temperature, calculator, start=start, **kwargs)
        return replace(result, status="max_iterations")

    monkeypatch.setattr(solver, "run", failed)
    with pytest.raises(SSCHAContinuationError, match="100 K"):
        solver.run_many([0, 100], calculator, pairs=12, max_iterations=1)
