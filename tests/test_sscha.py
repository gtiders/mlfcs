"""Small physical and ASE-only contracts for fixed-centroid SSCHA."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from math import factorial

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from numba import njit

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.lattice import expand
from mlfcs.mapping import ClusterMap
from mlfcs.reciprocal import SSCHA, QGrid, QStars, SSCHAContinuationError
from mlfcs.algebra.integer import adjugate_3x3
from mlfcs.mapping.geometry import quotient_kernel
from mlfcs.reciprocal.ensemble import HarmonicEnsemble, UnstableTrialError, _expand_vectors
from mlfcs.reciprocal.stars import StarPlan


def _system():
    atoms = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(atoms, symprec=1e-5, cutoffs={2: 1.1}, max_body_orders={2: 2})
    mapping = ClusterMap(space, atoms.repeat((2, 2, 2)))
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
    return atoms, mapping, model


def test_reduced_ensemble_is_real_paired_and_has_the_expected_covariance() -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell_matrix, initial=model, seed=7)
    ensemble = HarmonicEnsemble(model, mapping, solver.stars, 300, statistics="quantum")
    first = ensemble.sample(4000, seed=7, iteration=1)
    np.testing.assert_array_equal(first, ensemble.sample(4000, seed=7, iteration=1))
    assert first.shape == (4000, 8, 3)
    assert np.all(np.isfinite(first))
    masses = mapping.cluster_space.primitive_atoms.get_masses()[mapping.primitive_site_indices]
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
    _, mapping, model = _system()
    atoms = mapping.cluster_space.primitive_atoms
    supercell = ClusterMap(model.cluster_space, atoms.repeat((3, 1, 1)))
    stars = QStars.from_symmetry(QGrid.from_matrix(supercell.supercell_matrix), model.cluster_space.symmetry)
    ensemble = HarmonicEnsemble(model, supercell, stars, 300, statistics="classical")
    samples = ensemble.sample(4000, seed=17, iteration=1)
    theoretical = 0.0
    for member in range(stars.grid.size):
        star = int(stars.star_of[member])
        values = ensemble.eigenvalues[star]
        vectors = ensemble.vectors[star]
        covariance = (vectors * ensemble._variances(values)) @ vectors.conj().T
        theoretical += float(ensemble.plan.matrix(member, covariance)[0, 0].real)
    theoretical /= stars.grid.size * atoms.get_masses()[0]
    assert np.var(samples[:, 0, 0]) == pytest.approx(theoretical, rel=0.08)
    np.testing.assert_allclose(np.sum(samples, axis=1), 0.0, atol=1e-12)


def test_only_star_representatives_are_diagonalized(monkeypatch) -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=4)
    calls = []
    original = np.linalg.eigh

    def tracked(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", tracked)
    ensemble = HarmonicEnsemble(model, mapping, solver.stars, 300, statistics="quantum")
    assert len(calls) == len(solver.stars.representatives)
    assert len(calls) < solver.stars.grid.size
    ensemble.sample(8, seed=4, iteration=1)
    assert len(calls) == len(solver.stars.representatives)


def test_member_modal_vectors_include_the_positional_gauge() -> None:
    space = ClusterSpace(
        Atoms(
            "NaCl",
            scaled_positions=[[0, 0, 0], [0.25, 0, 0]],
            cell=np.diag([2.0, 3.0, 4.0]),
            pbc=True,
        ),
        symprec=1e-5,
        cutoffs={2: 0.01},
    )
    stars = QStars.from_symmetry(QGrid.from_matrix(np.diag([3, 1, 1])), space.symmetry)
    plan = StarPlan.from_stars(stars, space)
    member = next(index for index in range(stars.grid.size) if stars.antiunitary[index])
    source = np.arange(18, dtype=float).reshape(6, 3) + 1j * np.arange(18, 36).reshape(6, 3)
    actual = _expand_vectors(plan, member, source)
    expected = plan.matrix(member, source @ source.conj().T)
    np.testing.assert_allclose(actual @ actual.conj().T, expected, atol=1e-12)
    assert not np.allclose(actual, source.conj())


def test_two_site_sample_covariance_uses_the_star_gauge() -> None:
    atoms = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.25, 0, 0]],
        cell=np.diag([2.0, 3.0, 4.0]),
        pbc=True,
    )
    space = ClusterSpace(atoms, symprec=1e-5, cutoffs={2: 0.6}, max_body_orders={2: 2})
    coefficients = []
    for orbit in space.orbits:
        strength = 4.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else -1.0
        coefficients.extend(
            np.linalg.lstsq(orbit.component_basis, (strength * np.eye(3)).reshape(-1), rcond=None)[
                0
            ]
        )
    model = ForceConstants(space, {2: np.asarray(coefficients)})
    supercell = ClusterMap(space, atoms.repeat((3, 1, 1)))
    stars = QStars.from_symmetry(QGrid.from_matrix(supercell.supercell_matrix), space.symmetry)
    ensemble = HarmonicEnsemble(model, supercell, stars, 300, statistics="quantum")
    samples = ensemble.sample(6000, seed=5, iteration=1)
    first, second = 0, 1
    actual = float(np.mean(samples[:, first, 0] * samples[:, second, 0]))
    expected = 0.0
    positions = supercell.lattice_translations + space.scaled_positions[supercell.primitive_site_indices]
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
            (phase * expanded[3 * supercell.primitive_site_indices[first], 3 * supercell.primitive_site_indices[second]]).real
        )
    expected /= stars.grid.size * np.sqrt(
        atoms.get_masses()[supercell.primitive_site_indices[first]] * atoms.get_masses()[supercell.primitive_site_indices[second]]
    )
    assert actual == pytest.approx(expected, rel=0.12, abs=1e-5)


def test_ase_calculator_path_fits_harmonic_fc2_and_reports_energy() -> None:
    atoms, mapping, model = _system()
    calculator = _Taylor(model, mapping)
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=9)
    result = solver.run(300, calculator, pairs=12, max_iterations=2, tol_thz=1e-5, energy=True)
    assert result.converged
    assert len(result.history) == 1
    np.testing.assert_allclose(result.fc2.parameters(), model.parameters(), rtol=1e-8, atol=1e-8)
    assert result.history[0].free_energy is not None
    assert result.history[0].free_energy_error is not None
    assert result.history[0].mean_force_norm < 1e-10
    trial = HarmonicEnsemble(model, mapping, solver.stars, 300, statistics="quantum")
    assert result.history[0].free_energy == pytest.approx(trial.free_energy(), abs=1e-10)


def test_zero_temperature_quantum_sampling_is_defined_but_classical_is_not() -> None:
    atoms, mapping, model = _system()
    quantum = SSCHA(mapping, (2, 2, 2), initial=model, seed=3)
    ensemble = HarmonicEnsemble(model, mapping, quantum.stars, 0, statistics="quantum")
    assert np.any(ensemble.sample(2, seed=3, iteration=1))
    classical = SSCHA(mapping, (2, 2, 2), initial=model, statistics="classical", seed=3)
    with pytest.raises(ValueError, match="zero temperature"):
        classical.run(0, _Taylor(model, mapping), pairs=2)


def test_multiorder_map_cannot_be_silently_truncated_to_fc2() -> None:
    atoms, mapping, _ = _system()
    space = ClusterSpace(
        mapping.cluster_space.primitive_atoms,
        symprec=1e-5,
        cutoffs={2: 1.1, 4: 0.1},
        max_body_orders={2: 2, 4: 1},
    )
    mixed = ClusterMap(space, mapping.supercell_atoms)
    with pytest.raises(ValueError, match="FC2 only"):
        SSCHA(mixed, (2, 2, 2))


def test_unstable_trial_is_not_silently_sampled() -> None:
    atoms, mapping, model = _system()
    unstable = ForceConstants(model.cluster_space, {2: -model.coefficients[2]})
    solver = SSCHA(mapping, mapping.supercell_matrix, initial=unstable, seed=1)
    with pytest.raises(UnstableTrialError, match="positive Gaussian"):
        solver.run(300, _Taylor(model, mapping), pairs=4, max_iterations=1)

    bootstrapped = SSCHA(
        mapping,
        (2, 2, 2),
        initial=unstable,
        bootstrap_displacement=0.05,
        seed=1,
    ).run(300, _Taylor(model, mapping), pairs=12, max_iterations=2)
    assert bootstrapped.converged


def test_unstable_fitted_update_is_reported_without_modifying_the_trial() -> None:
    atoms, mapping, model = _system()
    unstable = ForceConstants(model.cluster_space, {2: -10.0 * model.coefficients[2]})
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=1)
    result = solver.run(
        300,
        _Taylor(unstable, mapping),
        pairs=12,
        mixing=1.0,
        max_backtracks=0,
        max_iterations=1,
    )
    assert result.status == "unstable"
    assert result.message is not None and "representative" in result.message
    np.testing.assert_array_equal(result.fc2.coefficients[2], model.coefficients[2])


class _OrderTerms:
    __slots__ = ("order", "atoms", "tensors", "components")

    def __init__(self, order, atoms, tensors, components):
        self.order, self.atoms, self.tensors, self.components = order, atoms, tensors, components


def _compile(model, mapping, order, translations):
    expanded = expand(model, order)
    atoms = np.empty((len(expanded.sites), len(translations), order), dtype=np.int32)
    tensors = np.empty((len(expanded.sites), 3**order), dtype=np.float64)
    for image, (sites, offsets, tensor) in enumerate(
        zip(expanded.sites, expanded.translations, expanded.tensors, strict=True)
    ):
        labels = ((0, 0, 0), *offsets)
        for cell_index, cell_translation in enumerate(translations):
            for axis, (site, offset) in enumerate(zip(sites, labels, strict=True)):
                shifted = tuple(
                    int(value + shift)
                    for value, shift in zip(offset, cell_translation, strict=True)
                )
                quotient = quotient_kernel(
                    np.asarray(shifted, dtype=np.int64),
                    adjugate_3x3(mapping.supercell_matrix),
                    abs(mapping.determinant),
                )
                hits = np.flatnonzero(
                    (mapping.primitive_site_indices == site)
                    & np.all(mapping.quotient_labels == quotient, axis=1)
                )
                if len(hits) != 1:
                    raise ValueError("lattice site is absent from the supercell")
                atoms[image, cell_index, axis] = int(hits[0])
        tensors[image] = np.asarray(tensor, dtype=np.float64).reshape(-1)
    return _OrderTerms(
        order=order,
        atoms=np.ascontiguousarray(atoms),
        tensors=np.ascontiguousarray(tensors),
        components=np.ascontiguousarray(tuple(np.ndindex((3,) * order)), dtype=np.int32),
    )


def _kernel(terms):
    atoms = terms.atoms.copy()
    tensors = terms.tensors.copy()
    components = terms.components.copy()
    factor = 1.0 / factorial(terms.order)
    order = terms.order

    @njit(nogil=True)
    def forces(displacement):
        result = np.zeros_like(displacement)
        for image in range(atoms.shape[0]):
            for translation in range(atoms.shape[1]):
                for component in range(components.shape[0]):
                    tensor_value = tensors[image, component]
                    for axis in range(order):
                        product = tensor_value
                        for other in range(order):
                            if other != axis:
                                product *= displacement[
                                    atoms[image, translation, other], components[component, other]
                                ]
                        result[atoms[image, translation, axis], components[component, axis]] -= (
                            factor * product
                        )
        return result

    return forces


class _Taylor(Calculator):
    """Test-local FC2 Taylor forces; the production calculator was removed."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model: ForceConstants, mapping: ClusterMap):
        super().__init__()
        if model.cluster_space.fingerprint != mapping.cluster_space.fingerprint:
            raise ValueError("force constants and cluster map use different cluster spaces")
        self._numbers = np.array(mapping.atomic_numbers, dtype=np.int32)
        self._cell = np.array(mapping.cell, dtype=np.float64, copy=True)
        self._positions = mapping.scaled_positions @ self._cell
        site0 = mapping.primitive_site_indices == 0
        translations = sorted(
            {tuple(int(value) for value in shift) for shift in mapping.lattice_translations[site0]}
        )
        if len(translations) != abs(mapping.determinant):
            raise ValueError("the mapping does not carry one translation per quotient")
        self._terms = _compile(model, mapping, 2, tuple(translations))
        self._forces = _kernel(self._terms)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if not np.array_equal(atoms.numbers, self._numbers):
            raise ValueError("atoms have a different supercell atom sequence")
        displacement, _ = find_mic(atoms.positions - self._positions, self._cell, pbc=True)
        displacement = np.ascontiguousarray(displacement)
        forces = self._forces(displacement)
        energy = -float(np.sum(displacement * forces)) / self._terms.order
        self.results = {"energy": energy, "forces": forces}


class _Counting(Calculator):
    implemented_properties: ClassVar[list[str]] = ["forces"]

    def __init__(self, model: ForceConstants, mapping: ClusterMap):
        super().__init__()
        self.inner = _Taylor(model, mapping)
        self.calls = 0

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        copy = atoms.copy()
        copy.calc = self.inner
        self.results = {"forces": copy.get_forces()}


def test_ase_calculator_recomputes_forces() -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell_matrix, initial=model, seed=2)
    calculator = _Counting(model, mapping)
    result = solver.run(300, calculator, pairs=6, max_iterations=1, tol_thz=1e-5)
    assert calculator.calls == 12
    assert result.history[0].free_energy is None
    with pytest.raises(Exception, match="energy"):
        solver.run(300, calculator, pairs=4, max_iterations=1, energy=True)


def test_gaussian_tail_cannot_be_silently_folded_in_the_fit() -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=2)
    calculator = _Counting(model, mapping)
    displacement = np.zeros((1, len(mapping.atomic_numbers), 3))
    displacement[0, 0, 0] = mapping.cell[0, 0]
    with pytest.raises(ValueError, match="minimum-image boundary"):
        solver._evaluate(displacement, calculator, energy=False)
    assert calculator.calls == 0


def test_cartesian_bootstrap_is_separate_from_canonical_iterations() -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell_matrix, seed=11)
    result = solver.run(300, _Taylor(model, mapping), pairs=12, max_iterations=2)
    assert result.converged
    assert len(result.history) == 1
    assert result.bootstrap_displacement == 0.01
    assert result.history[0].phase == "canonical"
    np.testing.assert_allclose(result.fc2.parameters(), model.parameters(), atol=1e-8)


def test_iteration_and_sampling_limits_are_not_reported_as_convergence(monkeypatch) -> None:
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, (2, 2, 2), initial=model, seed=3)
    calculator = _Taylor(model, mapping)
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
    atoms, mapping, model = _system()
    solver = SSCHA(mapping, mapping.supercell_matrix, initial=model, seed=12)
    calculator = _Taylor(model, mapping)
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
