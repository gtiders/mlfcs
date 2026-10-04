"""Focused contracts for the streamed force-fitting system."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.cluster_space import ClusterSpace
from mlfcs.errors import UnobservedParameterError
from mlfcs.fitting import FitSystem
from mlfcs.mapping import ClusterMap


def ar_mapping(orders=(2,)) -> tuple[ClusterMap, Atoms]:
    primitive_atoms = bulk("Ar", "sc", a=1.0)
    primitive = primitive_atoms
    space = ClusterSpace(
        primitive,
        cutoffs={order: 0.1 for order in orders},
        max_body_orders={order: 1 for order in orders},
        symprec=1e-05,
    )
    supercell_atoms = primitive_atoms.copy()
    return ClusterMap(
        space, supercell_atoms, supercell_matrix=np.eye(3, dtype=np.int64)
    ), supercell_atoms


def evaluated(supercell_atoms: Atoms, displacements, force) -> tuple[Atoms, ...]:
    structures = []
    for displacement in displacements:
        atoms = supercell_atoms.copy()
        atoms.positions += displacement
        atoms.calc = SinglePointCalculator(atoms, forces=force(np.asarray(displacement)))
        structures.append(atoms)
    return tuple(structures)


def representative_tensor(model, order: int) -> np.ndarray:
    block = model.cluster_space.block(order)
    assert block.orbits.stop - block.orbits.start == 1
    orbit = model.cluster_space.orbits[block.orbits.start]
    return (orbit.component_basis @ model.coefficients[order]).reshape((3,) * order)


def representative_tensors(model, order: int) -> dict[int, np.ndarray]:
    block = model.cluster_space.block(order)
    values = model.coefficients[order]
    tensors = {}
    offset = 0
    for orbit in model.cluster_space.orbits[block.orbits]:
        stop = offset + orbit.dimension
        site = orbit.representative.sites[0].site
        tensors[site] = (orbit.component_basis @ values[offset:stop]).reshape((3,) * order)
        offset = stop
    return tensors


def test_fit_system_uses_the_streamed_optimized_normal_path() -> None:
    mapping, supercell_atoms = ar_mapping()
    stiffness = 2.5
    displacements = [
        [[0.01, 0.00, 0.00]],
        [[0.00, -0.02, 0.00]],
        [[0.00, 0.00, 0.03]],
        [[0.01, -0.02, 0.03]],
    ]
    structures = evaluated(supercell_atoms, displacements, lambda value: -stiffness * value)

    system = FitSystem(mapping, iter(structures))
    merged = FitSystem(mapping, structures[:2]) + FitSystem(mapping, structures[2:])
    model = system.solve(rtol=1e-12)
    parameters = model.parameters()

    assert system.n_structures == len(structures)
    assert system.n_equations == 3 * len(structures)
    assert system.normal_matrix.shape == (mapping.cluster_space.n_parameters,) * 2
    assert system.unobserved_parameters == ()
    assert not system.normal_matrix.flags.writeable
    assert not system.normal_rhs.flags.writeable
    np.testing.assert_allclose(merged.normal_matrix, system.normal_matrix, atol=1e-18, rtol=1e-15)
    np.testing.assert_allclose(merged.normal_rhs, system.normal_rhs, atol=1e-18, rtol=1e-15)
    np.testing.assert_allclose(
        representative_tensor(model, 2), stiffness * np.eye(3), atol=1e-12, rtol=0.0
    )
    assert system.rmse(parameters) < 1e-14
    assert system.relative_error(parameters) < 1e-13


def test_zero_columns_can_be_merged_but_the_default_solver_refuses_them() -> None:
    mapping, _ = ar_mapping()
    count = mapping.cluster_space.n_parameters
    missing = FitSystem._from_equations(
        mapping.cluster_space, "normal", np.zeros((count, count)), np.zeros(count), 0.0, 3, 1
    )
    observed = FitSystem._from_equations(
        mapping.cluster_space, "normal", np.eye(count), np.ones(count), float(count), 3, 1
    )

    assert missing.unobserved_parameters == tuple(range(count))
    with pytest.raises(UnobservedParameterError, match="Regularization cannot recover"):
        missing.solve()

    combined = missing + observed
    assert combined.unobserved_parameters == ()
    np.testing.assert_allclose(combined.solve(rtol=1e-12).parameters(), 1.0, atol=1e-12, rtol=0.0)


def test_residual_rejects_inconsistent_statistics_but_allows_roundoff() -> None:
    mapping, _ = ar_mapping()
    count = mapping.cluster_space.n_parameters
    matrix = np.eye(count)
    rhs = np.zeros(count)
    parameters = np.zeros(count)
    rhs[0] = 2.0
    parameters[0] = 2.0
    inconsistent = FitSystem._from_equations(
        mapping.cluster_space, "normal", matrix, rhs, 1.0, 3, 1
    )

    with pytest.raises(ValueError, match="negative beyond roundoff"):
        inconsistent.residual(parameters)
    with pytest.raises(ValueError, match="negative beyond roundoff"):
        inconsistent.rmse(parameters)
    empty_count = FitSystem._from_equations(mapping.cluster_space, "normal", matrix, rhs, 1.0, 0, 0)
    with pytest.raises(ValueError, match="negative beyond roundoff"):
        empty_count.rmse(parameters)

    rhs[0] = 1.0
    parameters[0] = 1.0
    rounded = FitSystem._from_equations(
        mapping.cluster_space, "normal", matrix, rhs, 1.0 - np.finfo(float).eps, 3, 1
    )
    assert rounded.residual(parameters) == 0.0
    with pytest.raises(ValueError, match="parameters must be finite"):
        rounded.residual(np.full(count, np.nan))


def test_fit_system_accepts_only_atoms_with_stored_standard_forces() -> None:
    mapping, supercell_atoms = ar_mapping()
    atoms = supercell_atoms.copy()
    atoms.arrays["forces"] = np.zeros((len(atoms), 3))

    with pytest.raises(ValueError, match="no stored ASE forces"):
        FitSystem(mapping, [atoms])
    with pytest.raises(TypeError, match="iterable of ASE Atoms"):
        FitSystem(mapping, np.zeros((1, 1, 3)))


def test_joint_orders_and_pickle_preserve_the_complete_system() -> None:
    primitive_atoms = Atoms(
        numbers=[1, 2],
        scaled_positions=[[0.0, 0.0, 0.0], [0.17, 0.31, 0.23]],
        cell=[[3.1, 0.2, 0.1], [0.1, 3.7, 0.3], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    primitive = primitive_atoms
    space = ClusterSpace(
        primitive,
        cutoffs={2: 0.01, 3: 0.01},
        max_body_orders={2: 1, 3: 1},
        symprec=1e-05,
    )
    supercell_atoms = primitive_atoms.copy()
    mapping = ClusterMap(space, supercell_atoms, supercell_matrix=np.eye(3, dtype=np.int64))
    second = np.zeros((2, 3, 3))
    third = np.zeros((2, 3, 3, 3))
    for site in range(2):
        for axis in range(3):
            second[site, axis, axis] = 2.0 + site + axis
            third[site, axis, axis, axis] = 0.5 + site + axis

    def force(displacement):
        values = np.empty((2, 3))
        for site in range(2):
            values[site] = -second[site] @ displacement[site]
            values[site] -= 0.5 * np.einsum(
                "ijk,j,k->i", third[site], displacement[site], displacement[site]
            )
        return values

    rng = np.random.default_rng(7)
    displacements = rng.normal(scale=0.03, size=(20, 2, 3))
    structures = evaluated(supercell_atoms, displacements, force)

    system = FitSystem(mapping, structures)
    restored = pickle.loads(pickle.dumps(system))

    assert system.cluster_space.orders == (2, 3)
    assert system.unobserved_parameters == ()
    assert restored.fingerprint == system.fingerprint
    np.testing.assert_array_equal(restored.normal_matrix, system.normal_matrix)
    np.testing.assert_array_equal(restored.normal_rhs, system.normal_rhs)
    model = system.solve(rtol=1e-11, maxiter=5000)
    for site, tensor in representative_tensors(model, 2).items():
        np.testing.assert_allclose(tensor, second[site], atol=1e-9, rtol=0.0)
    for site, tensor in representative_tensors(model, 3).items():
        np.testing.assert_allclose(tensor, third[site], atol=2e-8, rtol=0.0)


def test_fitting_dependency_direction_is_explicit() -> None:
    assert internal_dependencies("fitting") == {
        "errors",
        "core",
        "force_constants",
        "mapping",
        "_arrays",
        "cluster_space",
    }


def test_raw_and_normal_keep_physical_equations_and_match_external_solution():
    mapping, atoms = ar_mapping((2, 4))
    from mlfcs.fitting.design import ForceDesign

    design = ForceDesign(mapping)
    target = np.linspace(1.0, 2.0, design.n_parameters)
    rng = np.random.default_rng(8)
    structures = evaluated(
        atoms,
        rng.normal(scale=0.05, size=(25, 1, 3)),
        lambda displacement: (design.matrix(displacement) @ target).reshape(1, 3),
    )
    raw = FitSystem(mapping, iter(structures), representation="raw")
    normal = FitSystem(mapping, iter(structures))
    original_a, original_f = raw.design_matrix.copy(), raw.forces.copy()
    np.testing.assert_allclose(raw.design_matrix.T @ raw.design_matrix, normal.normal_matrix)
    np.testing.assert_allclose(raw.design_matrix.T @ raw.forces, normal.normal_rhs)
    converted = raw.to_normal()
    np.testing.assert_allclose(converted.normal_matrix, normal.normal_matrix)
    external = np.linalg.lstsq(raw.design_matrix, raw.forces, rcond=None)[0]
    raw_model = raw.solve(atol=1e-12, btol=1e-12)
    normal_model = normal.solve(rtol=1e-12)
    np.testing.assert_allclose(raw_model.parameters(), target, atol=1e-10)
    np.testing.assert_allclose(normal_model.parameters(), external, atol=1e-9)
    np.testing.assert_array_equal(raw.design_matrix, original_a)
    np.testing.assert_array_equal(raw.forces, original_f)
    assert raw.rmse(raw.force_constants(external)) < 1e-14
    assert raw.rmse(raw_model) < 1e-14
    with pytest.raises(ValueError, match="only available"):
        _ = raw.normal_matrix
    with pytest.raises(ValueError, match="different representations"):
        _ = raw + normal
    with pytest.raises(TypeError):
        raw.solve(rtol=1e-8)
    with pytest.raises(TypeError):
        normal.solve(atol=1e-8)


def test_lsmr_normalizes_disparate_columns_and_restores_physical_parameters():
    mapping, _ = ar_mapping((2, 4))
    n = mapping.cluster_space.n_parameters
    rng = np.random.default_rng(14)
    amplitudes = np.logspace(-150, 150, n)
    matrix = rng.normal(size=(40, n)) * amplitudes
    target = np.linspace(0.5, 1.5, n) / amplitudes
    forces = matrix @ target
    raw = FitSystem._from_equations(
        mapping.cluster_space, "raw", matrix, forces, float(forces @ forces), 40, 1
    )
    physical = raw.solve(atol=1e-12, btol=1e-12).parameters()
    np.testing.assert_allclose(physical * amplitudes, target * amplitudes, atol=1e-11)
    np.testing.assert_array_equal(raw.design_matrix, matrix)
    np.testing.assert_array_equal(raw.forces, forces)
    assert raw.relative_error(physical) < 1e-11


def test_raw_merge_pickle_zero_columns_and_iteration_failure():
    mapping, atoms = ar_mapping((2, 4))
    from mlfcs.fitting.design import ForceDesign

    design = ForceDesign(mapping)
    target = np.arange(design.n_parameters, dtype=float) + 1
    rng = np.random.default_rng(19)
    structures = evaluated(
        atoms,
        rng.normal(scale=0.1, size=(12, 1, 3)),
        lambda displacement: (design.matrix(displacement) @ target).reshape(1, 3),
    )
    raw = FitSystem(mapping, structures, representation="raw")
    merged = FitSystem(mapping, structures[:6], representation="raw") + FitSystem(
        mapping, structures[6:], representation="raw"
    )
    restored = pickle.loads(pickle.dumps(raw))
    np.testing.assert_array_equal(merged.design_matrix, raw.design_matrix)
    np.testing.assert_array_equal(restored.forces, raw.forces)
    assert restored.fingerprint == raw.fingerprint
    assert not restored.design_matrix.flags.writeable
    assert not restored.forces.flags.writeable
    with pytest.raises(RuntimeError, match="LSMR did not converge"):
        raw.solve(atol=0, btol=0, conlim=0, maxiter=1)
    zero = FitSystem(
        mapping, evaluated(atoms, [np.zeros((1, 3))], lambda u: u), representation="raw"
    )
    with pytest.raises(UnobservedParameterError):
        zero.solve()


def test_lsmr_recovers_the_noisy_least_squares_solution():
    mapping, _ = ar_mapping((2, 4))
    rng = np.random.default_rng(21)
    matrix = rng.normal(size=(50, mapping.cluster_space.n_parameters))
    forces = matrix @ np.arange(1, matrix.shape[1] + 1) + rng.normal(scale=0.1, size=50)
    raw = FitSystem._from_equations(
        mapping.cluster_space, "raw", matrix, forces, float(forces @ forces), 50, 1
    )
    expected = np.linalg.lstsq(matrix, forces, rcond=None)[0]
    model = raw.solve(atol=1e-12, btol=1e-12)
    np.testing.assert_allclose(model.parameters(), expected, atol=1e-10)
    np.testing.assert_allclose(raw.residual(model), np.linalg.norm(matrix @ expected - forces))
