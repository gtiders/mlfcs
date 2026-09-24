"""Focused contracts for the streamed force-fitting system."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from _architecture_helpers import internal_dependencies
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.cluster_space import build_cluster_space
from mlfcs.core import PrimitiveCell
from mlfcs.core.errors import UnobservedParameterError
from mlfcs.fitting import FitSystem
from mlfcs.supercell import ClusterMap, Supercell


def ar_mapping(orders=(2,)) -> tuple[ClusterMap, Atoms]:
    primitive_atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={order: 0.1 for order in orders},
        max_body_orders={order: 1 for order in orders},
    )
    supercell_atoms = primitive_atoms.copy()
    supercell = Supercell.from_atoms(
        primitive,
        supercell_atoms,
        matrix=np.eye(3, dtype=np.int64),
    )
    return ClusterMap.build(space, supercell), supercell_atoms


def evaluated(supercell_atoms: Atoms, displacements, force) -> tuple[Atoms, ...]:
    structures = []
    for displacement in displacements:
        atoms = supercell_atoms.copy()
        atoms.positions += displacement
        atoms.calc = SinglePointCalculator(atoms, forces=force(np.asarray(displacement)))
        structures.append(atoms)
    return tuple(structures)


def representative_tensor(model, order: int) -> np.ndarray:
    block = model.space.block(order)
    assert block.orbits.stop - block.orbits.start == 1
    orbit = model.space.orbits[block.orbits.start]
    return (orbit.component_basis @ model.coefficients[order]).reshape((3,) * order)


def representative_tensors(model, order: int) -> dict[int, np.ndarray]:
    block = model.space.block(order)
    values = model.coefficients[order]
    tensors = {}
    offset = 0
    for orbit in model.space.orbits[block.orbits]:
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

    system = FitSystem.from_atoms(mapping, iter(structures))
    merged = FitSystem.from_atoms(mapping, structures[:2]) + FitSystem.from_atoms(
        mapping, structures[2:]
    )
    parameters = system.solve(rtol=1e-12)
    model = system.force_constants(parameters)

    assert system.n_structures == len(structures)
    assert system.n_equations == 3 * len(structures)
    assert system.matrix.shape == (mapping.space.n_parameters,) * 2
    assert system.unobserved_parameters == ()
    assert not system.matrix.flags.writeable
    assert not system.rhs.flags.writeable
    np.testing.assert_allclose(merged.matrix, system.matrix, atol=1e-18, rtol=1e-15)
    np.testing.assert_allclose(merged.rhs, system.rhs, atol=1e-18, rtol=1e-15)
    np.testing.assert_allclose(
        representative_tensor(model, 2), stiffness * np.eye(3), atol=1e-12, rtol=0.0
    )
    assert system.rmse(parameters) < 1e-14
    assert system.relative_error(parameters) < 1e-13


def test_zero_columns_can_be_merged_but_the_default_solver_refuses_them() -> None:
    mapping, _ = ar_mapping()
    count = mapping.space.n_parameters
    missing = FitSystem(mapping.space, np.zeros((count, count)), np.zeros(count), 0.0, 3, 1)
    observed = FitSystem(mapping.space, np.eye(count), np.ones(count), float(count), 3, 1)

    assert missing.unobserved_parameters == tuple(range(count))
    with pytest.raises(UnobservedParameterError, match="Regularization cannot recover"):
        missing.solve()

    combined = missing + observed
    assert combined.unobserved_parameters == ()
    np.testing.assert_allclose(combined.solve(rtol=1e-12), 1.0, atol=1e-12, rtol=0.0)


def test_residual_rejects_inconsistent_statistics_but_allows_roundoff() -> None:
    mapping, _ = ar_mapping()
    count = mapping.space.n_parameters
    matrix = np.eye(count)
    rhs = np.zeros(count)
    parameters = np.zeros(count)
    rhs[0] = 2.0
    parameters[0] = 2.0
    inconsistent = FitSystem(mapping.space, matrix, rhs, 1.0, 3, 1)

    with pytest.raises(ValueError, match="negative beyond roundoff"):
        inconsistent.residual(parameters)
    with pytest.raises(ValueError, match="negative beyond roundoff"):
        inconsistent.rmse(parameters)
    empty_count = FitSystem(mapping.space, matrix, rhs, 1.0, 0, 0)
    with pytest.raises(ValueError, match="negative beyond roundoff"):
        empty_count.rmse(parameters)

    rhs[0] = 1.0
    parameters[0] = 1.0
    rounded = FitSystem(mapping.space, matrix, rhs, 1.0 - np.finfo(float).eps, 3, 1)
    assert rounded.residual(parameters) == 0.0
    with pytest.raises(ValueError, match="parameters must be finite"):
        rounded.residual(np.full(count, np.nan))


def test_fit_system_accepts_only_atoms_with_stored_standard_forces() -> None:
    mapping, supercell_atoms = ar_mapping()
    atoms = supercell_atoms.copy()
    atoms.arrays["forces"] = np.zeros((len(atoms), 3))

    with pytest.raises(ValueError, match="no stored ASE forces"):
        FitSystem.from_atoms(mapping, [atoms])
    with pytest.raises(TypeError, match="iterable of ASE Atoms"):
        FitSystem.from_atoms(mapping, np.zeros((1, 1, 3)))


def test_joint_orders_and_pickle_preserve_the_complete_system() -> None:
    primitive_atoms = Atoms(
        numbers=[1, 2],
        scaled_positions=[[0.0, 0.0, 0.0], [0.17, 0.31, 0.23]],
        cell=[[3.1, 0.2, 0.1], [0.1, 3.7, 0.3], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={2: 0.01, 3: 0.01},
        max_body_orders={2: 1, 3: 1},
    )
    supercell_atoms = primitive_atoms.copy()
    supercell = Supercell.from_atoms(primitive, supercell_atoms, matrix=np.eye(3, dtype=np.int64))
    mapping = ClusterMap.build(space, supercell)
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

    system = FitSystem.from_atoms(mapping, structures)
    restored = pickle.loads(pickle.dumps(system))

    assert system.space.orders == (2, 3)
    assert system.unobserved_parameters == ()
    assert restored.fingerprint == system.fingerprint
    np.testing.assert_array_equal(restored.matrix, system.matrix)
    np.testing.assert_array_equal(restored.rhs, system.rhs)
    model = system.force_constants(system.solve(rtol=1e-11, max_steps=5000))
    for site, tensor in representative_tensors(model, 2).items():
        np.testing.assert_allclose(tensor, second[site], atol=1e-9, rtol=0.0)
    for site, tensor in representative_tensors(model, 3).items():
        np.testing.assert_allclose(tensor, third[site], atol=2e-8, rtol=0.0)


def test_fitting_dependency_direction_is_explicit() -> None:
    assert internal_dependencies("fitting") == {
        "cluster_space",
        "core",
        "force_constants",
        "supercell",
    }
