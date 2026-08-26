import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from scipy import sparse
from supercell_helpers import make_supercell

from mlfcs.fitting import ForceConstantFitter
from mlfcs.fitting.backends.wick.covariance import symmetrized_covariance as _symmetrized_covariance
from mlfcs.fitting.backends.wick.features import wick as _wick
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives as _wick_axis_derivatives
from mlfcs.fitting.backends.wick.lowering import lowered_fc1
from mlfcs.fitting.design_operator import ForceDesignOperator as _BatchedForceOperator
from mlfcs.fitting.design_operator import physical_tile_shape as _physical_tile_shape
from mlfcs.fitting.design_operator import (
    prepare_design_kernel_groups as _prepare_physical_design_builders,
)
from mlfcs.fitting.design_operator import prepare_device_reduction as _prepare_device_reduction
from mlfcs.fitting.gram import GramBuilder, GramStatistics
from mlfcs.fitting.linear_solvers import explicit_constraint_null_space
from mlfcs.fitting.parameterization import OrderParameterization as _OrderTensor
from mlfcs.force_constants.expansion import expand_fitted_orders as _expand_sparse


def test_fitter_fit_exposes_only_strict_solver_controls():
    signature = inspect.signature(ForceConstantFitter.fit)
    assert "damping" not in signature.parameters
    assert "frozen_force_constants" not in signature.parameters


def test_streaming_gram_zero_target_has_finite_zero_relative_error():
    system = GramStatistics(np.eye(2), np.zeros(2), 0.0, 2, {})
    rmse, relative = system.force_metrics(np.zeros(2))

    assert rmse == 0.0
    assert relative == 0.0


def test_explicit_constraint_parameterization_preserves_null_space():
    constraints = sparse.csr_matrix(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -2.0], [2.0, 2.0, 0.0, 0.0]]
    )

    parameter_map = explicit_constraint_null_space(constraints)

    assert parameter_map.shape == (4, 2)
    np.testing.assert_allclose((constraints @ parameter_map).toarray(), 0.0, atol=1e-13)
    assert np.linalg.matrix_rank(parameter_map.toarray()) == 2


def _one_parameter_fc2_tensor():
    representative = np.zeros((1, 9, 1))
    representative[0, 0, 0] = 1.0
    coordinates = np.zeros((1, 1, 1, 2), dtype=np.int32)
    return _OrderTensor(
        order=2,
        parameter_indices=np.zeros((1, 1), dtype=np.int32),
        parameter_mask=np.ones((1, 1), dtype=bool),
        representative_from_pivots=representative,
        rotations=np.eye(3).reshape(1, 1, 3, 3),
        component_permutations=np.arange(9).reshape(1, 1, 9),
        coordinates=coordinates,
        image_mask=np.ones((1, 1), dtype=bool),
    )


def test_wick_recursion_supports_arbitrary_degree():
    displacement = np.array([[2.0, 3.0, 5.0]])
    covariance = np.diag([0.5, 0.7, 1.1])
    coordinates = np.array([0, 0, 1, 2])

    actual = _wick(displacement, covariance, coordinates, order=4)
    expected = (2.0**2 - 0.5) * 3.0 * 5.0

    np.testing.assert_allclose(actual, expected)


def test_shared_wick_axis_derivatives_equal_independent_recursions():
    rng = np.random.default_rng(7)
    displacement = rng.normal(size=(2, 3))
    matrix = rng.normal(size=(6, 6))
    covariance = matrix @ matrix.T
    coordinates = rng.integers(0, 6, size=(4, 5))

    actual = _wick_axis_derivatives(displacement, covariance, coordinates, order=5)
    for axis, values in enumerate(actual):
        remaining = np.delete(np.arange(5), axis)
        expected = _wick(displacement, covariance, coordinates[..., remaining], order=4)
        np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-13)


def test_reduced_wick_transform_matches_sparse_tensor_conversion():
    from ase import Atoms

    from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
    from mlfcs.fitting.backends.wick.lowering import build_wick_to_taylor_transform

    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = make_supercell(primitive, (3, 3, 3))[0]
    calculations = tuple(
        FiniteDifferenceCalculation(
            primitive,
            order=order,
            reference=reference,
            cutoff=4.1,
        )
        for order in (2, 3, 4)
    )
    rng = np.random.default_rng(41)
    n_parameters = sum(
        orbit.dimension
        for calculation in calculations
        for orbit in calculation.realized_orbit_space.orbits
    )
    parameters = rng.normal(size=n_parameters)
    displacement = rng.normal(size=(20, len(reference), 3))
    covariance = _symmetrized_covariance(displacement, calculations[0])
    transform = build_wick_to_taylor_transform(calculations, covariance)
    actual = _expand_sparse(np.asarray(transform @ parameters), calculations)

    for order in (2, 3, 4):
        assert actual[order].tensors.shape[1:] == (3,) * order
        assert np.all(np.isfinite(actual[order].tensors))


def test_reported_omitted_fc1_reproduces_constant_wick_force():
    primitive = Atoms(
        "GaAs",
        cell=np.eye(3) * 5.6,
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    reference = primitive.copy()
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(3,),
        cutoffs={3: 3.0},
    )
    covariance = np.eye(6) * 0.04
    parameters = np.random.default_rng(4).normal(size=fitter.n_parameters)
    fc1 = lowered_fc1(fitter.calculations, parameters, covariance)
    operator = _BatchedForceOperator(
        np.zeros((1, 2, 3)),
        covariance,
        fitter.order_tensors,
        fitter.n_parameters,
        batch_size=1,
        axis_derivatives=wick_axis_derivatives,
    )
    force = operator.matvec(parameters).reshape(1, 2, 3)[0]

    np.testing.assert_allclose(force, -fc1, atol=1e-12, rtol=1e-12)


def test_unconverged_fit_requires_explicit_opt_in_and_exposes_gram_cache(monkeypatch, tmp_path):
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    structures = []
    for displacement in (-0.02, 0.02):
        atoms = reference.copy()
        atoms.positions[0, 0] += displacement
        forces = np.zeros((2, 3))
        forces[0, 0] = -displacement
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 4.1},
    )

    def incomplete(self, scale, constraints, **kwargs):
        return np.zeros_like(scale), 7, 7, 1.0, 1.0

    monkeypatch.setattr(GramStatistics, "solve", incomplete)
    gram = fitter.prepare_gram(structures, acoustic_sum_rule=False)
    with pytest.raises(RuntimeError, match="did not converge"):
        fitter.fit(gram, acoustic_sum_rule=False)
    result = fitter.fit(
        gram,
        acoustic_sum_rule=False,
        allow_unconverged=True,
    )
    assert result.stop_code == 7


def test_fitter_uses_reordered_reference_without_a_separate_supercell_argument():
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))[[1, 0]]
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
    )

    np.testing.assert_array_equal(fitter.reference.numbers, reference.numbers)
    np.testing.assert_array_equal(fitter.canonical_supercell.numbers, reference.numbers)
    assert fitter.index.representative(0) == 1


def test_fitter_reuses_one_reference_and_symmetry_frame_across_orders():
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((3, 3, 3))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3),
        cutoffs={2: 4.1, 3: 4.1},
    )

    first, second = fitter.calculations
    assert first.relation is second.relation is fitter.geometry
    assert first.index is second.index is fitter.index
    assert first.symmetry is second.symmetry
    assert first.frame.primitive_symmetry is second.frame.primitive_symmetry


def test_public_fitter_exposes_scaled_orbit_group_lasso():
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    structures = []
    for displacement in (-0.04, -0.02, 0.02, 0.04):
        atoms = reference.copy()
        atoms.positions[0, 0] += displacement
        forces = np.zeros((2, 3))
        forces[0, 0] = -2.0 * displacement
        forces[1, 0] = 2.0 * displacement
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 4.1},
    )
    gram = fitter.prepare_gram(structures, acoustic_sum_rule=False)
    result = fitter.fit(
        gram,
        acoustic_sum_rule=False,
        regularization="scaled_group_lasso",
        tolerance=1e-6,
        max_iterations=500,
    )

    assert result.stop_code == 0
    assert result.regularization == "scaled_group_lasso"
    assert result.effective_noise_scale > 0
    assert result.active_orbits == 2
    assert result.force_constants.metadata["regularization"] == "scaled_group_lasso"


def test_streaming_gram_recovers_force_constant_and_force_error():
    rng = np.random.default_rng(12)
    displacement = rng.normal(size=(9, 1, 3))
    covariance = np.eye(3)
    tensor = _one_parameter_fc2_tensor()
    operator = _BatchedForceOperator(
        displacement,
        covariance,
        (tensor,),
        1,
        batch_size=4,
        axis_derivatives=_wick_axis_derivatives,
    )
    expected = np.array([2.75])
    target = operator.matvec(expected)
    gram = GramBuilder.from_operator(operator, target)
    scale = gram.exact_column_scale()
    actual = (
        gram.solve(
            scale,
            sparse.csr_matrix((0, 1)),
            tolerance=1e-12,
            max_iterations=100,
        )[0]
        * scale
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    predicted = operator.matvec(actual)
    residual = predicted - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    assert rmse < 1e-12
    assert 100 * relative < 1e-9
    assert operator.program.gram_feature_passes == 1
    assert operator.program.prediction_feature_passes == 2


def test_physical_design_tiles_equal_matrix_free_operator():
    rng = np.random.default_rng(21)
    n_orbits = 35
    base = _one_parameter_fc2_tensor()
    tensor = _OrderTensor(
        order=2,
        parameter_indices=np.arange(n_orbits, dtype=np.int32).reshape(-1, 1),
        parameter_mask=np.ones((n_orbits, 1), dtype=bool),
        representative_from_pivots=np.repeat(base.representative_from_pivots, n_orbits, axis=0),
        rotations=np.repeat(base.rotations, n_orbits, axis=0),
        component_permutations=np.repeat(base.component_permutations, n_orbits, axis=0),
        coordinates=np.repeat(base.coordinates, n_orbits, axis=0),
        image_mask=np.repeat(base.image_mask, n_orbits, axis=0),
    )
    displacements = rng.normal(size=(5, 1, 3))
    operator = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        n_orbits,
        batch_size=4,
        axis_derivatives=_wick_axis_derivatives,
    )
    builders, _ = _prepare_physical_design_builders(operator)
    assert builders

    rows = int(np.prod(operator.force_shape))
    design = np.zeros((rows, n_orbits))
    displacement_batch = jnp.asarray(displacements)
    for group in builders:
        tiles = group.kernel(displacement_batch, operator.basis_state, *group.device_arguments)
        for tile, columns in zip(tiles, group.columns, strict=True):
            design[:, columns] += np.asarray(tile).reshape(rows, -1)

    parameters = rng.normal(size=n_orbits)
    np.testing.assert_allclose(
        design @ parameters,
        operator.matvec(parameters),
        rtol=1e-12,
        atol=1e-12,
    )
    by_order = operator.matvec_by_order(parameters)
    np.testing.assert_allclose(sum(by_order.values()), design @ parameters, rtol=1e-12, atol=1e-12)
    assert operator.with_displacements(displacements[:2]).program is operator.program
    assert operator.program.prediction_feature_passes == 2


def test_physical_tile_shape_splits_large_single_high_order_orbit():
    orbit_batch, image_batch, dimension_batch = _physical_tile_shape(
        order=5,
        n_orbits=1,
        n_images=480,
        n_dimensions=162,
        structure_batch=4,
        translations=12,
    )

    assert orbit_batch == 1
    assert image_batch < 480
    assert dimension_batch <= 162
    assert 4 * orbit_batch * image_batch * 12 * 3**5 * dimension_batch * 5 <= 32_000_000


def test_device_sparse_reduction_matches_scipy_without_host_round_trip():
    mapping = sparse.csc_matrix([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0], [-3.0, 0.0]])
    design = np.arange(12, dtype=float).reshape(3, 4)
    plan = _prepare_device_reduction(mapping, len(design), jax.devices()[0])
    actual = np.asarray(plan.apply(jax.device_put(design)))
    expected = np.asarray(mapping.T @ design.T).T
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_device_gram_pipeline_matches_cpu_with_a_sparse_null_space_map():
    rng = np.random.default_rng(81)
    base = _one_parameter_fc2_tensor()
    tensor = _OrderTensor(
        order=2,
        parameter_indices=np.arange(2, dtype=np.int32).reshape(-1, 1),
        parameter_mask=np.ones((2, 1), dtype=bool),
        representative_from_pivots=np.repeat(base.representative_from_pivots, 2, axis=0),
        rotations=np.repeat(base.rotations, 2, axis=0),
        component_permutations=np.repeat(base.component_permutations, 2, axis=0),
        coordinates=np.repeat(base.coordinates, 2, axis=0),
        image_mask=np.repeat(base.image_mask, 2, axis=0),
    )
    displacements = rng.normal(size=(5, 1, 3))
    parameter_map = sparse.csc_matrix([[1.0], [-0.5]])
    physical = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        2,
        batch_size=2,
        axis_derivatives=_wick_axis_derivatives,
    )
    target = physical.matvec(np.asarray(parameter_map @ np.array([1.75])).reshape(-1))
    cpu = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        2,
        batch_size=2,
        parameter_map=parameter_map,
        axis_derivatives=_wick_axis_derivatives,
    )
    device = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        2,
        batch_size=2,
        parameter_map=parameter_map,
        device_gram=True,
        axis_derivatives=_wick_axis_derivatives,
    )
    expected = GramBuilder.from_operator(cpu, target)
    actual = GramBuilder.from_operator(device, target)
    np.testing.assert_allclose(actual.gram, expected.gram, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.rhs, expected.rhs, rtol=1e-12, atol=1e-12)
