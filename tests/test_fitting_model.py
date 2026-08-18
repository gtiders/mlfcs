import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from scipy import sparse

from mlfcs.fitting import ForceConstantFitter
from mlfcs.fitting.basis import convert_sparse_wick_reference as _wick_to_taylor_sparse
from mlfcs.fitting.basis import symmetrized_covariance as _symmetrized_covariance
from mlfcs.fitting.basis import wick as _wick
from mlfcs.fitting.basis import wick_axis_derivatives as _wick_axis_derivatives
from mlfcs.fitting.constraints import omitted_taylor_fc1
from mlfcs.fitting.design import ForceDesignOperator as _BatchedForceOperator
from mlfcs.fitting.design import physical_tile_shape as _physical_tile_shape
from mlfcs.fitting.design import predict_force
from mlfcs.fitting.design import prepare_design_kernel_groups as _prepare_physical_design_builders
from mlfcs.fitting.design import prepare_device_reduction as _prepare_device_reduction
from mlfcs.fitting.model import (
    _force_metrics,
    _order_force_rms_from_reduced_gram,
    _StreamingGramSystem,
)
from mlfcs.fitting.parameterization import OrderParameterization as _OrderTensor
from mlfcs.fitting.parameterization import expand_sparse as _expand_sparse
from mlfcs.fitting.solver import explicit_constraint_null_space
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants


def test_streaming_gram_zero_target_has_finite_zero_relative_error():
    system = _StreamingGramSystem(np.eye(2), np.zeros(2), 0.0)
    rmse, relative = system.force_metrics(np.zeros(2), np.zeros(2))

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


def test_wick_sparse_coefficients_are_converted_to_taylor_coefficients():
    fc3 = SparseOrderForceConstants(3, 1, 1, np.zeros((1, 3), dtype=int), np.ones((1, 3, 3, 3)))
    fc5 = SparseOrderForceConstants(
        5, 1, 1, np.zeros((1, 5), dtype=int), np.ones((1, 3, 3, 3, 3, 3))
    )
    covariance = np.diag([2.0, 3.0, 5.0])

    converted = _wick_to_taylor_sparse({3: fc3, 5: fc5}, covariance)

    # :u^5: = u^5 - 10 sigma u^3 + ..., while the IFC convention
    # divides the two potential terms by 5! and 3!, giving -sigma/2.
    np.testing.assert_allclose(converted[3].tensors[0], 1.0 - 0.5 * np.trace(covariance))
    np.testing.assert_allclose(converted[5].tensors, fc5.tensors)


def test_reduced_wick_transform_matches_sparse_tensor_conversion():
    from ase import Atoms

    from mlfcs.api import ForceConstantCalculation
    from mlfcs.fitting.constraints import build_wick_to_taylor_transform

    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    calculations = tuple(
        ForceConstantCalculation(
            primitive,
            order=order,
            supercell=(2, 1, 1),
            cutoff=4.1,
            verbose=False,
        )
        for order in (2, 3, 4)
    )
    rng = np.random.default_rng(41)
    n_parameters = sum(
        orbit.dimension for calculation in calculations for orbit in calculation.orbit_space.orbits
    )
    parameters = rng.normal(size=n_parameters)
    displacement = rng.normal(size=(20, 2, 3))
    covariance = _symmetrized_covariance(displacement, calculations[0])
    raw = _expand_sparse(parameters, calculations, 1, 2)
    expected = _wick_to_taylor_sparse(raw, covariance)
    transform = build_wick_to_taylor_transform(calculations, covariance)
    actual = _expand_sparse(np.asarray(transform @ parameters), calculations, 1, 2)

    for order in (2, 3, 4):
        np.testing.assert_array_equal(actual[order].clusters, expected[order].clusters)
        np.testing.assert_allclose(actual[order].tensors, expected[order].tensors, atol=1e-11)


def test_wick_to_taylor_conversion_preserves_polynomial_force():
    sigma = 0.7
    phi3 = 2.5
    phi5 = -1.2
    tensor3 = np.zeros((1, 3, 3, 3))
    tensor5 = np.zeros((1, 3, 3, 3, 3, 3))
    tensor3[(0, 0, 0, 0)] = phi3
    tensor5[(0, 0, 0, 0, 0, 0)] = phi5
    fc3 = SparseOrderForceConstants(3, 1, 1, np.zeros((1, 3), dtype=int), tensor3)
    fc5 = SparseOrderForceConstants(5, 1, 1, np.zeros((1, 5), dtype=int), tensor5)
    covariance = np.diag([sigma, 0.0, 0.0])
    converted = _wick_to_taylor_sparse({3: fc3, 5: fc5}, covariance)
    for displacement in (-1.3, -0.2, 0.8):
        wick_force = -phi3 * (displacement**2 - sigma) / 2
        wick_force -= phi5 * (displacement**4 - 6 * sigma * displacement**2 + 3 * sigma**2) / 24
        # Constants in the force are the derivative of an omitted FC1 term;
        # compare the displacement-dependent FC3+FC5 part represented by IFCs.
        taylor_force = -converted[3].tensors[(0, 0, 0, 0)] * displacement**2 / 2
        taylor_force -= converted[5].tensors[(0, 0, 0, 0, 0, 0)] * displacement**4 / 24
        wick_force_without_constant = wick_force - (phi3 * sigma / 2 - phi5 * sigma**2 / 8)
        np.testing.assert_allclose(taylor_force, wick_force_without_constant)


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
        cutoffs={3: None},
        verbose=False,
    )
    covariance = np.eye(6) * 0.04
    parameters = np.random.default_rng(4).normal(size=fitter.n_parameters)
    fc1 = omitted_taylor_fc1(fitter.calculations, parameters, covariance)
    force = np.asarray(
        predict_force(
            jnp.asarray(parameters),
            jnp.zeros((1, 2, 3)),
            jnp.asarray(covariance),
            fitter.order_tensors,
        )
    )[0]

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
        cutoffs={2: None},
        verbose=False,
    )

    def incomplete(self, scale, constraints, **kwargs):
        return np.zeros_like(scale), 7, 7, 1.0, 1.0

    monkeypatch.setattr(_StreamingGramSystem, "solve", incomplete)
    with pytest.raises(RuntimeError, match="did not converge"):
        fitter.fit(structures, validation_split=0, acoustic_sum_rule=False)
    result = fitter.fit(
        structures,
        validation_split=0,
        acoustic_sum_rule=False,
        allow_unconverged=True,
        cache_directory=tmp_path / "fit-cache",
    )
    assert result.diagnostics.stop_code == 7
    cached = tuple((tmp_path / "fit-cache").glob("gram-*/complete"))
    assert len(cached) == 1
    assert result.cache_directory == cached[0].parent


def test_fitter_uses_reordered_reference_without_a_separate_supercell_argument():
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))[[1, 0]]
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
        verbose=False,
    )

    np.testing.assert_array_equal(fitter.reference.numbers, reference.numbers)
    np.testing.assert_array_equal(fitter.canonical_supercell.numbers, reference.numbers)
    assert fitter.index.representative(0) == 1


def test_fitter_freezes_exact_external_fc2_and_fits_only_residual_orders(tmp_path):
    from mlfcs.fitting.frozen import frozen_forces, prepare_frozen_force_constants
    from mlfcs.io.hdf5 import read_hdf5, write_hdf5

    primitive = Atoms(
        "GaAs", cell=np.eye(3) * 5.6, scaled_positions=[[0, 0, 0], [0.25] * 3], pbc=True
    )
    reference = primitive.repeat((2, 1, 1))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3),
        cutoffs={2: None, 3: None},
        verbose=False,
    )
    count = sum(orbit.dimension for orbit in fitter.calculations[0].orbit_space.orbits)
    fc2 = _expand_sparse(
        np.linspace(0.2, 0.2 * count, count), fitter.calculations[:1], 2, len(reference)
    )[2]
    frozen_source = ForceConstants({}, reference.copy(), sparse={2: fc2}, relation=fitter.geometry)
    source_path = tmp_path / "finite-difference-fc2.h5"
    write_hdf5(source_path, frozen_source)
    frozen_source = read_hdf5(source_path)
    prepared = prepare_frozen_force_constants(
        {2: frozen_source},
        primitive=fitter.primitive,
        reference=fitter.reference,
        calculations=fitter.calculations,
    )
    rng = np.random.default_rng(91)
    displacements = rng.normal(scale=0.02, size=(8, len(reference), 3))
    forces, _ = frozen_forces(prepared, displacements, fitter.index)
    structures = []
    for displacement, force in zip(displacements, forces, strict=True):
        atoms = reference.copy()
        atoms.positions += displacement
        atoms.calc = SinglePointCalculator(atoms, forces=force)
        structures.append(atoms)

    result = fitter.fit(
        structures,
        frozen_force_constants={2: frozen_source},
        validation_split=0,
        acoustic_sum_rule=False,
        tolerance=1e-10,
    )

    np.testing.assert_array_equal(result.force_constants.sparse[2].clusters, fc2.clusters)
    np.testing.assert_array_equal(result.force_constants.sparse[2].tensors, fc2.tensors)
    assert result.diagnostics.frozen_orders == (2,)
    assert result.diagnostics.maximum_frozen_taylor_residual < 1e-10
    assert result.diagnostics.training_force_rmse < 1e-12


def test_frozen_force_constants_require_a_low_order_prefix_and_strict_solver():
    primitive = Atoms(
        "GaAs", cell=np.eye(3) * 5.6, scaled_positions=[[0, 0, 0], [0.25] * 3], pbc=True
    )
    reference = primitive.repeat((2, 1, 1))
    fitter = ForceConstantFitter(
        primitive, reference, orders=(2, 3), cutoffs={2: None, 3: None}, verbose=False
    )
    empty = ForceConstants({}, reference.copy(), relation=fitter.geometry)
    atoms = reference.copy()
    atoms.calc = SinglePointCalculator(atoms, forces=np.zeros((len(reference), 3)))

    with pytest.raises(ValueError, match="prefix starting at 2"):
        fitter.fit([atoms], frozen_force_constants={3: empty}, validation_split=0)
    with pytest.raises(ValueError, match="regularization=None and damping=0"):
        fitter.fit([atoms], frozen_force_constants={2: empty}, damping=1e-4, validation_split=0)


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
        cutoffs={2: None},
        verbose=False,
    )
    result = fitter.fit(
        structures,
        validation_split=0,
        acoustic_sum_rule=False,
        regularization="scaled_group_lasso",
        tolerance=1e-6,
        max_iterations=500,
    )

    assert result.diagnostics.stop_code == 0
    assert result.diagnostics.regularization == "scaled_group_lasso"
    assert result.diagnostics.effective_noise_scale > 0
    assert result.diagnostics.active_orbits == 1
    assert result.diagnostics.design_kernel_signatures > 0
    assert result.diagnostics.design_tiles > 0
    assert result.diagnostics.static_device_bytes > 0
    assert result.diagnostics.gram_feature_passes == 1
    assert result.diagnostics.prediction_feature_passes == 0
    assert result.force_constants.metadata["regularization"] == "scaled_group_lasso"


def test_streaming_gram_recovers_force_constant_and_force_error():
    rng = np.random.default_rng(12)
    displacement = rng.normal(size=(9, 1, 3))
    covariance = np.eye(3)
    tensor = _one_parameter_fc2_tensor()
    operator = _BatchedForceOperator(displacement, covariance, (tensor,), 1, batch_size=4)
    expected = np.array([2.75])
    target = operator.matvec(expected)
    gram = _StreamingGramSystem.from_operator(operator, target)
    scale = gram.exact_column_scale()
    actual = (
        gram.solve(
            scale,
            sparse.csr_matrix((0, 1)),
            tolerance=1e-12,
            max_iterations=100,
            damping=0.0,
            verbose=False,
        )[0]
        * scale
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    rmse, relative = _force_metrics(operator.matvec(actual), target)
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
    operator = _BatchedForceOperator(displacements, np.eye(3), (tensor,), n_orbits, batch_size=4)
    builders, _ = _prepare_physical_design_builders(operator)
    assert builders

    rows = int(np.prod(operator.force_shape))
    design = np.zeros((rows, n_orbits))
    displacement_batch = jnp.asarray(displacements)
    for group in builders:
        tiles = group.kernel(displacement_batch, operator.covariance, *group.device_arguments)
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
    physical = _BatchedForceOperator(displacements, np.eye(3), (tensor,), 2, batch_size=2)
    target = physical.matvec(np.asarray(parameter_map @ np.array([1.75])).reshape(-1))
    cpu = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        2,
        batch_size=2,
        parameter_map=parameter_map,
    )
    device = _BatchedForceOperator(
        displacements,
        np.eye(3),
        (tensor,),
        2,
        batch_size=2,
        parameter_map=parameter_map,
        device_gram=True,
    )
    expected = _StreamingGramSystem.from_operator(cpu, target)
    actual = _StreamingGramSystem.from_operator(device, target)
    np.testing.assert_allclose(actual.gram, expected.gram, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.rhs, expected.rhs, rtol=1e-12, atol=1e-12)


def test_mixed_order_diagnostics_use_one_joint_prediction_pass():
    class Operator:
        def __init__(self):
            self.calls = 0

        def matvec_by_order(self, _parameters):
            self.calls += 1
            return {2: np.array([3.0, 4.0]), 3: np.array([0.0, 12.0])}

        def matvec(self, _parameters):
            raise AssertionError("mixed-order diagnostics must not predict each order separately")

    operator = Operator()
    system = _StreamingGramSystem(np.eye(1), np.zeros(1), 1.0)
    result = _order_force_rms_from_reduced_gram(
        system,
        np.ones(1),
        sparse.csc_matrix([[1.0], [1.0]]),
        operator,
        np.array([2.0, 5.0]),
        (2, 3),
        (1, 1),
        2,
    )
    assert operator.calls == 1
    assert result == {2: pytest.approx(5 / np.sqrt(2)), 3: pytest.approx(12 / np.sqrt(2))}
