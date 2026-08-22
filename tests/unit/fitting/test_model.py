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
from mlfcs.fitting.model import (
    _force_metrics,
    _StreamingGramSystem,
)
from mlfcs.fitting.parameterization import OrderParameterization as _OrderTensor
from mlfcs.fitting.parameterization import expand_sparse as _expand_sparse
from mlfcs.model import SparseOrderForceConstants


def test_streaming_gram_zero_target_has_finite_zero_relative_error():
    system = _StreamingGramSystem(np.eye(2), np.zeros(2), 0.0)
    rmse, relative = system.force_metrics(np.zeros(2), np.zeros(2))

    assert rmse == 0.0
    assert relative == 0.0


def _one_parameter_fc2_tensor():
    representative = np.zeros((1, 9, 1))
    representative[0, 0, 0] = 1.0
    components = np.asarray(tuple(np.ndindex(3, 3)), dtype=np.int32)
    coordinates = components.reshape(1, 1, 1, 9, 2)
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
        supercell=(1, 1, 1),
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


def test_unconverged_fit_requires_explicit_opt_in(monkeypatch):
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
        supercell=(2, 1, 1),
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
    )
    assert result.diagnostics.stop_code == 7


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
        design += np.asarray(
            group.kernel(displacement_batch, *map(jnp.asarray, group.arguments))
        ).reshape(rows, n_orbits)

    parameters = rng.normal(size=n_orbits)
    np.testing.assert_allclose(
        design @ parameters,
        operator.matvec(parameters),
        rtol=1e-12,
        atol=1e-12,
    )


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
