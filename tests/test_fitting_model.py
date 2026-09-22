import inspect

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.fitting import ForceConstantFitter
from mlfcs.fitting.design_operator import ForceDesignOperator as _ForceDesignOperator
from mlfcs.fitting.design_operator import ForceDesignPlan as _ForceDesignPlan
from mlfcs.fitting.gram import GramBuilder, GramStatistics
from mlfcs.fitting.parameterization import OrderParameterization as _OrderTensor
from mlfcs.fitting.parameterization import image_parameter_basis


def test_fitter_fit_exposes_only_strict_solver_controls():
    signature = inspect.signature(ForceConstantFitter.fit)
    assert "damping" not in signature.parameters
    assert "frozen_force_constants" not in signature.parameters
    assert "acoustic_sum_rule" not in inspect.signature(ForceConstantFitter.prepare_gram).parameters


def test_streaming_gram_zero_target_has_finite_zero_relative_error():
    system = GramStatistics(np.eye(2), np.zeros(2), 0.0, 2, {})
    rmse, relative = system.force_metrics(np.zeros(2))

    assert rmse == 0.0
    assert relative == 0.0


def _one_parameter_fc2_tensor(n_orbits: int = 1, n_parameters: int = 1):
    representative = np.zeros((1, 9, 1))
    representative[0, 0, 0] = 1.0
    coordinates = np.zeros((1, 1, 1, 2), dtype=np.int32)
    return _OrderTensor(
        order=2,
        parameter_indices=np.arange(n_parameters, dtype=np.int32).reshape(-1, 1),
        parameter_mask=np.ones((n_orbits, 1), dtype=bool),
        cartesian_basis=np.repeat(representative, n_orbits, axis=0),
        rotations=np.eye(3).reshape(1, 1, 3, 3).repeat(n_orbits, axis=0),
        component_permutations=np.arange(9).reshape(1, 1, 9).repeat(n_orbits, axis=0),
        coordinates=np.repeat(coordinates, n_orbits, axis=0),
        image_mask=np.ones((n_orbits, 1), dtype=bool),
    )


def _numpy_reference_design(displacement, parameterization, n_parameters):
    """Independent NumPy transcription of the documented design contribution."""
    from math import factorial

    order = parameterization.order
    components = np.array(tuple(np.ndindex((3,) * order)))
    image_counts = np.count_nonzero(parameterization.image_mask, axis=1)
    dimension_counts = np.count_nonzero(parameterization.parameter_mask, axis=1)
    basis, basis_offsets = image_parameter_basis(parameterization, image_counts)
    out = np.zeros((displacement.size, n_parameters))
    pair = 0
    for orbit in range(parameterization.coordinates.shape[0]):
        dimensions = int(dimension_counts[orbit])
        for image in range(int(image_counts[orbit])):
            begin = int(basis_offsets[pair])
            block = basis[begin : begin + 3**order * dimensions].reshape(3**order, dimensions)
            flat = np.asarray(parameterization.coordinates[orbit, image])[..., None, :] * 3
            flat = flat + components
            values = displacement.reshape(-1)[flat]
            for axis in range(order):
                others = [slot for slot in range(order) if slot != axis]
                monomial = np.prod(values[..., others], axis=-1)
                contribution = monomial[..., None] * block[None, :, :]
                rows = np.broadcast_to(flat[..., axis][..., None], contribution.shape).reshape(-1)
                columns = np.broadcast_to(
                    parameterization.parameter_indices[orbit, :dimensions][None, None, :],
                    contribution.shape,
                ).reshape(-1)
                values_flat = -contribution.reshape(-1) / factorial(order)
                keep = values_flat != 0.0
                np.add.at(out, (rows[keep], columns[keep]), values_flat[keep])
            pair += 1
    return out


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

    def incomplete(self, scale, **kwargs):
        return np.zeros_like(scale), 7, 7, 1.0, 1.0

    monkeypatch.setattr(GramStatistics, "solve", incomplete)
    gram = fitter.prepare_gram(structures)
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


def test_regularization_argument_is_gone():
    """The basis-dependent scaled group LASSO was removed, not deprecated.

    Its penalty lived in column-normalized coordinates, so after the orbit basis became an
    orthonormal Cartesian one the same keyword no longer described the same optimization
    problem. The fitter accepts no regularization argument at all, and a caller who passes
    one gets a Python error naming it instead of a silent fallback to least squares.
    """
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
    fitter = ForceConstantFitter(primitive, reference, orders=(2,), cutoffs={2: 4.1})
    gram = fitter.prepare_gram(structures)

    with pytest.raises(TypeError, match="regularization"):
        fitter.fit(gram, acoustic_sum_rule=False, regularization="scaled_group_lasso")

    result = fitter.fit(gram, acoustic_sum_rule=False, tolerance=1e-6, max_iterations=500)
    assert result.stop_code == 0
    assert not hasattr(result, "regularization")
    assert result.force_constants.metadata["solver"] == "gram"


def test_one_physical_gram_supports_raw_and_post_projected_results():
    primitive = Atoms("Ar", cell=np.eye(3) * 4, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    structures = []
    for displacement in (-0.04, -0.02, 0.02, 0.04):
        atoms = reference.copy()
        atoms.positions[0, 0] += displacement
        forces = np.zeros((2, 3))
        forces[0, 0] = -2.0 * displacement
        forces[1, 0] = 1.5 * displacement
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)
    fitter = ForceConstantFitter(primitive, reference, orders=(2,), cutoffs={2: 4.1})
    gram = fitter.prepare_gram(structures)

    raw = fitter.fit(gram, acoustic_sum_rule=False, tolerance=1e-10)
    projected = fitter.fit(gram, acoustic_sum_rule=True, tolerance=1e-10)

    assert raw.gram_statistics is projected.gram_statistics is gram
    np.testing.assert_array_equal(raw.fitting_parameters, raw.unprojected_parameters)
    np.testing.assert_allclose(
        projected.unprojected_parameters,
        raw.unprojected_parameters,
        rtol=1e-12,
        atol=1e-12,
    )
    assert projected.maximum_asr_residual_after <= projected.maximum_asr_residual_before


def test_gram_merge_rejects_a_different_physical_design():
    identity = {
        "design_schema": 1,
        "design_fingerprint": "first",
        "physical_parameter_count": 2,
        "orders": [2],
    }
    first = GramStatistics(np.eye(2), np.ones(2), 2.0, 2, identity)
    second_identity = dict(identity, design_fingerprint="second")
    second = GramStatistics(np.eye(2), np.ones(2), 2.0, 2, second_identity)

    with pytest.raises(ValueError, match="different physical designs"):
        first.merge(second)


def test_physical_gram_design_identity_survives_save_and_load(tmp_path):
    identity = {
        "design_schema": 1,
        "design_fingerprint": "stable",
        "physical_parameter_count": 2,
        "orders": [2, 3],
    }
    statistics = GramStatistics(np.eye(2), np.ones(2), 2.0, 2, identity)
    path = tmp_path / "physical-gram.npz"

    statistics.save(path)
    restored = GramStatistics.load(path)

    assert restored.design_identity == identity
    np.testing.assert_array_equal(restored.gram, statistics.gram)
    np.testing.assert_array_equal(restored.rhs, statistics.rhs)


def test_streaming_gram_recovers_force_constant_and_force_error():
    rng = np.random.default_rng(12)
    displacement = rng.normal(size=(9, 1, 3))
    tensor = _one_parameter_fc2_tensor()
    operator = _ForceDesignOperator(displacement, (tensor,))
    design = np.concatenate([operator.design(index) for index in range(len(displacement))])
    expected = np.array([2.75])
    target = design @ expected
    gram = GramBuilder.from_operator(operator, target)
    scale = gram.exact_column_scale()
    actual = gram.solve(scale, tolerance=1e-12, max_iterations=100)[0] * scale
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    residual = design @ actual - target
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    assert rmse < 1e-12
    assert 100 * relative < 1e-9


def test_physical_gram_solver_handles_a_consistent_rank_deficient_system():
    statistics = GramStatistics(
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([2.0, 2.0]),
        4.0,
        2,
        {},
    )

    parameters, stop_code, iterations, residual, stationarity = statistics.solve(
        np.ones(2), tolerance=1e-12, max_iterations=100
    )

    assert stop_code == 0
    assert iterations > 0
    assert residual == 0.0
    assert stationarity < 1e-12
    np.testing.assert_allclose(parameters.sum(), 2.0, rtol=1e-12, atol=1e-12)


def test_column_scaling_keeps_every_nonzero_physical_column():
    statistics = GramStatistics(
        np.diag([1.0, 1e-30, 0.0]),
        np.zeros(3),
        0.0,
        1,
        {},
    )

    scale = statistics.exact_column_scale()
    np.testing.assert_array_equal(scale[[0, 2]], [1.0, 0.0])
    assert scale[1] == pytest.approx(1e15)


@pytest.mark.parametrize("diagonal", ([1.0, -1e-30], [1.0, np.inf], [1.0, np.nan]))
def test_column_scaling_rejects_invalid_gram_norms(diagonal):
    statistics = GramStatistics(np.diag(diagonal), np.zeros(2), 0.0, 1, {})
    with pytest.raises(ValueError, match="finite nonnegative"):
        statistics.exact_column_scale()


def test_snapshot_design_matches_the_numpy_reference():
    rng = np.random.default_rng(23)
    n_orbits = 7
    tensor = _one_parameter_fc2_tensor(n_orbits=n_orbits, n_parameters=n_orbits)
    displacements = rng.normal(size=(5, 1, 3))
    operator = _ForceDesignOperator(displacements, (tensor,))

    for index, displacement in enumerate(displacements):
        design = operator.design(index)
        reference = _numpy_reference_design(displacement, tensor, n_orbits)
        np.testing.assert_allclose(design, reference, rtol=1e-12, atol=1e-14)


def test_operator_reuses_one_plan_for_another_snapshot_subset():
    rng = np.random.default_rng(31)
    tensor = _one_parameter_fc2_tensor()
    displacements = rng.normal(size=(4, 1, 3))
    operator = _ForceDesignOperator(displacements, (tensor,))

    subset = operator.with_displacements(displacements[:1])

    assert subset.plan is operator.plan
    assert subset.n_parameters == operator.n_parameters
    np.testing.assert_allclose(subset.design(0), operator.design(0))


@pytest.mark.parametrize("shape", [(3, 3), (1, 1, 3, 1)])
def test_operator_rejects_displacements_without_snapshot_atom_axis_layout(shape):
    tensor = _one_parameter_fc2_tensor()

    with pytest.raises(ValueError, match=r"\(snapshots, atoms, 3\)"):
        _ForceDesignOperator(np.zeros(shape), (tensor,))


def test_operator_normalizes_valid_displacements_to_force_rows():
    tensor = _one_parameter_fc2_tensor()
    operator = _ForceDesignOperator(np.zeros((4, 1, 3)), (tensor,))

    assert operator.force_shape == (4, 1, 3)
    assert operator.displacements.shape == (4, 3)
    assert operator.rows_per_snapshot == 3


def test_gram_matches_the_physical_design():
    rng = np.random.default_rng(81)
    tensor = _one_parameter_fc2_tensor(n_orbits=2, n_parameters=2)
    displacements = rng.normal(size=(5, 1, 3))
    operator = _ForceDesignOperator(displacements, (tensor,))
    design = np.concatenate([operator.design(index) for index in range(len(displacements))])
    target = design @ np.array([1.75, -0.5])

    statistics = GramBuilder.from_operator(operator, target)

    np.testing.assert_allclose(statistics.gram, design.T @ design, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(statistics.rhs, design.T @ target, rtol=1e-12, atol=1e-12)
    assert "parameter_map" not in statistics.metadata


def test_order_design_rejects_non_prefix_parameter_masks():
    tensor = _one_parameter_fc2_tensor(n_orbits=2, n_parameters=3)
    parameter_mask = np.asarray([[True, True, False], [True, False, True]])
    masked = _OrderTensor(
        order=tensor.order,
        parameter_indices=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        parameter_mask=parameter_mask,
        cartesian_basis=np.repeat(tensor.cartesian_basis, 2, axis=0),
        rotations=np.repeat(tensor.rotations, 2, axis=0),
        component_permutations=np.repeat(tensor.component_permutations, 2, axis=0),
        coordinates=np.repeat(tensor.coordinates, 2, axis=0),
        image_mask=np.ones((2, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="leading block"):
        _ForceDesignPlan.compile((masked,))


def test_order_design_rejects_non_contiguous_parameter_blocks():
    tensor = _one_parameter_fc2_tensor()
    scrambled = _OrderTensor(
        order=tensor.order,
        parameter_indices=np.asarray([[0, 2]], dtype=np.int32),
        parameter_mask=np.ones((1, 2), dtype=bool),
        cartesian_basis=np.repeat(tensor.cartesian_basis, 1, axis=0),
        rotations=np.repeat(tensor.rotations, 1, axis=0),
        component_permutations=np.repeat(tensor.component_permutations, 1, axis=0),
        coordinates=np.repeat(tensor.coordinates, 1, axis=0),
        image_mask=np.ones((1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="contiguous block"):
        _ForceDesignPlan.compile((scrambled,))
