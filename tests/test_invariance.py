"""Numerical behavior of invariance."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from scipy import sparse

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.asr import _project
from mlfcs.force_constants.expansion import expand_lattice_tensors
from mlfcs.force_constants.rotation import _fc2_moment_matrices


def physical_asr_residual(model: ForceConstants, order: int) -> float:
    """Compute the maximum absolute acoustic sum-rule residual for one order.

    Sum expanded Cartesian tensors over the last site for each anchored
    prefix. Return the largest component magnitude in eV/angstrom**order."""
    sums: dict[tuple[object, ...], np.ndarray] = {}
    values = expand_lattice_tensors(model, order)
    for sites, translations, tensor in zip(
        values.sites, values.translations, values.tensors, strict=True
    ):
        prefix = (sites[:-1], translations[:-1])
        sums[prefix] = sums.get(prefix, 0.0) + tensor
    return max((float(np.max(np.abs(value))) for value in sums.values()), default=0.0)


def physical_fc2_moments(model: ForceConstants) -> tuple[float, float, float]:
    """Compute acoustic, Born-Huang and Huang residuals from expanded FC2.

    Sum tensors, tensors times separation, and tensors times the separation
    outer product. Antisymmetrize the first moment on its last two axes and
    the globally summed second moment under exchange of axis pairs. Return
    their maximum magnitudes in eV/angstrom**2, eV/angstrom and eV."""
    primitive = model.cluster_space
    acoustic = np.zeros((primitive.n_atoms, 3, 3))
    first_moment = np.zeros((primitive.n_atoms, 3, 3, 3))
    second_moment = np.zeros((primitive.n_atoms, 3, 3, 3, 3))
    values = expand_lattice_tensors(model, 2)
    for sites, translations, tensor in zip(
        values.sites, values.translations, values.tensors, strict=True
    ):
        first, second = sites
        vector = (
            primitive.cartesian_positions[second]
            - primitive.cartesian_positions[first]
            + np.asarray(translations[0]) @ primitive.cell
        )
        acoustic[first] += tensor
        first_moment[first] += tensor[:, :, None] * vector[None, None, :]
        second_moment[first] += (
            tensor[:, :, None, None] * np.outer(vector, vector)[None, None, :, :]
        )
    born_huang = first_moment - np.swapaxes(first_moment, 2, 3)
    global_second_moment = second_moment.sum(axis=0)
    huang = global_second_moment - np.transpose(global_second_moment, (2, 3, 0, 1))
    return (
        float(np.max(np.abs(acoustic), initial=0.0)),
        float(np.max(np.abs(born_huang), initial=0.0)),
        float(np.max(np.abs(huang), initial=0.0)),
    )


def test_asr_projection_matches_the_minimum_norm_dense_oracle() -> None:
    """Verify asr projection matches the minimum norm dense oracle."""
    matrix = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0], [1.0, -1.0, 0.0, 0.0]])
    values = np.array([2.0, -3.0, 7.0, 11.0])
    projected, report = _project(2, sparse.csr_matrix(matrix), values, rtol=1e-12)
    oracle = values - matrix.T @ np.linalg.pinv(matrix @ matrix.T) @ (matrix @ values)
    np.testing.assert_allclose(projected, oracle, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(matrix @ projected, 0.0, atol=1e-11)
    assert report.relative_after <= 1e-12
    assert report.correction_norm > 0.0


def test_force_constants_project_each_order_without_a_supercell() -> None:
    """Verify force constants project each order without a supercell."""
    primitive = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(
        primitive, cutoffs={2: 1.1, 3: 1.1}, max_body_orders={2: 2, 3: 3}, symprec=1e-05
    )
    rng = np.random.default_rng(4)
    model = ForceConstants(
        space,
        {
            order: rng.normal(
                size=space.block(order).parameters.stop - space.block(order).parameters.start
            )
            for order in space.orders
        },
    )
    result = model.enforce_asr(rtol=1e-10)
    assert tuple(report.order for report in result.reports) == (2, 3)
    assert all(report.relative_after <= 1e-10 for report in result.reports)
    assert any(report.correction_norm > 0.0 for report in result.reports)
    for report in result.reports:
        assert physical_asr_residual(result.force_constants, report.order) == pytest.approx(
            report.residual_after
        )
    repeated = result.force_constants.enforce_asr(rtol=1e-10)
    for order in model.orders:
        np.testing.assert_allclose(
            repeated.force_constants.coefficients[order],
            result.force_constants.coefficients[order],
            rtol=1e-11,
            atol=1e-11,
        )


def test_asr_order_selection_preserves_other_orders_exactly() -> None:
    """Verify asr order selection preserves other orders exactly."""
    primitive = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(
        primitive, cutoffs={2: 1.1, 3: 1.1}, max_body_orders={2: 2, 3: 3}, symprec=1e-05
    )
    model = ForceConstants(
        space,
        {
            order: np.arange(
                1,
                space.block(order).parameters.stop - space.block(order).parameters.start + 1,
                dtype=float,
            )
            for order in space.orders
        },
    )
    result = model.enforce_asr(orders=(2,))
    np.testing.assert_array_equal(result.force_constants.coefficients[3], model.coefficients[3])
    assert result.report(2).order == 2


def test_rotational_projection_preserves_asr_and_higher_orders() -> None:
    """Verify rotational projection preserves asr and higher orders."""
    primitive = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(
        primitive, cutoffs={2: 1.1, 3: 1.1}, max_body_orders={2: 2, 3: 3}, symprec=1e-05
    )
    rng = np.random.default_rng(12)
    model = ForceConstants(
        space,
        {
            order: rng.normal(
                size=space.block(order).parameters.stop - space.block(order).parameters.start
            )
            for order in space.orders
        },
    )
    unprojected = model.enforce_rotation(born_huang=True, huang=True)
    assert unprojected.acoustic_before > 0.0
    assert unprojected.acoustic_after == pytest.approx(unprojected.acoustic_before, abs=1e-12)
    acoustic_model = model.enforce_asr(orders=(2,)).force_constants
    result = acoustic_model.enforce_rotation(born_huang=True, huang=True)
    assert result.relative_after <= 1e-10
    assert result.acoustic_after == pytest.approx(result.acoustic_before, abs=1e-12)
    assert result.born_huang_after is not None
    assert result.huang_after is not None
    assert result.born_huang_after <= result.length_scale * 1e-09
    assert result.huang_after <= result.length_scale**2 * 1e-09
    np.testing.assert_array_equal(result.force_constants.coefficients[3], model.coefficients[3])
    assert result.retained_rank >= 0
    assert result.rank_cutoff >= 0.0
    physical = physical_fc2_moments(result.force_constants)
    assert physical[0] == pytest.approx(result.acoustic_after)
    assert physical[1] == pytest.approx(result.born_huang_after)
    assert physical[2] == pytest.approx(result.huang_after)
    repeated = result.force_constants.enforce_rotation(born_huang=True, huang=True)
    np.testing.assert_allclose(
        repeated.force_constants.coefficients[2],
        result.force_constants.coefficients[2],
        rtol=1e-10,
        atol=1e-10,
    )


def test_rotation_defaults_to_born_huang() -> None:
    """Verify rotation defaults to born huang."""
    primitive = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(primitive, cutoffs={2: 1.1}, max_body_orders={2: 2}, symprec=1e-05)
    model = ForceConstants(space, {2: np.ones(space.n_parameters)})
    born_only = model.enforce_rotation()
    assert born_only.born_huang
    assert not born_only.huang


@pytest.mark.parametrize(
    "cell_error,site_error,symprec",
    [(3e-07, 1e-07, 1e-05), (0.0, 0.0, 0.01), (0.001, 0.0001, 0.01)],
)
def test_rotational_rank_uses_measured_geometry_not_symprec(
    cell_error: float, site_error: float, symprec: float
) -> None:
    """Verify rotational rank uses measured geometry not symprec."""
    a = 3.14879776
    cell = np.array([[a, 0.0, 0.0], [-a / 2, np.sqrt(3) * a / 2, 0.0], [0.0, 0.0, 27.14312451]])
    positions = np.array(
        [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.44212744], [2 / 3, 1 / 3, 0.55787256]]
    )
    cell[1, 1] -= cell_error
    positions[0, 1] -= 2 * site_error
    positions[1:, 0] -= site_error
    primitive = Atoms(numbers=[42, 16, 16], cell=cell, scaled_positions=positions, pbc=True)
    space = ClusterSpace(primitive, cutoffs={2: 8.0}, max_body_orders={2: 2}, symprec=symprec)
    model = ForceConstants(space, {2: np.random.default_rng(0).normal(size=space.n_parameters)})
    acoustic_model = model.enforce_asr().force_constants
    huang = _fc2_moment_matrices(acoustic_model.cluster_space)[1]
    assert huang.shape == (81, space.n_parameters)
    result = acoustic_model.enforce_rotation(huang=True)
    assert result.retained_rank == 2
    assert result.smallest_retained_singular_value > result.rank_cutoff
    assert result.largest_discarded_singular_value <= result.rank_cutoff
    assert result.acoustic_after == pytest.approx(result.acoustic_before, abs=1e-12)
    allowed_fraction = 0.001 if cell_error > 0.0001 else 0.0001
    assert result.born_huang_after < result.born_huang_before * allowed_fraction
    assert result.huang_after < result.huang_before * allowed_fraction
    if cell_error:
        assert result.geometry_residual > 1e-07
        assert result.rank_cutoff > 1e-06
        if cell_error > 0.0001:
            assert result.geometry_residual > 0.0001
            assert result.rank_cutoff > 0.001
    else:
        assert result.geometry_residual < 1e-12
        assert result.rank_cutoff < 1e-10
    disabled = acoustic_model.enforce_rotation(huang=True, rank_rtol=1.0)
    assert disabled.retained_rank == 0
    np.testing.assert_array_equal(
        disabled.force_constants.coefficients[2], acoustic_model.coefficients[2]
    )
