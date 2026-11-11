"""Numerical behavior of invariance."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.expansion import expand_lattice_tensors


def physical_fc2_moments(model: ForceConstants) -> tuple[float, float]:
    """Compute Born-Huang and Huang residuals from expanded FC2.

    Sum tensors times separation and tensors times the separation
    outer product. Antisymmetrize the first moment on its last two axes and
    the globally summed second moment under exchange of axis pairs. Return
    their maximum magnitudes in eV/angstrom and eV."""
    cell = model.cluster_space
    first_moment = np.zeros((cell.n_atoms, 3, 3, 3))
    second_moment = np.zeros((cell.n_atoms, 3, 3, 3, 3))
    values = expand_lattice_tensors(model, 2)
    for sites, translations, tensor in zip(
        values.sites, values.translations, values.tensors, strict=True
    ):
        first, second = sites
        vector = (
            cell.cartesian_positions[second]
            - cell.cartesian_positions[first]
            + np.asarray(translations[0]) @ cell.cell
        )
        first_moment[first] += tensor[:, :, None] * vector[None, None, :]
        second_moment[first] += (
            tensor[:, :, None, None] * np.outer(vector, vector)[None, None, :, :]
        )
    born_huang = first_moment - np.swapaxes(first_moment, 2, 3)
    global_second_moment = second_moment.sum(axis=0)
    huang = global_second_moment - np.transpose(global_second_moment, (2, 3, 0, 1))
    return (
        float(np.max(np.abs(born_huang), initial=0.0)),
        float(np.max(np.abs(huang), initial=0.0)),
    )


def test_rotational_projection_preserves_higher_orders() -> None:
    """Verify rotational moments, idempotence and unchanged higher orders."""
    cell = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(
        cell, cutoffs={2: 1.1, 3: 1.1}, max_body_orders={2: 2, 3: 3}, symprec=1e-05, asr=True
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
    coordinates = space.acoustic_coordinates(2)
    acoustic_model = ForceConstants(
        space,
        {**model.coefficients, 2: coordinates.lift(rng.normal(size=coordinates.dimension))},
    )
    result = acoustic_model.enforce_rotation(born_huang=True, huang=True)
    assert result.relative_after <= 1e-10
    assert result.born_huang_after is not None
    assert result.huang_after is not None
    assert result.born_huang_after <= result.length_scale * 1e-09
    assert result.huang_after <= result.length_scale**2 * 1e-09
    np.testing.assert_array_equal(result.force_constants.coefficients[3], model.coefficients[3])
    assert result.retained_rank >= 0
    assert result.structural_rank_gap > 1e3
    assert result.structural_rank_bound >= 0.0
    physical = physical_fc2_moments(result.force_constants)
    assert physical[0] == pytest.approx(result.born_huang_after)
    assert physical[1] == pytest.approx(result.huang_after)
    repeated = result.force_constants.enforce_rotation(born_huang=True, huang=True)
    np.testing.assert_allclose(
        repeated.force_constants.coefficients[2],
        result.force_constants.coefficients[2],
        rtol=1e-10,
        atol=1e-10,
    )


def test_rotation_defaults_to_born_huang() -> None:
    """Verify rotation defaults to born huang."""
    cell = bulk("Ar", "sc", a=1.0)
    space = ClusterSpace(cell, cutoffs={2: 1.1}, max_body_orders={2: 2}, symprec=1e-05, asr=True)
    model = ForceConstants(space, {2: np.ones(space.n_parameters)})
    born_only = model.enforce_rotation()
    assert born_only.born_huang
    assert not born_only.huang
    assert born_only.structural_rank_gap > 1e3


@pytest.mark.parametrize(
    "cell_error,site_error,symprec",
    [(3e-07, 1e-07, 1e-05), (0.0, 0.0, 0.01), (0.001, 0.0001, 0.01)],
)
def test_rotational_rank_is_structural_for_perturbed_cells(
    cell_error: float, site_error: float, symprec: float
) -> None:
    """Verify perturbed cells keep the idealized structural rank."""
    a = 3.14879776
    cell = np.array([[a, 0.0, 0.0], [-a / 2, np.sqrt(3) * a / 2, 0.0], [0.0, 0.0, 27.14312451]])
    positions = np.array(
        [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.44212744], [2 / 3, 1 / 3, 0.55787256]]
    )
    cell[1, 1] -= cell_error
    positions[0, 1] -= 2 * site_error
    positions[1:, 0] -= site_error
    cell_atoms = Atoms(numbers=[42, 16, 16], cell=cell, scaled_positions=positions, pbc=True)
    space = ClusterSpace(
        cell_atoms, cutoffs={2: 8.0}, max_body_orders={2: 2}, symprec=symprec, asr=True
    )
    coordinates = space.acoustic_coordinates(2)
    # Match the physical coefficient scale of the measured-geometry fixture;
    # a dense lift is an independent small-test oracle, never production data.
    basis = np.column_stack([coordinates.lift(column) for column in np.eye(coordinates.dimension)])
    observations = np.random.default_rng(0).normal(size=space.n_parameters)
    free = np.linalg.lstsq(basis, observations, rcond=None)[0]
    acoustic_model = ForceConstants(space, {2: coordinates.lift(free)})
    result = acoustic_model.enforce_rotation(huang=True)
    assert result.retained_rank == 2
    assert result.structural_rank_gap > 1e3
    assert result.structural_rank_bound > 0.0
    assert result.idealization_position_residual <= symprec
    assert result.idealization_symmetry_closure <= 1e-10
    allowed_fraction = 0.001 if cell_error > 0.0001 else 0.0001
    assert result.born_huang_after < result.born_huang_before * allowed_fraction
    assert result.huang_after < result.huang_before * allowed_fraction
