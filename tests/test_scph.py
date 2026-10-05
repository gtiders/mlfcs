"""Numerical behavior of scph."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import ForceConstants
from mlfcs.phonon import SCPH, Harmonic
from mlfcs.phonon.dynamics import translation_complement
from mlfcs.phonon.scph import (
    _VARIANCE,
    _contract,
    _fc2_tensors,
    _modal_covariance,
)


def test_modal_covariance_matches_zero_point_and_gamma_subspace() -> None:
    """Verify modal covariance matches zero point and gamma subspace."""
    matrix = np.diag([4.0, 9.0, 16.0]).astype(complex)
    covariance = _modal_covariance(
        matrix, np.asarray([1.0]), 0.0, gamma=False, statistics="quantum"
    )
    np.testing.assert_allclose(np.diag(covariance), _VARIANCE / np.asarray([2.0, 3.0, 4.0]))
    basis = translation_complement(np.asarray([1.0, 4.0]))
    assert basis.shape == (6, 3)
    for axis in range(3):
        translation = np.zeros(6)
        translation[axis::3] = [1.0, 2.0]
        np.testing.assert_allclose(translation @ basis, 0.0, atol=1e-15)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-15)


def test_quartic_loop_contraction_contains_the_one_half_factor() -> None:
    """Verify quartic loop contraction contains the one half factor."""
    tensor = np.zeros((1, 3, 3, 3, 3))
    tensor[0, 0, 0, 0, 0] = 2.0
    covariance = np.zeros((1, 3, 3), dtype=complex)
    covariance[0, 0, 0] = 3.0
    actual = _contract(tensor, covariance, np.asarray([0]), np.asarray([0]), 1)
    assert actual[0, 0, 0] == pytest.approx(3.0)


def test_scph_uses_mass_preserving_stars() -> None:
    """Verify scph uses mass preserving stars."""
    space = ClusterSpace(bulk("Si", "diamond", a=5.43), cutoffs={2: 0.01, 4: 0.01})
    space = space.with_masses([28.0, 29.0])
    model = ForceConstants(
        space,
        {
            2: np.ones(space.block(2).parameters.stop - space.block(2).parameters.start),
            4: np.zeros(space.block(4).parameters.stop - space.block(4).parameters.start),
        },
    )
    scph = SCPH(model, (2, 2, 2))
    assert scph.stars.symmetry.size < space.symmetry.size
    result = scph.run(100, max_iterations=2)
    assert result.converged
    np.testing.assert_allclose(
        result.frequencies, Harmonic(result.fc2).frequencies(scph.stars), atol=1e-12
    )


def test_temperature_sequence_returns_converged_phonons() -> None:
    """Verify temperature sequence returns converged phonons."""
    space = ClusterSpace(
        bulk("Ar", "sc", a=1.0),
        symprec=1e-05,
        cutoffs={2: 1.1, 4: 0.1},
        max_body_orders={2: 2, 4: 1},
    )
    coefficients = []
    for orbit in space.orbits[space.block(2).orbits]:
        tensor = (
            6.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else -1.0
        ) * np.eye(3)
        parameters = np.linalg.lstsq(orbit.component_basis, tensor.reshape(-1), rcond=None)[0]
        coefficients.extend(parameters)
    model = ForceConstants(
        space,
        {
            2: np.asarray(coefficients),
            4: np.zeros(space.block(4).parameters.stop - space.block(4).parameters.start),
        },
    )
    scph = SCPH(model, (2, 2, 2))
    np.testing.assert_allclose(
        _fc2_tensors(
            model.coefficients[2], scph._fc2_bases, scph._fc2_offsets, scph._fc2_dimensions
        ),
        Harmonic(model)._tensors,
        atol=1e-14,
    )
    results = scph.run_many([0, 100], tolerance=1e-05, max_iterations=3)
    assert isinstance(results, list)
    assert [result.temperature for result in results] == [0.0, 100.0]
    assert all(result.converged for result in results)
    np.testing.assert_allclose(
        results[1].frequencies, results[0].frequencies, rtol=1e-10, atol=1e-05
    )
    for result in results:
        np.testing.assert_allclose(
            result.frequencies, Harmonic(result.fc2).frequencies(scph.stars), rtol=1e-12, atol=1e-12
        )


def test_imaginary_harmonic_input_is_accepted_and_reported() -> None:
    """Verify imaginary harmonic input is accepted and reported."""
    space = ClusterSpace(
        bulk("Ar", "sc", a=1.0),
        symprec=1e-05,
        cutoffs={2: 1.1, 4: 0.1},
        max_body_orders={2: 2, 4: 1},
    )
    coefficients = []
    for orbit in space.orbits[space.block(2).orbits]:
        tensor = (
            -6.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else 1.0
        ) * np.eye(3)
        coefficients.extend(
            np.linalg.lstsq(orbit.component_basis, tensor.reshape(-1), rcond=None)[0]
        )
    model = ForceConstants(
        space,
        {
            2: np.asarray(coefficients),
            4: np.zeros(space.block(4).parameters.stop - space.block(4).parameters.start),
        },
    )
    result = SCPH(model, np.diag([2, 2, 2])).run(300.0)
    assert result.converged
    assert result.has_imaginary_modes
    assert result.minimum_mode_thz < 0.0
