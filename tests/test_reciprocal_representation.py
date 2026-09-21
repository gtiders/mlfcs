"""One space-group representation for the mass-weighted displacement space.

The dynamical matrix is not merely labelled by a q point: it transforms covariantly under
every operation that keeps the reciprocal grid.  These tests pin that relation where it is
observable, because the symmetry reduction, the covariance expansion and any symmetry-aware
sampling all build on it:

* ``D(gq) = U_g(q) D(q) U_g(q)^dagger`` for every grid-preserving operation and every q
  label, including on a skewed cell and a non-diagonal supercell, where a transposed
  convention fails;
* time reversal acts as an antiunitary member, ``D(-q) = conj(D(q))``;
* ``U_g(q)`` is unitary and moves atom ``a`` onto ``site_permutations[g][a]``;
* degenerate subspaces are compared through projectors, never through eigenvectors, since
  eigenvectors of a degenerate eigenvalue have no canonical direction;
* a force-constant set that breaks the symmetry of its structure is detected *and located*
  instead of being averaged away, and the diagnostics leave it untouched.

The force constants come from a finite-difference run of a symmetric potential, so they
respect the crystal symmetry to contraction roundoff without being built by hand.
The gauge itself is pinned separately in ``test_reciprocal_gauge.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from reciprocal_helpers import CASES, SUPERCELLS, crystal_case

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrix, fourier_terms
from mlfcs.reciprocal.grid import (
    reciprocal_grid_symmetry,
    reciprocal_quotient_grid,
    rotate_label,
)
from mlfcs.reciprocal.symmetry import (
    conjugate_matrix,
    covariance_residuals,
    displacement_representation,
    maximum_covariance_residual,
)


def _degenerate_clusters(values: np.ndarray, tolerance: float = 1e-8) -> list[np.ndarray]:
    """Return index arrays of eigenvalues that are equal within ``tolerance``."""
    clusters: list[np.ndarray] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or abs(values[index] - values[start]) > tolerance:
            clusters.append(np.arange(start, index))
            start = index
    return clusters


@pytest.mark.parametrize("case_name", CASES)
def test_dynamical_matrix_is_covariant_under_the_grid_preserving_group(case_name):
    """``D(gq) = U_g(q) D(q) U_g(q)^dagger`` with an exactly formed phase."""
    _, force_constants, primitive, symmetry = crystal_case(case_name)
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])
    compatible = reciprocal_grid_symmetry(SUPERCELLS["diagonal"], symmetry)

    residual = maximum_covariance_residual(
        terms,
        masses,
        symmetry,
        grid.labels,
        grid.denominator,
        operations=compatible.operation_indices,
    )
    assert residual < 1e-8, residual

    # The measurement is complete: every grid-preserving operation appears, and each of
    # them keeps the grid exactly (checked through the integer label action).
    measured = {
        (entry.operation, entry.label)
        for entry in covariance_residuals(
            terms,
            masses,
            symmetry,
            grid.labels,
            grid.denominator,
            operations=compatible.operation_indices,
        )
    }
    expected = {
        (int(operation), tuple(int(value) for value in label))
        for operation in compatible.operation_indices
        for label in grid.labels
    }
    assert measured == expected


@pytest.mark.parametrize("case_name", CASES)
def test_time_reversal_is_antiunitary(case_name):
    """``D(-q) = conj(D(q))``, stated as an antiunitary member rather than a rotation."""
    _, force_constants, primitive, _ = crystal_case(case_name)
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])

    # The negation is taken on the unreduced label: `grid.negative_label` reduces modulo
    # the grid, and that reduction is a primitive reciprocal lattice translation, i.e. a
    # site-diagonal gauge factor in the positional gauge of the dynamical matrix.
    for label in grid.labels:
        direct = dynamical_matrix(terms, masses, -np.asarray(label, dtype=float) / grid.denominator)
        np.testing.assert_allclose(
            direct,
            conjugate_matrix(dynamical_matrix(terms, masses, label / grid.denominator)),
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("case_name", CASES)
def test_representation_is_unitary_and_permutes_atoms(case_name):
    """``U_g(q)`` is unitary and moves atom ``a`` onto ``site_permutations[g][a]``."""
    _, _, _, symmetry = crystal_case(case_name)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])
    label = grid.labels[-1]
    for operation in range(symmetry.size):
        unitary = displacement_representation(symmetry, operation, label, grid.denominator)
        np.testing.assert_allclose(unitary @ unitary.conj().T, np.eye(len(unitary)), atol=1e-12)
        occupancy = np.abs(unitary) > 1e-12
        n = len(symmetry.site_permutations[operation])
        blocks = occupancy.reshape(n, 3, n, 3).any(axis=(1, 3))
        for site in range(n):
            assert int(np.argmax(blocks[site])) == int(symmetry.site_permutations[operation, site])


@pytest.mark.parametrize("case_name", CASES)
def test_degenerate_subspaces_map_covariantly(case_name):
    """Degenerate clusters are compared through projectors, never through eigenvectors."""
    _, force_constants, primitive, symmetry = crystal_case(case_name)
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])
    compatible = reciprocal_grid_symmetry(SUPERCELLS["diagonal"], symmetry)

    for operation in compatible.operation_indices:
        rotation = np.asarray(symmetry.rotations[operation], dtype=np.int64)
        for label in grid.labels:
            partner = np.asarray(
                rotate_label(label, rotation, grid.denominator, reduce=False), dtype=float
            )
            values, vectors = np.linalg.eigh(
                dynamical_matrix(terms, masses, label / grid.denominator)
            )
            partner_values, partner_vectors = np.linalg.eigh(
                dynamical_matrix(terms, masses, partner / grid.denominator)
            )
            np.testing.assert_allclose(values, partner_values, rtol=1e-9, atol=1e-11)
            unitary = displacement_representation(symmetry, int(operation), label, grid.denominator)
            for cluster in _degenerate_clusters(values):
                projector = vectors[:, cluster] @ vectors[:, cluster].conj().T
                mapped = unitary @ projector @ unitary.conj().T
                other = partner_vectors[:, cluster] @ partner_vectors[:, cluster].conj().T
                np.testing.assert_allclose(mapped, other, rtol=1e-8, atol=1e-10)


def _degenerate_clusters(values: np.ndarray, tolerance: float = 1e-8) -> list[np.ndarray]:
    """Return index arrays of eigenvalues that are equal within ``tolerance``."""
    clusters: list[np.ndarray] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or abs(values[index] - values[start]) > tolerance:
            clusters.append(np.arange(start, index))
            start = index
    return clusters


@pytest.mark.parametrize("case_name", CASES)
def test_broken_symmetry_is_detected_and_located(case_name):
    """A force-constant set that breaks its crystal symmetry is reported, not averaged."""
    _, force_constants, primitive, symmetry = crystal_case(case_name)
    lattice = dict(lattice_fc2(force_constants))
    # Scaling the on-site block would keep the model symmetric: it is site-symmetric in
    # every crystal, so only an off-site bond can break the space-group relation.
    off_site = [key for key in lattice if key[2] != (0, 0, 0) or key[0] != key[1]]
    assert off_site, "the fixture has no off-site interaction to break"
    key = max(off_site, key=lambda item: np.abs(lattice[item]).sum())
    broken = {name: value.copy() for name, value in lattice.items()}
    broken[key] = broken[key] * 1.3
    terms = fourier_terms(broken, primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])
    compatible = reciprocal_grid_symmetry(SUPERCELLS["diagonal"], symmetry)

    residuals = covariance_residuals(
        terms,
        masses,
        symmetry,
        grid.labels,
        grid.denominator,
        operations=compatible.operation_indices,
    )
    worst = max(residuals, key=lambda entry: entry.residual)
    assert worst.residual > 1e-3
    assert worst.operation in {int(value) for value in compatible.operation_indices}
    assert len(worst.label) == 3
    assert max(entry.residual for entry in residuals) >= worst.residual


@pytest.mark.parametrize("case_name", CASES)
def test_symmetry_broken_force_constants_are_only_reported_by_the_diagnostics(case_name):
    """The diagnostics never modify the force constants they measure."""
    _, force_constants, primitive, symmetry = crystal_case(case_name)
    before = {name: value.copy() for name, value in lattice_fc2(force_constants).items()}
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    grid = reciprocal_quotient_grid(SUPERCELLS["diagonal"])

    maximum_covariance_residual(terms, masses, symmetry, grid.labels, grid.denominator)

    after = lattice_fc2(force_constants)
    assert set(before) == set(after)
    for name, tensor in before.items():
        np.testing.assert_array_equal(after[name], tensor)
