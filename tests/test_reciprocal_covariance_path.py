"""The SCPH covariance is built on representatives and expanded over each star.

The reduction is only legitimate if it keeps the *full* physical sum: a star member is
related to its representative by a rotation and a complex conjugation, never by a scalar
weight, so the covariance is expanded matrix by matrix and then summed over every full q
point.  These tests pin that against a direct full-grid computation of the same quantity,
and they pin the cost claim as well:

* every required ``(a, b, R)`` block equals the direct full-grid value;
* the eigensolver is entered once per sweep, for the representatives only;
* replacing the star sum by a weight times the representative matrix is *detected*;
* the antiunitary members of a star are actually exercised;
* the worker count does not change the result.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from test_reciprocal_full_grid_oracle import pair_bond_force_constants, scph_case

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, dynamical_matrix, fourier_terms
from mlfcs.reciprocal.scph.fourier import _needed_covariances
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.statistics import mode_sigma
from mlfcs.reciprocal.symmetry import star_member_gauge, star_member_operator

CASES = ("diamond_2x1x1", "diamond_nondiagonal", "hcp_2x1x1", "cubic_2x1x1", "rhombohedral_2x1x1")
TEMPERATURE = 300.0


def _assert_block_close(star, direct, key) -> None:
    """Compare one covariance block with a tolerance set by its own physical scale.

    A cell whose harmonic spectrum has near-zero acoustic modes produces on-site blocks of
    order ``1e13`` while the interesting inter-site elements are of order ``1e-4``: those
    elements are the residue of a cancellation between large numbers, so the achievable
    agreement is set by the relative accuracy of the diagonalization (about ``1e-16``) times
    the block scale, not by the size of the element.  A single absolute tolerance would
    either pass everything or fail the honest implementation, so the tolerance is scaled by
    the block.
    """
    scale = float(np.max(np.abs(direct[key])))
    np.testing.assert_allclose(star[key], direct[key], rtol=1e-9, atol=1e-13 * scale)
    np.testing.assert_allclose(star[key].imag, 0.0, rtol=0.0, atol=1e-13 * scale)


def _solver(name: str, *, multiplier: int = 1, **options) -> LoopSCPH:
    fc2, fc4 = scph_case(name)
    return LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=multiplier,
        statistics="classical",
        mixing=1.0,
        max_iterations=1,
        **options,
    )


def _direct_full_grid_covariance(solver: LoopSCPH, lattice, multiplier: int):
    """Return the covariance of a direct full-grid sweep, as an independent oracle."""
    relation = solver.fc2.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    positions = relation.primitive.get_scaled_positions(wrap=False)
    terms = fourier_terms(lattice, relation.primitive)
    qpoints = solver._qpoints(multiplier)
    n_q = len(qpoints)
    eigenvalues, vectors = np.linalg.eigh(dynamical_matrices(terms, masses, qpoints))
    sigma2 = (
        mode_sigma(
            eigenvalues,
            temperature=TEMPERATURE,
            statistics=solver.statistics,
            cutoff_frequency_thz=solver.frequency_cutoff_thz,
        )
        ** 2
    )
    weighted = (vectors * sigma2[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
    result = {}
    for a, b, r in sorted(_needed_covariances(solver.fc4.sparse[4])):
        block = weighted[:, 3 * a : 3 * a + 3, 3 * b : 3 * b + 3] / np.sqrt(masses[a] * masses[b])
        displacement = positions[a] - positions[b] + np.asarray(r)
        phase = np.exp(2j * np.pi * (qpoints @ displacement))
        result[(a, b, r)] = np.sum(block * phase[:, None, None], axis=0) / n_q
    return result


@pytest.mark.parametrize("name", CASES)
def test_every_required_block_matches_the_direct_full_grid_sum(name: str) -> None:
    """Each covariance block the quartic terms need equals the full-grid computation."""
    solver = _solver(name)
    lattice = lattice_fc2(solver.fc2)
    star = solver._covariance(lattice, 1, TEMPERATURE)
    direct = _direct_full_grid_covariance(solver, lattice, 1)

    assert set(star) == set(direct)
    assert set(star) == _needed_covariances(solver.fc4.sparse[4])
    for key in sorted(direct):
        _assert_block_close(star, direct, key)


@pytest.mark.parametrize("name", CASES)
def test_covariance_diagonalizes_only_representatives(name: str, monkeypatch) -> None:
    """One sweep enters the eigensolver once, on a batch of representatives."""
    solver = _solver(name)
    mesh = solver._mesh(1)
    calls: list[tuple[int, ...]] = []
    original = np.linalg.eigh

    def counting(values, *args, **kwargs):
        calls.append(np.asarray(values).shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", counting)
    solver._covariance(lattice_fc2(solver.fc2), 1, TEMPERATURE)

    n_modes = 3 * len(solver.fc2.relation.primitive)
    assert calls == [(mesh.n_irreducible, n_modes, n_modes)]
    assert mesh.n_irreducible <= mesh.n_qpoints
    assert int(mesh.weights.sum()) == mesh.n_qpoints


@pytest.mark.parametrize("name", ("diamond_2x1x1", "diamond_nondiagonal"))
def test_a_weighted_representative_is_not_the_star_sum(name: str) -> None:
    """Scaling a representative by its star weight is wrong and is detected here."""
    solver = _solver(name, multiplier=2)
    lattice = lattice_fc2(solver.fc2)
    mesh = solver._mesh(2)
    assert int(mesh.weights.max()) > 1, "this case must have a star with more than one member"

    relation = solver.fc2.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    positions = relation.primitive.get_scaled_positions(wrap=False)
    terms = fourier_terms(lattice, relation.primitive)
    points = mesh.full.points[mesh.representatives]
    eigenvalues, vectors = np.linalg.eigh(dynamical_matrices(terms, masses, points))
    sigma2 = (
        mode_sigma(
            eigenvalues,
            temperature=TEMPERATURE,
            statistics=solver.statistics,
            cutoff_frequency_thz=solver.frequency_cutoff_thz,
        )
        ** 2
    )
    weighted = (vectors * sigma2[..., None, :]) @ vectors.conj().swapaxes(-1, -2)

    star = solver._covariance(lattice, 2, TEMPERATURE)
    for a, b, r in sorted(_needed_covariances(solver.fc4.sparse[4])):
        blocks = weighted[:, 3 * a : 3 * a + 3, 3 * b : 3 * b + 3] / np.sqrt(masses[a] * masses[b])
        displacement = positions[a] - positions[b] + np.asarray(r)
        phases = np.exp(2j * np.pi * (points @ displacement))
        naive = np.sum(
            (mesh.weights / mesh.n_qpoints)[:, None, None] * blocks * phases[:, None, None], axis=0
        )
        difference = float(np.max(np.abs(star[(a, b, r)] - naive)))
        assert difference > 1e-12, f"block {(a, b, r)} cannot distinguish the star sum"


@pytest.mark.parametrize("case", ("diamond_2x1x1", "hcp_2x1x1"))
def test_expansion_with_the_gauge_reproduces_the_member_matrix(case: str) -> None:
    """``Gamma_g U_g W(q_s) U_g^dagger Gamma_g^dagger`` is ``W`` at the member's own label.

    The operation carries the representative onto its *unreduced* image, which differs from
    the stored label by a primitive reciprocal lattice translation.  In the positional gauge
    that translation is a site-diagonal factor, and without it the expansion is wrong by
    order one on exactly the members whose label reduction bites.  This is the identity the
    covariance sum depends on, so it is pinned on the matrices themselves.
    """
    force_constants, _ = scph_case(case)
    solver = _solver(case)
    mesh = solver._mesh(1)
    relation = force_constants.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    positions = relation.primitive.get_scaled_positions(wrap=False)
    terms = fourier_terms(lattice_fc2(force_constants), relation.primitive)

    checked = 0
    for star in mesh.stars:
        representative = int(star.representative)
        source_matrix = dynamical_matrix(
            terms, masses, mesh.full.points[representative]
        )
        for position, member in enumerate(star.members.tolist()):
            unitary, antiunitary = star_member_operator(solver._symmetry, mesh, member)
            gauge = star_member_gauge(solver._symmetry, mesh, member, positions)
            source = np.conjugate(source_matrix) if antiunitary else source_matrix
            expanded = unitary @ source @ unitary.conj().T
            expanded = gauge[:, None] * expanded * gauge.conj()[None, :]
            direct = dynamical_matrix(terms, masses, mesh.full.points[member])
            np.testing.assert_allclose(expanded, direct, rtol=1e-9, atol=1e-12)
            checked += 1
    assert checked == mesh.n_qpoints


def _non_centrosymmetric_solver() -> LoopSCPH:
    """Return an SCPH solver of zincblende GaAs on a grid that needs time reversal.

    Every pinned crystal of the full-grid oracle is centrosymmetric, where inversion already
    maps ``q`` onto ``-q`` and the antiunitary branch is never taken.  Zincblende has no
    inversion centre, and this supercell has stars whose members are reachable only through
    time reversal, so the conjugation in the expansion is exercised for real.
    """
    primitive = bulk("GaAs", "zincblende", a=5.653)
    matrix = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    fc2, fc4 = pair_bond_force_constants(primitive, matrix, cutoff=3.2, spring=1.0, bend=0.5)
    return LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        statistics="classical",
        mixing=1.0,
        max_iterations=1,
    )


def test_antiunitary_members_reproduce_the_full_grid_covariance() -> None:
    """A star whose members are reached through time reversal keeps the physical sum."""
    solver = _non_centrosymmetric_solver()
    mesh = solver._mesh(1)
    assert int(np.count_nonzero(mesh.full_antiunitary)) > 0
    assert int(mesh.weights.sum()) == mesh.n_qpoints
    assert mesh.n_irreducible < mesh.n_qpoints

    lattice = lattice_fc2(solver.fc2)
    star = solver._covariance(lattice, 1, TEMPERATURE)
    direct = _direct_full_grid_covariance(solver, lattice, 1)
    assert set(star) == set(direct)
    for key in sorted(direct):
        _assert_block_close(star, direct, key)
