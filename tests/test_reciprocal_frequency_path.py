"""The frequency path computes and diagonalizes only irreducible representative points.

The reduction is a bookkeeping change: a star carries the eigenvalues of its
representative, so expanding the irreducible arrays reproduces the full mesh exactly and
no eigensolver is entered for a star member.  These tests pin both halves of that claim,
because a reduction that silently changes the frequencies, or an expansion that
re-diagonalizes the full mesh, would look identical from the outside:

* the irreducible wedge and the star weights agree with the grid decomposition;
* the expansion reproduces a direct full-grid diagonalization;
* the eigensolver is entered exactly ``N_irr`` times per mesh, and never during expansion;
* the stopping metric is the star-weighted full-grid RMS;
* turning time reversal off changes the wedge but not the physics.

The force constants come from the pinned full-grid oracle, so the numbers are the ones
stage A recorded before any symmetry reduction existed.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_reciprocal_full_grid_oracle import scph_case

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.statistics import OMEGA_TO_THZ
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

CASES = ("cubic_2x1x1", "hcp_2x1x1", "diamond_2x1x1", "diamond_nondiagonal")
TEMPERATURE = 300.0


def _direct_full_grid_frequencies(fc2, multiplier: int) -> np.ndarray:
    """Return the frequencies of the full grid from an independent direct computation."""
    relation = fc2.relation
    terms = fourier_terms(lattice_fc2(fc2), relation.primitive)
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    grid = irreducible_reciprocal_grid(multiplier * relation.supercell_matrix, _symmetry(fc2))
    values = np.linalg.eigvalsh(dynamical_matrices(terms, masses, grid.full.points))
    return np.sqrt(np.abs(values)) * np.sign(values) * OMEGA_TO_THZ


def _symmetry(fc2) -> PrimitiveSymmetryOperations:
    return PrimitiveSymmetryOperations.from_atoms(fc2.relation.primitive, symprec=1e-5)


@pytest.mark.parametrize("name", CASES)
def test_mesh_is_the_irreducible_wedge_of_its_grid(name: str) -> None:
    """The mesh reports the star decomposition it was built from, exactly."""
    force_constants, _ = scph_case(name)
    mesh = harmonic_frequencies(force_constants, 1)
    grid = irreducible_reciprocal_grid(
        force_constants.relation.supercell_matrix, _symmetry(force_constants)
    )

    np.testing.assert_array_equal(mesh.weights, grid.weights)
    assert mesh.n_irreducible == len(grid.representatives)
    assert mesh.n_qpoints == len(grid.full.labels)
    assert int(mesh.weights.sum()) == mesh.n_qpoints
    assert mesh.reduction_ratio == mesh.n_qpoints / mesh.n_irreducible

    # Representatives are the smallest full-grid index of their star, so the reported q
    # points are the ones the decomposition calls irreducible, in ascending order.
    np.testing.assert_array_equal(mesh.irreducible_qpoints, grid.full.points[grid.representatives])
    np.testing.assert_array_equal(mesh.full_qpoints(), grid.full.points)
    np.testing.assert_array_equal(mesh.irreducible_frequencies.shape, (mesh.n_irreducible, 3 * len(
        force_constants.relation.primitive
    )))


@pytest.mark.parametrize("name", CASES)
def test_expansion_reproduces_the_full_grid_diagonalization(name: str) -> None:
    """The expanded mesh equals a direct full-grid diagonalization."""
    force_constants, _ = scph_case(name)
    mesh = harmonic_frequencies(force_constants, 1)
    np.testing.assert_allclose(
        mesh.expand_frequencies(),
        _direct_full_grid_frequencies(force_constants, 1),
        rtol=1e-10,
        atol=1e-9,
    )


@pytest.mark.parametrize("name", CASES)
def test_only_representatives_reach_the_eigensolver(name: str, monkeypatch) -> None:
    """A mesh costs ``N_irr`` diagonalizations and the expansion costs none."""
    force_constants, _ = scph_case(name)
    calls = {"count": 0}
    original = np.linalg.eigvalsh

    def counting(values, *args, **kwargs):
        calls["count"] += 1
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", counting)
    mesh = harmonic_frequencies(force_constants, 1)
    assert calls["count"] == 1, "the representative batch must be diagonalized in one call"

    batch = np.linalg.eigvalsh(
        np.zeros((mesh.n_irreducible, 3, 3)) + np.eye(3)[None, :, :]
    )
    assert batch.shape == (mesh.n_irreducible, 3)

    before = calls["count"]
    expanded = mesh.expand_frequencies()
    assert calls["count"] == before, "expansion must not re-enter the eigensolver"
    assert expanded.shape == (mesh.n_qpoints, mesh.irreducible_frequencies.shape[1])


@pytest.mark.parametrize("name", ("cubic_2x1x1", "cubic_3x1x1"))
def test_scph_frequency_path_diagonalizes_only_representatives(name: str, monkeypatch) -> None:
    """One SCPH run enters the frequency eigensolver once per representative per sweep."""
    force_constants, fc4 = scph_case(name)
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=2,
        mixing=0.5,
        max_iterations=3,
        # The classical covariance of a translation-invariant model diverges at the acoustic
        # zero modes, so a cutoff keeps the iterate well posed; the default of zero would make
        # the update non-covariant through roundoff rather than through the physics here.
        frequency_cutoff_thz=1.0,
    )
    calls = {"count": 0}
    original = np.linalg.eigvalsh

    def counting(values, *args, **kwargs):
        calls["count"] += 1
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", counting)
    result = solver._run_single(TEMPERATURE, None)

    sweeps = len(result.history) + 1  # the initial frequencies plus one per iteration
    assert calls["count"] == sweeps
    assert int(result.weights.sum()) == result.n_qpoints
    np.testing.assert_array_equal(
        result.expand_frequencies().shape,
        (result.n_qpoints, result.irreducible_frequencies.shape[1]),
    )


@pytest.mark.parametrize("name", CASES)
def test_stopping_metric_is_the_star_weighted_full_grid_rms(name: str) -> None:
    """The weighted representative sum equals the RMS of the expanded change."""
    force_constants, _ = scph_case(name)
    mesh = harmonic_frequencies(force_constants, 1)
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=scph_case(name)[1],
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        mixing=1.0,
        max_iterations=1,
    )
    result = solver._run_single(TEMPERATURE, None)

    delta = result.irreducible_frequencies - mesh.irreducible_frequencies
    weights = np.asarray(result.weights, dtype=float)
    weighted = np.sqrt(
        np.sum(weights[:, None] * delta**2) / (result.n_qpoints * delta.shape[1])
    )
    assert result.history[0].frequency_change_thz == pytest.approx(weighted, rel=1e-12)
    expanded_change = result.expand_frequencies() - mesh.expand_frequencies()
    assert weighted == pytest.approx(float(np.sqrt(np.mean(expanded_change**2))), rel=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_time_reversal_switch_changes_the_wedge_and_not_the_physics(name: str) -> None:
    """Turning time reversal off is exact, and the expanded mesh is unchanged."""
    force_constants, _ = scph_case(name)
    with_reversal = harmonic_frequencies(force_constants, 1)
    without = harmonic_frequencies(force_constants, 1, time_reversal=False)

    assert without.time_reversal is False
    assert without.n_irreducible >= with_reversal.n_irreducible
    assert int(without.weights.sum()) == without.n_qpoints
    np.testing.assert_allclose(
        without.expand_frequencies(), with_reversal.expand_frequencies(), rtol=0.0, atol=1e-12
    )
