"""Second-phase contract: the Gamma null space, the mode policy and the finite gates.

The first phase proved that a star expansion is covariant.  These tests are the contract for what
it still does *not* prove:

* the three mass-weighted Gamma translations must leave the modal space structurally, not through a
  frequency cutoff that happens to exclude them;
* a soft optical mode must never be mistaken for one of those translations;
* the covariance needs its own certificate, because ``1 / lambda`` amplifies the error a covariant
  matrix is allowed to carry;
* a non-covariant loop correction must be located as an ASR violation, not as a generic star
  failure;
* non-finite data must not slip past a gate whose tolerance is switched off.

They are red on the current baseline on purpose; each one names the behaviour that is still wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from test_reciprocal_full_grid_oracle import pair_bond_force_constants, scph_case

from mlfcs.exceptions import SymmetryViolationError
from mlfcs.force_constants.dense import lattice_fc2, replace_lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

TEMPERATURE = 300.0
SYMPREC = 1e-5


def _hcp_model(multiplier: int = 1):
    """Return the pinned hexagonal model, its grid and the batched dynamical-matrix builder."""
    force_constants, _ = scph_case("hcp_2x1x1")
    relation = force_constants.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    symmetry = PrimitiveSymmetryOperations.from_atoms(relation.primitive, symprec=SYMPREC)
    grid = irreducible_reciprocal_grid(multiplier * relation.supercell_matrix, symmetry)
    terms = fourier_terms(lattice_fc2(force_constants), relation.primitive)
    return force_constants, relation, masses, symmetry, grid, terms


def test_the_default_cutoff_keeps_the_gamma_translations_out_of_the_covariance() -> None:
    """With the default cutoff the Gamma covariance must not be dominated by translation noise.

    Today the three numerically nonzero translations enter ``kT / omega^2`` and the on-site blocks
    reach 1e13, which is the rounding noise of the acoustic sum rule rather than physics.
    """
    force_constants, _, _, _, _, _ = _hcp_model()
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=scph_case("hcp_2x1x1")[1],
        temperature=TEMPERATURE,
        statistics="classical",
        interpolation_multiplier=1,
        scph_multiplier=1,
        mixing=1.0,
        max_iterations=1,
    )
    covariance = solver._covariance(lattice_fc2(force_constants), 1, TEMPERATURE)
    largest = max(float(np.max(np.abs(block))) for block in covariance.values())
    assert np.isfinite(largest)
    assert largest < 1e3, (
        f"the Gamma covariance is dominated by the translation zero modes: largest block "
        f"{largest:.3e}"
    )


def test_the_three_excluded_modes_are_the_mass_weighted_translations() -> None:
    """A soft optical mode must not be selected as one of the three translations.

    The current sampler picks the three eigenvectors with the largest overlap with the translation
    basis, which is gauge dependent exactly when a soft optical mode sits near them.
    """
    primitive = bulk("Mg", "hcp", a=3.2, c=5.2)
    matrix = np.diag((2, 1, 1)).astype(np.int64)
    force_constants, _ = pair_bond_force_constants(
        primitive, matrix, cutoff=3.3, spring=1.0, bend=0.5
    )
    from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler

    sampler = HarmonicSampler(
        primitive,
        force_constants.relation.reference,
        force_constants.materialize(2),
        temperature=TEMPERATURE,
        statistics="classical",
        imaginary_modes="absolute",
    )
    translations = sampler._translation_basis()
    # The projector onto the translations must be exactly the one built from the masses.
    projector = translations @ translations.T
    expected = np.zeros_like(projector)
    for axis in range(3):
        direction = np.zeros(3 * len(primitive))
        for site, mass in enumerate(np.asarray(primitive.get_masses(), dtype=float)):
            direction[3 * site + axis] = np.sqrt(mass)
        direction /= np.linalg.norm(direction)
        expected += np.outer(direction, direction)
    np.testing.assert_allclose(projector, expected, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(translations))


def test_a_non_asr_force_constant_set_fails_the_gamma_acoustic_certificate() -> None:
    """``||D(Gamma) B||`` is the certificate the plan asks for, and it must be enforced."""
    from mlfcs.reciprocal.modes import gamma_acoustic_residual

    force_constants, relation, masses, _, _, terms = _hcp_model()
    lattice = dict(lattice_fc2(force_constants))
    key = max(lattice, key=lambda entry: np.abs(lattice[entry]).sum())
    lattice[key] = lattice[key] * 1.3
    broken = replace_lattice_fc2(force_constants, lattice)
    terms_broken = fourier_terms(lattice_fc2(broken), relation.primitive)

    gamma_matrix = dynamical_matrices(terms_broken, masses, np.zeros((1, 3)))[0]
    residual = gamma_acoustic_residual(gamma_matrix, masses)
    scale = float(np.max(np.abs(dynamical_matrices(terms, masses, np.zeros((1, 3))))))
    assert residual > 1e-6 * scale, f"the broken ASR was not detected: residual {residual:.3e}"

    solver = LoopSCPH(
        fc2=broken,
        fc4=scph_case("hcp_2x1x1")[1],
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        mixing=1.0,
        max_iterations=1,
        symmetry_tolerance=None,
    )
    with pytest.raises(SymmetryViolationError) as failure:
        solver._run_single(TEMPERATURE, None)
    assert "ASR" in str(failure.value)


def test_a_non_covariant_loop_correction_is_reported_as_an_asr_violation() -> None:
    """The correction must be located as an ASR problem, not as a generic star failure."""
    from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants

    force_constants, fc4 = scph_case("diamond_2x1x1")
    sparse = fc4.sparse[4]
    tensors = np.array(sparse.tensors, dtype=float)
    index = int(np.argmax(np.abs(tensors).sum(axis=(1, 2, 3, 4))))
    tensors[index] *= 1.3
    broken_fc4 = ForceConstants(
        {},
        fc4.supercell,
        dict(fc4.metadata),
        {4: SparseOrderForceConstants(4, sparse.sites, sparse.translations, tensors)},
        fc4.relation,
    )
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=broken_fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=2,
        mixing=1.0,
        max_iterations=1,
        frequency_cutoff_thz=1.0,
    )
    with pytest.raises(SymmetryViolationError) as failure:
        solver._run_single(TEMPERATURE, None)
    message = str(failure.value)
    assert "loop correction" in message and "ASR" in message
    assert "iteration" in message


def test_the_covariance_gate_rejects_a_model_the_matrix_gate_accepts() -> None:
    """A covariant matrix is not a covariant covariance: ``1 / lambda`` amplifies the residual."""
    from mlfcs.reciprocal.modes import require_star_covariance_matrix

    _, relation, masses, symmetry, grid, terms = _hcp_model()
    from functools import partial

    from mlfcs.reciprocal.symmetry import require_star_covariance

    positions = np.asarray(relation.primitive.get_scaled_positions(wrap=False), dtype=float)
    build = partial(dynamical_matrices, terms, masses)
    # The matrix gate is the one that exists today and it accepts the model.
    require_star_covariance(build, symmetry, grid, positions, tolerance=1e-6, context="matrix")
    # The covariance certificate is the stronger statement the plan asks for.
    with pytest.raises(SymmetryViolationError):
        require_star_covariance_matrix(
            build,
            masses,
            symmetry,
            grid,
            positions,
            tolerance=1e-6,
            temperature=TEMPERATURE,
            statistics="classical",
            frequency_cutoff_thz=1e-3,
            context="covariance",
        )


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_nonfinite_data_is_rejected_even_with_the_gate_switched_off(value: float) -> None:
    """``symmetry_tolerance=None`` may skip a comparison; it may not admit NaN or Inf."""
    from functools import partial

    from mlfcs.reciprocal.symmetry import require_hermitian, require_star_covariance

    force_constants, relation, masses, symmetry, grid, _ = _hcp_model()
    positions = np.asarray(relation.primitive.get_scaled_positions(wrap=False), dtype=float)
    lattice = dict(lattice_fc2(force_constants))
    key = max(lattice, key=lambda entry: np.abs(lattice[entry]).sum())
    lattice[key] = np.asarray(lattice[key], dtype=float).copy()
    lattice[key][0, 0] = value
    terms_broken = fourier_terms(lattice, relation.primitive)
    build = partial(dynamical_matrices, terms_broken, masses)

    with pytest.raises((SymmetryViolationError, ValueError)) as failure:
        require_star_covariance(
            build, symmetry, grid, positions, tolerance=None, context="nonfinite"
        )
    assert "finite" in str(failure.value).lower() or "nan" in str(failure.value).lower()

    with pytest.raises((SymmetryViolationError, ValueError)):
        require_hermitian(
            np.asarray([[value, 0.0], [0.0, 1.0]]),
            scale=1.0,
            tolerance=None,
            label=(0, 0, 0),
            operation=0,
            context="nonfinite",
        )


def test_the_tutorial_script_no_longer_passes_the_removed_worker_argument() -> None:
    """The K4As4Pt2 script has to run on the current HEAD, which has no qpoint_workers."""
    source = (__import__("pathlib").Path("tutorial/scph/K4As4Pt2/run.py")).read_text()
    assert "qpoint_workers" not in source
