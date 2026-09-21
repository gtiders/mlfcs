"""The failure policy of the reciprocal paths: report, never average away.

The plan's section 11 lists the situations that must raise with locating information instead
of silently producing a plausible-looking result.  Most of them are pinned next to the code
that implements them (grid closure and partition completeness in ``test_reciprocal_stars.py``,
the gauge identity in ``test_reciprocal_covariance_path.py``); this module covers the ones
that are new here: the crystal-symmetry gate of every consumer, the mass consistency of a
site permutation, the Hermiticity of an expanded matrix, the sampler's expansion
self-consistency, and the message of an undetermined spglib symmetry.

Each test asserts that the *specific* failure is reported with the operation, the label or
the site it comes from, because a bare "symmetry error" would not tell a caller which force
constant to fix.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from test_reciprocal_full_grid_oracle import pair_bond_force_constants, scph_case

from mlfcs.exceptions import SymmetryViolationError
from mlfcs.force_constants.dense import lattice_fc2, replace_lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, dynamical_matrix, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.symmetry import (
    require_hermitian,
    require_little_group_covariance,
    validate_site_masses,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

CASES = ("diamond_2x1x1", "hcp_2x1x1", "cubic_2x1x1")
TEMPERATURE = 300.0


def _broken_case(name: str, multiplier: int = 2):
    """Return ``(terms, masses, symmetry, grid, scale)`` of a deliberately asymmetric model.

    The grid is refined by ``multiplier`` because the perturbation has to be *visible* at a
    representative: on a two-point grid the little-group relation of this particular model
    still holds after one bond is scaled, so a coarser grid would make the gate look broken
    instead of the model.
    """
    force_constants, _ = scph_case(name)
    primitive = force_constants.relation.primitive
    lattice = dict(lattice_fc2(force_constants))
    # Scale one off-site bond: the on-site blocks are site symmetric in every crystal, so
    # only an off-site interaction can break the space-group relation.
    key = max(
        (entry for entry in lattice if entry[2] != (0, 0, 0)),
        key=lambda entry: np.abs(lattice[entry]).sum(),
    )
    lattice[key] = lattice[key] * 1.3
    terms = fourier_terms(lattice, primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    grid = irreducible_reciprocal_grid(multiplier * force_constants.relation.supercell_matrix, symmetry)
    scale = max(
        float(np.max(np.abs(dynamical_matrix(terms, masses, grid.full.points[int(star.representative)]))))
        for star in grid.stars
    )
    return terms, masses, symmetry, grid, scale


@pytest.mark.parametrize("name", CASES)
def test_a_symmetric_model_passes_the_gate(name: str) -> None:
    """The gate must not fire on a model that satisfies its own crystal symmetry."""
    force_constants, _ = scph_case(name)
    mesh = harmonic_frequencies(force_constants, 1)
    assert mesh.symmetry_tolerance == 1e-6
    residual = require_little_group_covariance(
        partial(
            dynamical_matrices,
            fourier_terms(lattice_fc2(force_constants), force_constants.relation.primitive),
            np.asarray(force_constants.relation.primitive.get_masses(), dtype=float),
        ),
        np.asarray(force_constants.relation.primitive.get_masses(), dtype=float),
        PrimitiveSymmetryOperations.from_atoms(
            force_constants.relation.primitive, symprec=1e-5
        ),
        mesh.grid,
        tolerance=1e-6,
        context="test",
    )
    assert residual < 1e-6


def test_the_frequency_path_reports_a_broken_crystal_symmetry() -> None:
    """A force-constant set that breaks its symmetry is reported, with its location."""
    force_constants, _ = scph_case("hcp_2x1x1")
    lattice = dict(lattice_fc2(force_constants))
    key = max(
        (entry for entry in lattice if entry[2] != (0, 0, 0)),
        key=lambda entry: np.abs(lattice[entry]).sum(),
    )
    lattice[key] = lattice[key] * 1.3
    broken = type(force_constants)(
        dict(force_constants.metadata),
        force_constants.supercell,
        sparse={2: force_constants.sparse[2]},
        relation=force_constants.relation,
    )
    broken = replace_lattice_fc2(broken, lattice)

    with pytest.raises(SymmetryViolationError) as failure:
        harmonic_frequencies(broken, 2)
    message = str(failure.value)
    assert "operation" in message and "label" in message and "residual" in message
    assert "symmetry_tolerance" in message

    # The opt-out is explicit and local: the same model is accepted when the caller says so.
    mesh = harmonic_frequencies(broken, 2, symmetry_tolerance=None)
    assert mesh.symmetry_tolerance is None
    assert mesh.n_qpoints > 0


def test_the_little_group_gate_names_the_offending_operation_and_label() -> None:
    """The residual carries the operation index and the exact integer label."""
    terms, masses, symmetry, grid, scale = _broken_case("hcp_2x1x1")
    with pytest.raises(SymmetryViolationError) as failure:
        require_little_group_covariance(
            partial(dynamical_matrices, terms, masses),
            masses,
            symmetry,
            grid,
            tolerance=1e-6,
            context="diagnostics",
        )
    message = str(failure.value)
    assert "diagnostics" in message
    assert "operation" in message and "label" in message
    assert f"{scale:.6e}" in message or "scale" in message


def test_loop_scph_records_and_applies_the_symmetry_tolerance() -> None:
    """The tolerance is a public argument, recorded in the result and in the metadata."""
    force_constants, fc4 = scph_case("cubic_2x1x1")
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        mixing=1.0,
        max_iterations=1,
        symmetry_tolerance=1e-9,
    )
    result = solver._run_single(TEMPERATURE, None)
    assert result.symmetry_tolerance == 1e-9
    assert result.force_constants.metadata["symmetry_tolerance"] == 1e-9
    assert result.force_constants.metadata["symprec"] == solver.symprec

    broken_solver = LoopSCPH(
        fc2=force_constants,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        mixing=1.0,
        max_iterations=1,
        symmetry_tolerance=None,
    )
    assert broken_solver._run_single(TEMPERATURE, None).symmetry_tolerance is None

    with pytest.raises(ValueError):
        LoopSCPH(
            fc2=force_constants,
            fc4=fc4,
            temperature=TEMPERATURE,
            symmetry_tolerance=-1.0,
        )


def test_a_site_permutation_may_not_move_a_mass_between_species() -> None:
    """A permutation that maps a heavy site onto a light one is not a symmetry."""
    # Two sublattices with different masses, so a swap really is a different mass scale.
    from ase.build import bulk

    crystal = bulk("GaAs", "zincblende", a=5.653)
    symmetry = PrimitiveSymmetryOperations.from_atoms(crystal, symprec=1e-5)
    masses = np.asarray(crystal.get_masses(), dtype=float)
    assert masses[0] != masses[1]
    validate_site_masses(symmetry, masses, context="test")

    swapped = PrimitiveSymmetryOperations(
        rotations=symmetry.rotations,
        translations=symmetry.translations,
        cartesian_rotations=symmetry.cartesian_rotations,
        site_permutations=np.asarray([[1, 0]] * symmetry.size, dtype=np.int32),
        site_shifts=symmetry.site_shifts,
        symbol=symmetry.symbol,
    )
    with pytest.raises(SymmetryViolationError) as failure:
        validate_site_masses(swapped, masses, context="swapped")
    assert "swapped" in str(failure.value)
    assert "site" in str(failure.value) and "mass" in str(failure.value)


def test_require_hermitian_rejects_a_non_hermitian_expansion() -> None:
    """A non-Hermitian expanded matrix is an error, not something to symmetrize silently."""
    matrix = np.asarray([[2.0, 1.0j], [-1.0j, 2.0]])
    require_hermitian(
        matrix, scale=2.0, tolerance=1e-6, label=(0, 0, 0), operation=0, context="test"
    )
    broken = matrix + np.asarray([[0.0, 0.0], [0.5, 0.0]])
    with pytest.raises(SymmetryViolationError) as failure:
        require_hermitian(
            broken, scale=2.0, tolerance=1e-6, label=(0, 0, 1), operation=3, context="test"
        )
    message = str(failure.value)
    assert "operation 3" in message and "[0, 0, 1]" in message
    assert "Hermitian" in message


def test_the_sampler_expansion_is_checked_against_each_member_matrix(monkeypatch) -> None:
    """Dropping the space-group operator makes the sampler refuse to sample, loudly.

    This is the runtime counterpart of the covariance test: the sampler validates its own
    expansion on every member, so a regression in the rotation, the gauge or the
    antiunitary conjugation cannot silently produce plausible-looking displacements.  The
    case has a hexagonal cell, where the operations that reach a star member really mix the
    Cartesian axes (a monoatomic cubic cell would only see a sign, which the reconstruction
    cannot distinguish from the identity).
    """
    from ase.build import bulk
    from test_reciprocal_sampling_path import _sampler_of

    primitive = bulk("Mg", "hcp", a=3.2, c=5.2)
    matrix = np.diag((2, 2, 2)).astype(np.int64)
    force_constants, _ = pair_bond_force_constants(
        primitive, matrix, cutoff=3.3, spring=1.0, bend=0.5
    )
    _sampler_of(force_constants)  # the honest sampler must construct without complaining

    monkeypatch.setattr(
        "mlfcs.reciprocal.sampling.harmonic.star_member_operator",
        lambda symmetry, grid, member: (
            np.eye(3 * len(symmetry.site_permutations[0]), dtype=complex),
            False,
        ),
    )
    with pytest.raises(SymmetryViolationError) as failure:
        _sampler_of(force_constants)
    message = str(failure.value)
    assert "do not reproduce that member's dynamical matrix" in message
    assert "label" in message and "residual" in message


def test_the_sampler_records_both_tolerances_with_its_state() -> None:
    """The geometric and the physical tolerance travel with the state they produced."""
    from test_reciprocal_sampling_path import _sampler

    sampler = _sampler("hcp_2x1x1")
    state = sampler.state
    assert state.symprec == sampler.symprec == 1e-5
    assert state.symmetry_tolerance == sampler.symmetry_tolerance == 1e-6
    assert state.n_qpoints >= state.n_irreducible

    strict = _sampler("hcp_2x1x1", symmetry_tolerance=None)
    assert strict.state.symmetry_tolerance is None


def test_an_undetermined_spglib_symmetry_names_the_cell_and_tolerance(monkeypatch) -> None:
    """A structure spglib cannot classify is reported with the data that was passed."""
    from ase.build import bulk

    import mlfcs.structure.symmetry as module

    primitive = bulk("Si", "diamond", a=5.43)
    monkeypatch.setattr(module.spglib, "get_symmetry_dataset", lambda *args, **kwargs: None)
    with pytest.raises(ValueError) as failure:
        PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-4)
    message = str(failure.value)
    assert "2 atoms" in message
    assert "symprec 0.0001" in message
    assert "cell" in message


def test_pair_bond_models_are_accepted_by_the_gate() -> None:
    """The gate accepts a model that is symmetric by construction, including GaAs."""
    from ase.build import bulk

    primitive = bulk("GaAs", "zincblende", a=5.653)
    matrix = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    fc2, _ = pair_bond_force_constants(primitive, matrix, cutoff=3.2, spring=1.0, bend=0.5)
    mesh = harmonic_frequencies(fc2, 1)
    assert mesh.n_irreducible < mesh.n_qpoints
