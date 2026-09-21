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

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from ase import Atoms
from test_reciprocal_full_grid_oracle import pair_bond_force_constants, scph_case

from mlfcs import build_supercell
from mlfcs.exceptions import SymmetryViolationError
from mlfcs.force_constants.dense import lattice_fc2, replace_lattice_fc2
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.reciprocal.fourier import dynamical_matrices, dynamical_matrix, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid, reciprocal_quotient_grid
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.symmetry import (
    require_hermitian,
    require_little_group_covariance,
    validate_site_masses,
)
from mlfcs.structure.relation import StructureRelation
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

    import mlfcs.reciprocal.sampling.harmonic as harmonic_module

    honest = harmonic_module.star_member_action

    def without_rotation(symmetry, grid, member, positions):
        action = honest(symmetry, grid, member, positions)
        return replace(action, unitary=np.eye(len(action.unitary), dtype=complex))

    monkeypatch.setattr(harmonic_module, "star_member_action", without_rotation)
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


# --------------------------------------------------------------------------------------
# P0 counterexamples: the little group is not a sufficient star-covariance certificate
# --------------------------------------------------------------------------------------


def _scalar_spectrum_force_constants(
    overrides: dict[tuple[int, int, int], float] | None = None,
):
    """Return a real single-atom cubic FC2 whose grid spectrum is ``d(q) * I``.

    The spectrum is chosen on the full ``3x3x3`` quotient and turned into a real-space FC2
    by the exact inverse discrete Fourier transform over the same quotient, so the direct
    dynamical matrix of the resulting ``ForceConstants`` equals ``d(q) * I`` at every grid
    point.  ``d`` is even, so the real-space tensors come out real and ``D(-q) = D(q)^*``.
    """
    primitive = Atoms("Ar", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    matrix = np.diag((3, 3, 3)).astype(np.int64)
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference)
    grid = reciprocal_quotient_grid(matrix)
    decomposition = irreducible_reciprocal_grid(
        matrix, PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    )
    spectrum = {}
    for star, representative in enumerate(decomposition.representatives.tolist()):
        for member in decomposition.stars[star].members.tolist():
            spectrum[tuple(int(value) for value in grid.labels[member])] = 1.0 + 0.1 * star
    spectrum.update(overrides or {})

    cells = np.asarray(relation.index.cell_representatives, dtype=np.int64)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    tensors = np.zeros((len(cells), 3, 3))
    for index, cell in enumerate(cells):
        value = 0.0
        for label, d in spectrum.items():
            value += d * np.exp(-2j * np.pi * (np.asarray(label, dtype=float) / grid.denominator) @ cell)
        tensors[index] = (value / len(grid.labels) * masses[0]).real * np.eye(3)
    sparse = SparseOrderForceConstants(
        2,
        np.zeros((len(cells), 2), dtype=np.int32),
        cells.reshape((-1, 1, 3)).astype(np.int32),
        tensors,
    )
    return ForceConstants({}, reference, sparse={2: sparse}, relation=relation)


def test_the_little_group_gate_alone_accepts_a_broken_star_member() -> None:
    """A spectrum that is constant on representatives but broken on a member must be caught.

    This is the counterexample of the merge-blocker review: any ``d(q)`` satisfies the
    little-group condition at the representatives, because the little group fixes the
    representative itself, while the star expansion of that representative silently
    produces the wrong matrix at the members.  The old gate passes; the new full-star gate
    has to refuse, and the public frequency path has to refuse with it.
    """
    from mlfcs.reciprocal.symmetry import require_little_group_covariance

    base = _scalar_spectrum_force_constants()
    grid = reciprocal_quotient_grid(base.relation.supercell_matrix)
    symmetry = PrimitiveSymmetryOperations.from_atoms(base.relation.primitive, symprec=1e-5)
    decomposition = irreducible_reciprocal_grid(base.relation.supercell_matrix, symmetry)
    assert len(grid.labels) == 27 and len(decomposition.representatives) == 4

    # Pick a non-trivial star, a member that is not its representative, and the -q partner.
    star = next(entry for entry in decomposition.stars if len(entry.members) > 1)
    member = next(
        int(value)
        for value in star.members
        if int(value) != int(star.representative)
        and tuple(int(v) for v in grid.labels[int(value)])
        != grid.negative_label(grid.labels[int(star.representative)])
    )
    partner = int(np.flatnonzero(np.all(grid.labels == np.asarray(grid.negative_label(grid.labels[member])), axis=1))[0])
    assert partner != member
    overrides = {
        tuple(int(value) for value in grid.labels[member]): 2.0,
        tuple(int(value) for value in grid.labels[partner]): 2.0,
    }
    broken = _scalar_spectrum_force_constants(overrides)

    masses = np.asarray(broken.relation.primitive.get_masses(), dtype=float)
    terms = fourier_terms(lattice_fc2(broken), broken.relation.primitive)
    build = partial(dynamical_matrices, terms, masses)

    # 1. The little-group gate accepts the broken model: it only ever looks at the
    #    representatives, which are unchanged.
    require_little_group_covariance(
        build,
        masses,
        symmetry,
        decomposition,
        tolerance=1e-6,
        context="counterexample",
    )

    # 2. The member's own matrix disagrees with the representative it is expanded from.
    representative = int(star.representative)
    direct = build(grid.points[member].reshape(1, 3))[0]
    source = build(grid.points[representative].reshape(1, 3))[0]
    assert float(np.max(np.abs(direct - source))) > 1e-3

    # 3. The public expansion and the public frequency path must refuse it.
    from mlfcs.reciprocal.symmetry import expand_star_matrices

    positions = np.asarray(broken.relation.primitive.get_scaled_positions(wrap=False), dtype=float)
    expanded = expand_star_matrices(
        build(grid.points[decomposition.representatives]), decomposition, symmetry, positions
    )
    # The gauge-correct expansion of the *representative* cannot reproduce a member that
    # was deliberately changed: that disagreement is the false positive this test is about.
    assert float(np.max(np.abs(expanded[member] - direct))) > 1e-3

    from mlfcs.reciprocal.symmetry import require_star_covariance

    with pytest.raises(SymmetryViolationError):
        require_star_covariance(
            build, symmetry, decomposition, positions, tolerance=1e-6, context="counterexample"
        )
    with pytest.raises(SymmetryViolationError):
        harmonic_frequencies(broken, 1)


def test_scph_refuses_a_non_covariant_updated_force_constants() -> None:
    """A quartic tensor that breaks the symmetry must not produce a returned FC2.

    The gate that guards the *input* is not enough: what gets expanded and handed back is the
    updated FC2 of every iteration, so the update itself has to be validated before the next
    expansion and before the result is returned.
    """
    from mlfcs.reciprocal.scph.solver import LoopSCPH

    force_constants, fc4 = scph_case("hcp_2x1x1")
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

    def solver(**options) -> LoopSCPH:
        return LoopSCPH(
            fc2=force_constants,
            fc4=broken_fc4,
            temperature=TEMPERATURE,
            interpolation_multiplier=1,
            scph_multiplier=1,
            mixing=1.0,
            max_iterations=1,
            **options,
        )

    with pytest.raises(SymmetryViolationError) as failure:
        solver()._run_single(TEMPERATURE, None)
    message = str(failure.value)
    assert "updated" in message, message
    assert "iteration" in message, message

    # With the gate switched off the solver returns an FC2 that the same gate measures as
    # broken, which is exactly the invalid result the default must never hand back.
    from mlfcs.reciprocal.symmetry import require_star_covariance

    result = solver(symmetry_tolerance=None)._run_single(TEMPERATURE, None)
    relation = result.force_constants.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    positions = relation.primitive.get_scaled_positions(wrap=False)
    symmetry = PrimitiveSymmetryOperations.from_atoms(relation.primitive, symprec=1e-5)
    decomposition = irreducible_reciprocal_grid(relation.supercell_matrix, symmetry)
    terms = fourier_terms(lattice_fc2(result.force_constants), relation.primitive)
    with pytest.raises(SymmetryViolationError):
        require_star_covariance(
            partial(dynamical_matrices, terms, masses),
            symmetry,
            decomposition,
            positions,
            tolerance=1e-6,
            context="returned FC2",
        )


def test_sscha_propagates_the_reciprocal_tolerances() -> None:
    """SSCHA has to hand its own symprec and symmetry tolerance to every sampler it builds."""
    from ase.calculators.lj import LennardJones

    from mlfcs.reciprocal.sscha.solver import SSCHA

    force_constants, _ = scph_case("cubic_2x1x1")
    relation = force_constants.relation
    common = {
        "reference": relation.reference,
        "cutoff": 3.2,
        "symprec": 2.5e-4,
        "symmetry_tolerance": 3.0e-8,
        "max_iterations": 1,
        "snapshots": 32,
        "initial_displacement": 0.02,
    }
    solver = SSCHA(relation.primitive, temperature=300.0, **common)
    ensemble = solver._make_ensemble(force_constants.materialize(2))
    assert ensemble.symprec == 2.5e-4
    assert ensemble.symmetry_tolerance == 3.0e-8

    scheduled = SSCHA(relation.primitive, temperature=[300.0, 400.0], **common)
    results = scheduled.run(LennardJones(epsilon=1.0, sigma=3.0, rc=9.0))
    assert len(results) == 2
    for result in results:
        assert result.symprec == 2.5e-4
        assert result.symmetry_tolerance == 3.0e-8


def _gauge_case(name: str):
    """Return one multi-atomic case whose mesh contains a member with a reduced label.

    The gauge only bites where a member's stored label is a reduced image of ``g q_s``, so
    the cases are chosen (and verified below) for exactly that: a monoatomic coarse mesh can
    look correct while the expansion is wrong.
    """
    if name == "GaAs":
        from ase.build import bulk
        from test_reciprocal_full_grid_oracle import pair_bond_force_constants

        primitive = bulk("GaAs", "zincblende", a=5.653)
        matrix = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
        return pair_bond_force_constants(
            primitive, matrix, cutoff=3.2, spring=1.0, bend=0.5
        )[0]
    return scph_case({"diamond": "diamond_2x1x1", "hcp": "hcp_2x1x1"}[name])[0]


@pytest.mark.parametrize("case", ("diamond", "hcp", "GaAs"))
def test_public_matrix_expansion_applies_the_stored_label_gauge(case: str) -> None:
    """The public expansion must land on the member's own label, gauge included.

    Every member whose stored label is a reduced image of ``g q_s`` needs the positional
    gauge ``Gamma_G``; without it the expansion is wrong by a factor of order one, which a
    single-atom cell or a coarse mesh never shows.
    """
    from mlfcs.reciprocal.symmetry import (
        expand_star_matrices,
        star_member_gauge,
        star_member_operator,
    )

    force_constants = _gauge_case(case)
    relation = force_constants.relation
    positions = np.asarray(relation.primitive.get_scaled_positions(wrap=False), dtype=float)
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    symmetry = PrimitiveSymmetryOperations.from_atoms(relation.primitive, symprec=1e-5)
    decomposition = irreducible_reciprocal_grid(2 * relation.supercell_matrix, symmetry)
    terms = fourier_terms(lattice_fc2(force_constants), relation.primitive)

    representatives = dynamical_matrices(
        terms, masses, decomposition.full.points[decomposition.representatives]
    )
    direct = dynamical_matrices(terms, masses, decomposition.full.points)
    gauges = [
        float(
            np.max(np.abs(star_member_gauge(symmetry, decomposition, int(member), positions) - 1.0))
        )
        for star in decomposition.stars
        for member in star.members
    ]
    assert max(gauges) > 1e-9, "this case must contain a member whose label is reduced"

    expanded = expand_star_matrices(representatives, decomposition, symmetry, positions)
    scale = float(max(np.max(np.abs(direct)), np.max(np.abs(expanded))))
    np.testing.assert_allclose(expanded, direct, rtol=1e-9, atol=1e-12 * scale)

    without = np.empty_like(direct)
    for member in range(len(decomposition.full.labels)):
        star = int(decomposition.full_to_irreducible[member])
        unitary, antiunitary = star_member_operator(symmetry, decomposition, member)
        source = np.conjugate(representatives[star]) if antiunitary else representatives[star]
        without[member] = unitary @ source @ unitary.conj().T
    assert float(np.max(np.abs(without - direct))) > 1e-3 * scale


def test_pair_bond_models_are_accepted_by_the_gate() -> None:
    """The gate accepts a model that is symmetric by construction, including GaAs."""
    from ase.build import bulk

    primitive = bulk("GaAs", "zincblende", a=5.653)
    matrix = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    fc2, _ = pair_bond_force_constants(primitive, matrix, cutoff=3.2, spring=1.0, bend=0.5)
    mesh = harmonic_frequencies(fc2, 1)
    assert mesh.n_irreducible < mesh.n_qpoints
