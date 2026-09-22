r"""The harmonic sampler diagonalizes representatives and keeps every random degree of freedom.

Stage F of
``docs/zh/development/research/reciprocal-space-symmetry-reduction-plan.md`` replaces the
per-``±``-pair eigen-decomposition of the sampler by the star reduction, and that change has
two halves which fail in completely different ways:

* **the eigen-decomposition is irreducible**: one batched ``eigh`` call covers the ``N_irr``
  representatives, and the modes of every other full q point are the column-wise image of
  its representative's modes under ``Gamma_g U_g`` (with a complex conjugation first on an
  antiunitary member).  A wrong expansion would still return a spectrum -- of the wrong
  matrices -- so the residuals below are measured against a dynamical matrix rebuilt
  independently at each member's own label, and against the real-space Bloch phases the
  synthesized field must carry;
* **the random degrees of freedom are complete**: every full q point keeps its own amplitude
  stream, keyed by the exact integer label, a q/-q pair draws one complex amplitude and reads
  the conjugate partner inside the real part of that one field term, a label with
  ``q = -q + G`` draws real degrees of freedom, and no star weight ever scales an amplitude.

The covariance claims are compared against two independent oracles: the exact Cartesian
supercell covariance of the same force constants (``_classical_supercell_covariance``) and
the full-grid Bloch kernel built from the sampler's own mode records
(``_sampler_bloch_covariance``).  A sampler that scaled a representative by its star weight,
that reused one random draw for a whole star, or that dropped the antiunitary conjugation
fails one of these and not the other, which is why both are checked.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from test_reciprocal_full_grid_oracle import (
    _CASES,
    _classical_supercell_covariance,
    _force_constants,
    _sampler_bloch_covariance,
    _supercell_force_constants,
    cubic_argon,
    pair_bond_force_constants,
)

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.modes import internal_mode_basis
from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler
from mlfcs.reciprocal.statistics import HBAR_ASE, OMEGA_TO_THZ
from mlfcs.reciprocal.symmetry import star_member_gauge, star_member_operator
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

TEMPERATURE = 300.0
FREQUENCY_RTOL = 1e-9
FREQUENCY_ATOL = 1e-6
#: Eight standard errors of a covariance entry drawn from independent snapshots.
SAMPLING_STANDARD_ERRORS = 8.0
SAMPLING_SNAPSHOTS = 200_000
SAMPLING_SEED = 815

CASES = (
    "cubic_2x1x1",
    "cubic_3x1x1",
    "fcc_2x1x1",
    "hcp_2x1x1",
    "diamond_2x1x1",
    "diamond_nondiagonal",
)
#: ``2x2x1`` cubic argon: the q grid carries a star of size two, so a star-weighted
#: amplitude would be visible in the covariance.
STAR_WEIGHT_SUPERCELL = (2, 2, 1)
STAR_WEIGHT_CUTOFF = 4.5
ZINCBLENDE_SUPERCELL = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)


def _sampler(name: str, **options) -> HarmonicSampler:
    force_constants, _ = _force_constants(name)
    return HarmonicSampler(
        force_constants.relation.primitive,
        force_constants.relation.reference,
        force_constants.materialize(2),
        temperature=TEMPERATURE,
        statistics="classical",
        **options,
    )


def _star_weight_case():
    """Return the force constants and exact covariance of the ``2x2x1`` cubic grid."""
    primitive = cubic_argon()
    force_constants, _ = pair_bond_force_constants(
        primitive,
        STAR_WEIGHT_SUPERCELL,
        cutoff=STAR_WEIGHT_CUTOFF,
        spring=1.0,
        bend=0.5,
    )
    reference = force_constants.relation.reference
    exact = _classical_supercell_covariance(
        primitive,
        reference,
        _supercell_force_constants(
            primitive, reference, cutoff=STAR_WEIGHT_CUTOFF, spring=1.0, bend=0.5
        ),
        TEMPERATURE,
    )
    return force_constants, exact


def _zincblende_case():
    """Return the zincblende GaAs force constants and exact supercell covariance.

    Every crystal pinned by the full-grid oracle is centrosymmetric, where inversion already
    maps ``q`` onto ``-q`` and the antiunitary branch of the expansion is never taken.
    Zincblende has no inversion centre, and this grid has stars whose members are reachable
    only through time reversal.
    """
    primitive = bulk("GaAs", "zincblende", a=5.653)
    force_constants, _ = pair_bond_force_constants(
        primitive, ZINCBLENDE_SUPERCELL, cutoff=3.2, spring=1.0, bend=0.5
    )
    reference = force_constants.relation.reference
    exact = _classical_supercell_covariance(
        primitive,
        reference,
        _supercell_force_constants(primitive, reference, cutoff=3.2, spring=1.0, bend=0.5),
        TEMPERATURE,
    )
    return force_constants, exact


def _sampler_of(force_constants, **options) -> HarmonicSampler:
    return HarmonicSampler(
        force_constants.relation.primitive,
        force_constants.relation.reference,
        force_constants.materialize(2),
        temperature=TEMPERATURE,
        statistics="classical",
        **options,
    )


def _dynamical_matrices(sampler: HarmonicSampler, force_constants) -> np.ndarray:
    """Return every full-grid dynamical matrix from the exact-lattice kernel."""
    primitive = force_constants.relation.primitive
    masses = np.asarray(primitive.get_masses(), dtype=float)
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    return dynamical_matrices(terms, masses, sampler.full_qpoints())


def _full_grid_frequencies(sampler: HarmonicSampler, force_constants) -> np.ndarray:
    values = np.linalg.eigvalsh(_dynamical_matrices(sampler, force_constants))
    return np.sqrt(np.abs(values)) * np.sign(values) * OMEGA_TO_THZ


def _atom_phase(positions: np.ndarray, qpoint: np.ndarray) -> np.ndarray:
    return np.kron(np.exp(2j * np.pi * (positions @ qpoint)), np.ones(3, dtype=complex))


def _free_energy(frequencies_thz: np.ndarray, temperature: float, statistics: str) -> np.ndarray:
    """Return the per-mode harmonic free energy of frequencies given in THz."""
    energy = HBAR_ASE * np.asarray(frequencies_thz) / OMEGA_TO_THZ
    if statistics == "classical":
        if temperature == 0:
            return np.zeros_like(energy)
        return units.kB * temperature * np.log(energy / (units.kB * temperature))
    if temperature == 0:
        return energy / 2
    x = energy / (units.kB * temperature)
    return energy / 2 + units.kB * temperature * np.log(-np.expm1(-x))


def _diagonalization_residual(matrix: np.ndarray, vectors: np.ndarray, values: np.ndarray) -> float:
    """Return ``max |D V - V diag(lambda)|`` for one candidate eigenbasis."""
    return float(np.abs(matrix @ vectors - vectors * values).max())


@pytest.mark.parametrize("name", CASES)
def test_sampler_diagonalizes_representatives_once(name: str, monkeypatch) -> None:
    """One construction enters the eigensolver once per representative, and never again.

    The shape of each call is not pinned: the Gamma representative is solved in the internal
    subspace and the others in the full mass-weighted space, so their widths legitimately
    differ.  What matters is that the cost is one solve per irreducible star.
    """
    calls: list[tuple[int, ...]] = []
    original = np.linalg.eigh

    def counting(values, *args, **kwargs):
        calls.append(np.asarray(values).shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigh", counting)
    sampler = _sampler(name)

    mesh = sampler.grid
    assert len(calls) == mesh.n_irreducible
    assert all(shape[0] == shape[1] for shape in calls)
    assert mesh.n_irreducible <= mesh.n_qpoints

    before = len(calls)
    sampler.expand_frequencies()
    sampler.sample(4, random_seed=3)
    assert len(calls) == before, "neither the expansion nor a draw may re-enter the eigensolver"


@pytest.mark.parametrize("name", CASES)
def test_sampler_reports_the_irreducible_wedge(name: str) -> None:
    """The public surface names the representatives, their weights and the full grid."""
    sampler = _sampler(name)
    mesh = sampler.grid

    assert sampler.grid is mesh
    np.testing.assert_array_equal(
        sampler.irreducible_qpoints, mesh.full.points[mesh.representatives]
    )
    np.testing.assert_array_equal(sampler.weights, mesh.weights)
    np.testing.assert_array_equal(sampler.full_qpoints(), mesh.full.points)
    assert int(sampler.weights.sum()) == mesh.n_qpoints

    state = sampler.state
    assert state.n_qpoints == mesh.n_qpoints
    assert state.n_irreducible == mesh.n_irreducible
    assert state.n_qpoints == len(sampler.full_qpoints())
    assert state.n_irreducible == len(sampler.irreducible_qpoints)
    assert state.sampled_modes <= state.total_modes
    assert state.excluded_modes == state.total_modes - state.sampled_modes
    np.testing.assert_array_equal(
        sampler.irreducible_frequencies, sampler.expand_frequencies()[mesh.representatives]
    )


@pytest.mark.parametrize("name", CASES)
def test_expanded_modes_diagonalize_the_member_dynamical_matrix(name: str) -> None:
    """Every member's modes are eigenvectors of a dynamical matrix built at its own label."""
    sampler = _sampler(name)
    force_constants, _ = _force_constants(name)
    matrices = _dynamical_matrices(sampler, force_constants)
    basis = internal_mode_basis(sampler._masses)
    projector = basis @ basis.conj().T

    for index, member in enumerate(sampler._members):
        matrix = matrices[index]
        if member.translations.any():
            # Gamma: the sampler diagonalizes the internal subspace, so the member's own matrix
            # has to be projected onto the same subspace before comparing.  The translations are
            # carried as zero modes, which reproduce a zero block either way.
            matrix = projector @ matrix @ projector
        residual = _diagonalization_residual(matrix, member.eigenvectors, member.eigenvalues)
        assert residual < 1e-10, f"member {member.label} is not diagonalized: {residual}"


@pytest.mark.parametrize("name", CASES)
def test_synthesized_fields_carry_the_member_bloch_phases(name: str) -> None:
    """The field built from a member's modes is a single Bloch wave at that member's label.

    The complex Bloch field of mode ``alpha`` has exactly one non-zero discrete Fourier
    coefficient, at the member's own label, and it equals the member's star-expanded Bloch
    vector.  Its real part is the displacement field a sampler can draw, and it has exactly
    two coefficients (one when ``q = -q + G``): they are conjugates, and on a q/-q pair the
    pair's two conjugate halves reconstruct the member's own Bloch vector.  That conjugation
    is why one complex amplitude per pair is enough for a real displacement field.
    """
    sampler = _sampler(name)
    positions = sampler._positions
    n_cells = sampler._n_cells
    n_primitive = sampler._n_primitive
    n_modes = 3 * n_primitive
    cell_translations = sampler._cell_translations

    for member in sampler._members:
        partner = sampler._members[member.partner]
        assert partner.partner == member.index
        vectors = (member.eigenvectors * _atom_phase(positions, member.qpoint)[:, None]).reshape(
            n_primitive, 3, n_modes
        )
        cell_phase = np.exp(2j * np.pi * (cell_translations @ member.qpoint))
        partner_phase = np.exp(2j * np.pi * (cell_translations @ partner.qpoint))
        for alpha in range(n_modes):
            complex_field = vectors[:, None, :, alpha] * cell_phase[None, :, None]
            real_field = np.real(complex_field)
            own = np.einsum("c,acd->ad", np.conj(cell_phase), complex_field) / n_cells
            np.testing.assert_allclose(own, vectors[:, :, alpha], rtol=1e-10, atol=1e-12)
            own_real = np.einsum("c,acd->ad", np.conj(cell_phase), real_field) / n_cells
            partner_real = np.einsum("c,acd->ad", np.conj(partner_phase), real_field) / n_cells
            np.testing.assert_allclose(partner_real, np.conj(own_real), rtol=1e-10, atol=1e-12)
            if member.partner == member.index:
                np.testing.assert_allclose(
                    own_real, np.real(vectors[:, :, alpha]), rtol=1e-10, atol=1e-12
                )
            else:
                np.testing.assert_allclose(
                    2.0 * own_real, vectors[:, :, alpha], rtol=1e-10, atol=1e-12
                )
            for neighbour in sampler._members:
                if neighbour.index in (member.index, member.partner):
                    continue
                foreign_phase = np.exp(2j * np.pi * (cell_translations @ neighbour.qpoint))
                leaked = np.einsum("c,acd->ad", np.conj(foreign_phase), real_field) / n_cells
                assert np.abs(leaked).max() < 1e-12, "a Bloch wave leaked onto another label"


@pytest.mark.parametrize("name", CASES)
def test_exactly_one_side_of_every_q_pair_draws(name: str) -> None:
    """Every full q point is covered by exactly one drawing side of its ``q/-q`` pair."""
    sampler = _sampler(name)
    drawn = [member for member in sampler._members if member.partner >= member.index]
    covered: set[int] = set()
    for member in drawn:
        assert member.index not in covered, f"label {member.label} draws twice"
        covered.update({member.index, member.partner})
    assert covered == set(range(sampler.grid.n_qpoints))
    for member in sampler._members:
        negative = np.mod(-np.asarray(member.label), sampler.grid.full.denominator)
        assert tuple(negative.tolist()) == sampler._members[member.partner].label


@pytest.mark.parametrize("name", CASES)
def test_expanded_covariance_equals_the_exact_supercell_covariance(name: str) -> None:
    """The covariance the sampler's records imply is the exact Cartesian supercell value."""
    force_constants, _ = _force_constants(name)
    _, _, cutoff, parameters, _ = _CASES[name]
    exact = _classical_supercell_covariance(
        force_constants.relation.primitive,
        force_constants.relation.reference,
        _supercell_force_constants(
            force_constants.relation.primitive,
            force_constants.relation.reference,
            cutoff=cutoff,
            spring=parameters.get("spring", 1.0),
            bend=parameters.get("bend", 0.0),
        ),
        TEMPERATURE,
    )
    sampler = _sampler(name)
    np.testing.assert_allclose(_sampler_bloch_covariance(sampler), exact, rtol=1e-10, atol=1e-14)


def test_sampled_covariance_matches_the_theory_for_a_general_q_pair() -> None:
    """A general ``q/-q`` pair keeps the full pair covariance, not half of it.

    ``cubic_3x1x1`` is the pinned case whose grid contains ``(1/3, 0, 0)`` and
    ``(2/3, 0, 0)``: one complex amplitude for the pair must reproduce *both* members'
    contributions, which is exactly what the two-sided multiplicity of the full-grid sum
    means.  Half of it would still look like a plausible spectrum.
    """
    sampler = _sampler("cubic_3x1x1")
    drawn = [member for member in sampler._members if member.partner != member.index]
    assert drawn, "this case must have a q/-q pair"
    theory = _sampler_bloch_covariance(sampler)

    samples = sampler.sample(SAMPLING_SNAPSHOTS, random_seed=SAMPLING_SEED)
    flat = samples.reshape(len(samples), -1)
    empirical = flat.T @ flat / len(flat)
    bound = SAMPLING_STANDARD_ERRORS * float(np.abs(theory).max()) / np.sqrt(SAMPLING_SNAPSHOTS)
    np.testing.assert_allclose(empirical, theory, rtol=0.0, atol=bound)
    np.testing.assert_allclose(
        np.abs(samples.mean(axis=0)).max(),
        0.0,
        rtol=0.0,
        atol=SAMPLING_STANDARD_ERRORS * float(np.sqrt(theory.diagonal().max() / len(samples))),
    )


def test_star_weights_never_scale_an_amplitude() -> None:
    """A star of size two is expanded member by member, never summed as one weighted draw."""
    force_constants, exact = _star_weight_case()
    sampler = _sampler_of(force_constants)
    weights = sampler.weights
    assert int(weights.max()) > 1, "this grid must have a star with more than one member"
    assert int(weights.sum()) == sampler.grid.n_qpoints
    assert sampler.grid.n_irreducible < sampler.grid.n_qpoints

    theory = _sampler_bloch_covariance(sampler)
    np.testing.assert_allclose(theory, exact, rtol=1e-10, atol=1e-14)
    samples = sampler.sample(SAMPLING_SNAPSHOTS, random_seed=SAMPLING_SEED)
    flat = samples.reshape(len(samples), -1)
    empirical = flat.T @ flat / len(flat)
    bound = SAMPLING_STANDARD_ERRORS * float(np.abs(theory).max()) / np.sqrt(SAMPLING_SNAPSHOTS)
    np.testing.assert_allclose(empirical, theory, rtol=0.0, atol=bound)


def test_antiunitary_members_need_the_conjugation() -> None:
    """Time-reversal members are expanded with a complex conjugation first."""
    force_constants, exact = _zincblende_case()
    sampler = _sampler_of(force_constants)
    grid = sampler.grid
    antiunitary = [member for member in sampler._members if member.antiunitary]
    assert antiunitary, "this crystal must have members reachable only through time reversal"
    assert int(np.count_nonzero(grid.full_antiunitary)) == len(antiunitary)
    assert grid.n_irreducible < grid.n_qpoints
    # The pinned oracle crystals are centrosymmetric, so inversion already carries q to -q
    # and no member of their grids ever needs the conjugation.
    assert not np.any(_sampler("hcp_2x1x1").grid.full_antiunitary)

    matrices = _dynamical_matrices(sampler, force_constants)
    for member in antiunitary:
        record = sampler._stars[member.star]
        unitary, flag = star_member_operator(sampler._symmetry, grid, member.index)
        gauge = star_member_gauge(sampler._symmetry, grid, member.index, sampler._positions)
        assert flag is True
        conjugated = gauge[:, None] * (unitary @ record.eigenvectors.conj())
        np.testing.assert_allclose(conjugated, member.eigenvectors, rtol=0.0, atol=1e-12)
        assert (
            _diagonalization_residual(
                matrices[member.index], member.eigenvectors, record.eigenvalues
            )
            < 1e-10
        )
        # Without the conjugation the same formula is not an eigenbasis at all.
        unconjugated = gauge[:, None] * (unitary @ record.eigenvectors)
        assert (
            _diagonalization_residual(matrices[member.index], unconjugated, record.eigenvalues)
            > 1e-3
        )

    np.testing.assert_allclose(_sampler_bloch_covariance(sampler), exact, rtol=1e-10, atol=1e-14)


def test_the_partner_basis_is_the_conjugate_times_a_block_unitary() -> None:
    """``Psi(-q) = conj(Psi(q)) U`` with a block-preserving ``U`` that is not the identity.

    The unitary relates the two sides' expanded bases.  It preserves the equal-amplitude
    blocks, which is what makes the two members carry equal covariance kernels, so one
    drawing side with a ``sqrt(2)`` amplitude is the whole pair.  It is not the identity:
    on the monoatomic cell it is a global sign, so the "symmetric" implementation --
    conjugating the coefficients of both sides -- cancels the pair instead of doubling it.
    """
    zincblende, _ = _zincblende_case()
    cases = (
        ("cubic_3x1x1", _sampler("cubic_3x1x1"), True),
        ("zincblende", _sampler_of(zincblende), False),
    )

    for tag, sampler, monoatomic in cases:
        positions = sampler._positions
        pairs = 0
        for member in sampler._members:
            partner = sampler._members[member.partner]
            if partner.index <= member.index:
                continue
            pairs += 1
            vectors = member.eigenvectors * _atom_phase(positions, member.qpoint)[:, None]
            other = partner.eigenvectors * _atom_phase(positions, partner.qpoint)[:, None]
            unitary = vectors.T @ other
            np.testing.assert_allclose(
                unitary @ unitary.conj().T,
                np.eye(unitary.shape[0]),
                rtol=0.0,
                atol=1e-12,
            )
            sigma2 = np.where(member.included, sampler._mode_sigma(member.eigenvalues), 0.0) ** 2
            block = unitary @ np.diag(sigma2) @ unitary.conj().T - np.diag(sigma2)
            assert np.abs(block).max() < 1e-12, f"{tag}: U does not preserve the mode blocks"
            if monoatomic:
                np.testing.assert_allclose(unitary, -np.eye(unitary.shape[0]), rtol=0.0, atol=1e-12)
        assert pairs > 0, f"{tag} must have a q/-q pair"


def test_fixed_seed_reproduces_samples_and_labels_own_their_streams() -> None:
    """A fixed seed is reproducible and each q point owns the numbers it receives."""
    first = _sampler("hcp_2x1x1")
    second = _sampler("hcp_2x1x1")
    left = first.sample(64, random_seed=20240921)
    right = second.sample(64, random_seed=20240921)
    np.testing.assert_array_equal(left, right)

    changed = first.sample(64, random_seed=20240922)
    assert not np.array_equal(left, changed)

    root = np.random.SeedSequence(20240921)
    streams = {
        member.label: np.random.default_rng(first._label_stream(root, member.label))
        .standard_normal(4)
        .tolist()
        for member in first._members
        if member.partner >= member.index
    }
    assert len(streams) == len({tuple(values) for values in streams.values()})
    for member in first._members:
        if member.partner < member.index:
            continue
        generator = np.random.default_rng(first._label_stream(root, member.label))
        np.testing.assert_array_equal(generator.standard_normal(4), streams[member.label])


def test_operation_order_does_not_move_the_streams(monkeypatch) -> None:
    """The stream belongs to a q label, so spglib's operation order cannot move it.

    Reversing the order of the primitive operations leaves the star decomposition and the
    pairing unchanged as sets and changes which operation is recorded for a member, which is
    exactly the freedom the sampler must not read the random numbers from.  The members'
    bases may legitimately differ by phases and by unitaries inside degenerate blocks, so the
    invariant compared here is the covariance those records imply, which is basis free.
    """
    reference = _sampler("diamond_2x1x1")
    original = PrimitiveSymmetryOperations.from_atoms

    def reversed_operations(cls, primitive, *, symprec):
        symmetry = original(primitive, symprec=symprec)
        order = np.arange(symmetry.size)[::-1]
        return dataclasses.replace(
            symmetry,
            rotations=symmetry.rotations[order],
            translations=symmetry.translations[order],
            cartesian_rotations=symmetry.cartesian_rotations[order],
            site_permutations=symmetry.site_permutations[order],
            site_shifts=symmetry.site_shifts[order],
        )

    monkeypatch.setattr(PrimitiveSymmetryOperations, "from_atoms", classmethod(reversed_operations))
    permuted = _sampler("diamond_2x1x1")

    assert permuted.grid.n_irreducible == reference.grid.n_irreducible
    assert permuted.grid.n_qpoints == reference.grid.n_qpoints
    assert permuted.state == reference.state
    root = np.random.SeedSequence(20240921)
    for left, right in zip(permuted._members, reference._members, strict=True):
        assert left.label == right.label
        assert left.partner == right.partner
        if left.partner < left.index:
            continue
        np.testing.assert_array_equal(
            np.random.default_rng(permuted._label_stream(root, left.label)).standard_normal(4),
            np.random.default_rng(reference._label_stream(root, right.label)).standard_normal(4),
        )
    np.testing.assert_allclose(
        _sampler_bloch_covariance(permuted),
        _sampler_bloch_covariance(reference),
        rtol=1e-10,
        atol=1e-14,
    )


@pytest.mark.parametrize("name", ("cubic_2x1x1", "cubic_3x1x1", "hcp_2x1x1", "diamond_2x1x1"))
def test_free_energy_and_mode_counts_are_the_full_grid_sums(name: str) -> None:
    """Spectrum, free energy and mode counts are checked against a full-grid diagonalization.

    The sampler represents the internal modes of Gamma, so the three uniform translations of
    the supercell are outside ``total_modes``; every other full-grid mode is counted once,
    and the star-weighted representative sum must reproduce the full-grid sum exactly.
    """
    sampler = _sampler(name)
    force_constants, _ = _force_constants(name)
    frequencies = _full_grid_frequencies(sampler, force_constants)
    n_modes = frequencies.shape[1]

    represented = np.ones(frequencies.shape, dtype=bool)
    gamma = int(np.flatnonzero(np.all(sampler.grid.full.labels == 0, axis=1))[0])
    represented[gamma, np.argsort(np.abs(frequencies[gamma]))[:3]] = False
    included = represented & (np.abs(frequencies) > sampler.cutoff_frequency)
    imaginary = represented & (frequencies < 0.0)

    state = sampler.state
    assert state.total_modes == int(np.count_nonzero(represented))
    assert state.sampled_modes == int(np.count_nonzero(included))
    assert state.excluded_modes == state.total_modes - state.sampled_modes
    assert state.imaginary_modes == int(np.count_nonzero(imaginary))
    assert state.total_modes == (sampler.grid.n_qpoints - 1) * n_modes + (n_modes - 3)

    np.testing.assert_allclose(
        np.sort(sampler.expand_frequencies(), axis=1),
        np.sort(frequencies, axis=1),
        rtol=FREQUENCY_RTOL,
        atol=FREQUENCY_ATOL,
    )
    expected = float(
        np.sum(_free_energy(frequencies[included], sampler.temperature, sampler.statistics))
    )
    assert sampler.harmonic_free_energy() * sampler._n_cells == pytest.approx(expected, rel=1e-8)
    # The reported minimum is a physical mode: the exactly projected translations are never
    # allowed to report themselves as the softest frequency.
    assert state.minimum_frequency_thz == pytest.approx(
        float(frequencies[represented].min()), rel=FREQUENCY_RTOL, abs=1e-6
    )
