r"""Canonical harmonic sampling from one eigen-decomposition per star.

The sampler keeps the two halves of the symmetry reduction strictly apart, because
conflating them is exactly the mistake the reduction invites:

* **the eigen-decomposition is irreducible**: every star representative is diagonalized
  once, in a single batched :func:`numpy.linalg.eigh` call over the ``N_irr``
  representatives, and every full q point receives its modes through
  :func:`~mlfcs.reciprocal.symmetry.star_member_operator` and
  :func:`~mlfcs.reciprocal.symmetry.star_member_gauge`;
* **the random degrees of freedom are complete**: every full q point carries its own
  amplitude, each stream is derived from the global seed and the *exact integer label* of
  its q point, and no star weight ever multiplies a random amplitude.

Vector expansion
================

With the positional gauge of :mod:`mlfcs.reciprocal.fourier`,

.. math::

    D_{ab}(q) = \frac{1}{\sqrt{m_a m_b}}\sum_R \Phi_{ab}(R)
        \exp\left[2\pi i\, q \cdot (R + \tau_b - \tau_a)\right],

a member :math:`m` of the star of representative :math:`s` is reached by the recorded
operation :math:`g` -- conjugated first when the member is reached through time reversal --
with :math:`g q_s = q_m + G`, and the covariance relation of
:mod:`mlfcs.reciprocal.symmetry` reads

.. math::

    D(q_m) = \Gamma_G\, U_g\, \overline{D(q_s)}^{[a]}\, U_g^{\dagger}\, \Gamma_G^{\dagger}.

Applying both sides to one eigenvector column of the representative and using
:math:`\overline{D}\,\overline{w} = \overline{\lambda w} = \lambda \overline{w}` (the
eigenvalues are real, also on an antiunitary member) gives

.. math::

    D(q_m)\left(\Gamma_G U_g \overline{w_\alpha(q_s)}^{[a]}\right)
        = \lambda_\alpha \left(\Gamma_G U_g \overline{w_\alpha(q_s)}^{[a]}\right),

so a **vector** expands with exactly the operator that expands a matrix, applied column by
column:

.. math::

    w_\alpha(q_m) = \Gamma_G\, U_g\, \overline{w_\alpha(q_s)}^{[a]}.

The eigenvalues travel with the representative, in the representative's order, so no member
is ever diagonalized again and the mode order is identical on a whole star.  The tests pin
this both ways: the expanded columns diagonalize a dynamical matrix rebuilt independently at
the member's own label, and the real-space field synthesized from them carries the member's
Bloch phases.

The Bloch field
===============

The physical displacement of mode :math:`\alpha` at member :math:`m` is the Bloch wave
:math:`\Phi_{m\alpha}(c, a) = w_\alpha(q_m)(a)\exp[2\pi i\, q_m \cdot (R_c + \tau_a)]` over
the cells ``c`` of the supercell, so one drawn label contributes the real field

.. math::

    u_{c,a} = \frac{1}{\sqrt{m_a}}\, \mathrm{Re}\left[
        f_m \sum_\alpha \xi_\alpha \sigma_\alpha \Phi_{m\alpha}(c, a)\right],

with :math:`\xi` a standard complex coefficient and :math:`\sigma` the displacement
amplitude of :func:`~mlfcs.reciprocal.statistics.mode_sigma`.  The cell translations
:math:`R_c` are integers, so the reduced negative label of a q point satisfies
:math:`\exp[2\pi i\, q_{-l}\cdot R_c] = \overline{\exp[2\pi i\, q_l \cdot R_c]}`: the
discrete Fourier coefficients of a real field obey :math:`\hat u(-l) = \overline{\hat u(l)}`.
A q/-q pair therefore carries *one* complex amplitude vector, and its real field term is the
``Re`` of the drawing side alone -- the conjugate side is what that ``Re`` already contains.

Synthesizing both members with conjugated coefficients instead would add a cross term the
physical ensemble does not have.  Writing :math:`\Psi(q_{-l}) = \overline{\Psi(q_l)}\,U` for
the relation between the two sides' expanded bases -- :math:`U` is unitary and preserves the
equal-amplitude blocks, which the tests measure -- the conjugate amplitude the partner owes
the real field is :math:`U^{\dagger}\overline{\xi}`, not :math:`\overline{\xi}`.  That extra
unitary is not the identity: on the simplest cell it is a global sign, and conjugating the
coefficients alone then *cancels* the pair instead of doubling it.  The pair is therefore
synthesized from the drawing side alone, whose real part already contains the conjugate
partner; this is the one place where the matrix rule of
:func:`~mlfcs.reciprocal.symmetry.expand_star_matrices` cannot be replayed verbatim for a
field.  The pair's two label contributions are equal as covariance matrices
(:math:`\mathrm{Re}[W(q_m)\Phi_m\Phi_m^{\dagger}] =
\mathrm{Re}[W(q_n)\Phi_n\Phi_n^{\dagger}]`, which follows from the block-preserving
:math:`U` and from the two labels' cell phases being exact conjugates), so scaling the
drawing side by :math:`\sqrt{2/N_c}` gives the pair exactly its two contributions.

A label with :math:`q = -q + G` is its own negative: its field term is a standing wave whose
amplitude is real, and the sampler spans that real amplitude space with the real and
imaginary parts of the member's Bloch vectors at the scale :math:`1/\sqrt{N_c}`.  That
reproduces the full-grid covariance -- which the old single-real-draw treatment did only up
to an arbitrary eigensolver phase -- while using real coefficients throughout.

The Gamma point additionally projects the three uniform translations out before the single
batched diagonalization, exactly as before, and marks them so that they are never sampled,
never counted and never reported as the minimum frequency.

Random streams
==============

Each drawn label owns one generator seeded by the global seed *and* the exact full-grid
integer label, so the numbers a label receives cannot move when spglib returns the same
operations in another order, when the star decomposition is enumerated differently, or when
another q point is added to the same sampler.  The drawing side of a pair is the smaller
full-grid index, which is a property of the labels alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms, units

from mlfcs.exceptions import SymmetryViolationError
from mlfcs.reciprocal.fourier import compact_dynamical_matrix
from mlfcs.reciprocal.grid import IrreducibleReciprocalGrid, irreducible_reciprocal_grid
from mlfcs.reciprocal.statistics import HBAR_ASE, OMEGA_TO_THZ, mode_sigma
from mlfcs.reciprocal.symmetry import (
    star_member_gauge,
    star_member_operator,
    validate_site_masses,
)
from mlfcs.structure.supercell_mapping import PeriodicIndex
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

Statistics = Literal["quantum", "classical"]
ImaginaryModePolicy = Literal["error", "absolute", "exclude"]

_HBAR_ASE = HBAR_ASE
_OMEGA_TO_THZ = OMEGA_TO_THZ
#: Uniform translations projected out of the Gamma dynamical matrix before sampling.
_TRANSLATIONS = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingState:
    n_qpoints: int
    n_irreducible: int
    total_modes: int
    sampled_modes: int
    excluded_modes: int
    imaginary_modes: int
    minimum_frequency_thz: float
    maximum_displacement: float | None
    maximum_sampled_displacement: float
    clipped_atoms: int
    affected_snapshots: int
    #: The geometric tolerance that identified the structure and the relative physical
    #: tolerance the force constants had to satisfy; both are recorded with the state they
    #: produced rather than left as module constants.
    symprec: float
    symmetry_tolerance: float | None


@dataclass(frozen=True, slots=True)
class _StarModes:
    """One star representative: the only place the sampler enters the eigensolver."""

    index: int
    label: tuple[int, int, int]
    qpoint: np.ndarray
    weight: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    frequencies_thz: np.ndarray
    included: np.ndarray
    translations: np.ndarray


@dataclass(frozen=True, slots=True)
class _MemberModes:
    """One full q point and the eigen-decomposition expanded from its representative."""

    index: int
    label: tuple[int, int, int]
    qpoint: np.ndarray
    star: int
    partner: int
    antiunitary: bool
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    frequencies_thz: np.ndarray
    included: np.ndarray
    translations: np.ndarray


class HarmonicSampler:
    """Canonical harmonic sampling directly from translation-reduced FC2."""

    def __init__(
        self,
        primitive: Atoms,
        supercell: Atoms,
        compact_fc2: np.ndarray,
        *,
        temperature: float,
        statistics: Statistics = "quantum",
        cutoff_frequency: float = 0.01,
        imaginary_modes: ImaginaryModePolicy = "error",
        imaginary_tolerance: float = 1e-6,
        max_displacement: float | None = None,
        symprec: float = 1e-5,
        symmetry_tolerance: float | None = 1e-6,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if statistics not in {"quantum", "classical"}:
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if cutoff_frequency < 0 or imaginary_tolerance < 0:
            raise ValueError("frequency tolerances must be non-negative")
        if imaginary_modes not in {"error", "absolute", "exclude"}:
            raise ValueError("imaginary_modes must be 'error', 'absolute', or 'exclude'")
        if max_displacement is not None and max_displacement <= 0:
            raise ValueError("max_displacement must be positive or None")
        if symmetry_tolerance is not None and symmetry_tolerance < 0:
            raise ValueError("symmetry_tolerance must be non-negative or None")

        self.primitive = primitive.copy()
        self.supercell = supercell.copy()
        self.temperature = float(temperature)
        self.statistics = statistics
        self.cutoff_frequency = float(cutoff_frequency)
        self.imaginary_modes = imaginary_modes
        if imaginary_modes != "error":
            logger.warning(
                "Imaginary harmonic modes use policy '%s'; frequencies below %.3e THz "
                "will not raise an exception",
                imaginary_modes,
                imaginary_tolerance,
            )
        self.imaginary_tolerance = float(imaginary_tolerance)
        self.max_displacement = max_displacement
        # The tolerance that identifies the primitive symmetry is a geometric quantity and
        # is therefore an explicit argument, recorded next to the decomposition it built.
        self.symprec = float(symprec)
        # A physical covariance check tolerance, distinct from symprec: symprec decides
        # which operations the structure has, this decides how exactly the force constants
        # must respect them.  None switches the check off explicitly.
        self.symmetry_tolerance = (
            None if symmetry_tolerance is None else float(symmetry_tolerance)
        )
        self._compact = np.asarray(compact_fc2, dtype=float)
        self._n_primitive = len(primitive)
        self._translations = np.asarray(supercell.arrays["cell_translation"], dtype=np.int64)
        self._primitive_index = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
        matrix = supercell.info.get("mlfcs_supercell_matrix")
        if matrix is None:
            raise ValueError("supercell is missing the MLFCS supercell-matrix metadata")
        self._index = PeriodicIndex(self._primitive_index, self._translations, np.asarray(matrix))
        self._n_cells = self._index.n_cells
        expected = (self._n_primitive, len(supercell), 3, 3)
        if self._compact.shape != expected:
            raise ValueError(f"compact FC2 must have shape {expected}, got {self._compact.shape}")
        if len(supercell) != self._n_cells * self._n_primitive:
            raise ValueError("supercell atom count and translation metadata disagree")
        self._masses = np.asarray(primitive.get_masses(), dtype=float)
        self._positions = np.asarray(primitive.get_scaled_positions(wrap=False), dtype=float)
        cells: dict[tuple[int, int, int], np.ndarray] = {}
        for translation in self._translations:
            cells.setdefault(self._index.residue(translation), translation)
        self._cell_translations = np.asarray(list(cells.values()), dtype=np.int32)
        self._cell_atoms = np.asarray(
            [
                [self._index.atom(site, translation) for site in range(self._n_primitive)]
                for translation in self._cell_translations
            ],
            dtype=np.int32,
        )
        self._symmetry = PrimitiveSymmetryOperations.from_atoms(
            self.primitive, symprec=self.symprec
        )
        # Time reversal is a symmetry of any real force-constant model, so the decomposition
        # always uses it: a star that is not closed under q -> -q would hide the conjugate
        # partner of a pair in another star and diagonalize the same spectrum twice.
        self._grid = irreducible_reciprocal_grid(
            self._index.supercell_matrix, self._symmetry, time_reversal=True
        )
        validate_site_masses(self._symmetry, self._masses, context="HarmonicSampler")
        self._stars = self._prepare_irreducible()
        self._members = self._expand_full_grid()
        self._validate_expansion()
        self._last_state: SamplingState | None = None

    @property
    def grid(self) -> IrreducibleReciprocalGrid:
        """Return the exact star decomposition every reduction here is built on."""
        return self._grid

    @property
    def irreducible_qpoints(self) -> np.ndarray:
        """Return the representative q points, in representative order."""
        return self._grid.full.points[self._grid.representatives]

    @property
    def weights(self) -> np.ndarray:
        """Return the star weights, which sum to the number of full q points."""
        return np.asarray(self._grid.weights, dtype=np.int64)

    @property
    def irreducible_frequencies(self) -> np.ndarray:
        """Return the frequencies of the representatives, in THz, one row per star."""
        return np.asarray([star.frequencies_thz for star in self._stars])

    def full_qpoints(self) -> np.ndarray:
        """Return the q points of the full grid, in full-grid order."""
        return self._grid.full.points

    def expand_frequencies(self) -> np.ndarray:
        """Return the full-grid frequencies, in full-grid order.

        A star member carries its representative's spectrum, so this is a copy of the
        representative array indexed by the star map and never an eigensolver call.
        """
        return np.asarray([member.frequencies_thz for member in self._members])

    @property
    def state(self) -> SamplingState:
        if self._last_state is None:
            return self._state(0.0, 0, 0)
        return self._last_state

    def sample(self, snapshots: int, *, random_seed: int | None = None) -> np.ndarray:
        """Draw ``snapshots`` supercell displacement fields from the sampler's ensemble."""
        if snapshots < 1:
            raise ValueError("snapshots must be positive")
        root = np.random.SeedSequence(random_seed)
        displacement = np.zeros((snapshots, self._n_cells, self._n_primitive, 3), dtype=float)
        inverse_root_mass = 1.0 / np.sqrt(self._masses)[None, :, None]
        for modes in self._members:
            # Exactly one side of a q/-q pair draws: its negative is the conjugate amplitude
            # inside the real part taken below, and the smaller full-grid index is the side
            # the labels alone single out.
            if modes.partner < modes.index:
                continue
            generator = np.random.default_rng(self._label_stream(root, modes.label))
            draw = generator.standard_normal((2, snapshots, len(modes.eigenvalues)))
            sigma = np.where(modes.included, self._mode_sigma(modes.eigenvalues), 0.0)
            coefficients = (draw[0] + 1j * draw[1]) * sigma
            reduced = coefficients @ modes.eigenvectors.T
            phase = np.exp(
                2j
                * np.pi
                * np.einsum(
                    "cad,d->ca",
                    self._cell_translations[:, None, :] + self._positions[None, :, :],
                    modes.qpoint,
                )
            )
            # A pair contributes twice and a self-conjugate label once; both carry the same
            # unit-variance complex coefficient, so only this factor distinguishes them.
            factor = (
                np.sqrt(2.0 / self._n_cells)
                if modes.partner != modes.index
                else 1.0 / np.sqrt(self._n_cells)
            )
            field = factor * np.real(
                reduced.reshape(snapshots, self._n_primitive, 3)[:, None, :, :]
                * phase[None, :, :, None]
            )
            displacement += field * inverse_root_mass

        values = np.zeros((snapshots, len(self.supercell), 3), dtype=float)
        values[:, self._cell_atoms.reshape(-1)] = displacement.reshape(
            snapshots, self._n_cells * self._n_primitive, 3
        )
        norms = np.linalg.norm(values, axis=2)
        maximum_sampled = float(np.max(norms))
        clipped_atoms = affected_snapshots = 0
        if self.max_displacement is not None:
            clipped = norms > self.max_displacement
            clipped_atoms = int(np.count_nonzero(clipped))
            affected_snapshots = int(np.count_nonzero(np.any(clipped, axis=1)))
            scale = np.ones_like(norms)
            scale[clipped] = self.max_displacement / norms[clipped]
            values *= scale[..., None]
        self._last_state = self._state(maximum_sampled, clipped_atoms, affected_snapshots)
        return values

    def harmonic_free_energy(self) -> float:
        """Return harmonic free energy per primitive cell in eV.

        The representatives carry the whole star, so weighting their contributions by the
        star sizes is the full-grid sum and never a repeated eigensolver call or a scaled
        random amplitude.
        """
        total = 0.0
        for star in self._stars:
            omega = np.sqrt(np.abs(star.eigenvalues[star.included]))
            energy = _HBAR_ASE * omega
            if self.statistics == "classical":
                if self.temperature == 0:
                    contribution = np.zeros_like(energy)
                else:
                    contribution = (
                        units.kB * self.temperature * np.log(energy / (units.kB * self.temperature))
                    )
            elif self.temperature == 0:
                contribution = energy / 2
            else:
                x = energy / (units.kB * self.temperature)
                contribution = energy / 2 + units.kB * self.temperature * np.log(-np.expm1(-x))
            total += float(star.weight) * float(np.sum(contribution))
        return total / self._n_cells

    def _prepare_irreducible(self) -> tuple[_StarModes, ...]:
        r"""Diagonalize every star representative once and cache the record.

        One batched ``eigh`` call covers the whole irreducible wedge.  The Gamma matrix is
        projected onto the orthogonal complement of the mass-weighted uniform translations
        first, so the three zero modes of the acoustic sum rule cannot couple to the
        internal modes, exactly as the pre-reduction sampler did at Gamma.
        """
        grid = self._grid
        points = grid.full.points[grid.representatives]
        matrices = np.stack([self._dynamical_matrix(qpoint) for qpoint in points])
        bases = self._translation_basis()
        gamma = self._gamma_star()
        if gamma is not None:
            matrices[gamma] = self._project_gamma(matrices[gamma], bases)
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
        stars = []
        imaginary_count = 0
        for star, index in enumerate(grid.representatives.tolist()):
            values = eigenvalues[star]
            frequencies = np.sqrt(np.abs(values)) * np.sign(values) * _OMEGA_TO_THZ
            translations = np.zeros(len(values), dtype=bool)
            if star == gamma:
                overlap = bases.T @ eigenvectors[star]
                strength = np.einsum("kj,kj->j", overlap, overlap)
                translations[np.argsort(strength)[-_TRANSLATIONS:]] = True
                # The projected directions are zero modes by construction; reporting the
                # rounding noise of the projection as a frequency would be an artefact.
                frequencies = np.where(translations, 0.0, frequencies)
            # The projected translations are not modes of the sampler: they are never
            # sampled, counted or reported, exactly as when they were dropped outright.
            imaginary = (frequencies < -self.imaginary_tolerance) & ~translations
            if self.imaginary_modes == "error" and np.any(imaginary):
                minimum = float(np.min(frequencies[~translations]))
                raise ValueError(
                    f"imaginary harmonic modes detected (minimum {minimum:.8f} THz); "
                    "choose imaginary_modes='absolute' or 'exclude' explicitly"
                )
            included = np.abs(frequencies) > self.cutoff_frequency
            if self.imaginary_modes == "exclude":
                included &= ~imaginary
            included &= ~translations
            imaginary_count += int(grid.weights[star]) * int(np.count_nonzero(imaginary))
            stars.append(
                _StarModes(
                    int(index),
                    tuple(int(value) for value in grid.full.labels[index]),
                    grid.full.points[index],
                    int(grid.weights[star]),
                    values,
                    eigenvectors[star],
                    frequencies,
                    included,
                    translations,
                )
            )
        self._imaginary_count = imaginary_count
        return tuple(stars)

    def _expand_full_grid(self) -> tuple[_MemberModes, ...]:
        r"""Carry every representative's modes onto every full q point of its star.

        The expansion is the column-wise form of the matrix rule the SCPH covariance uses:
        ``w(gq) = Gamma_g U_g w(q)``, with a complex conjugation first on an antiunitary
        member.  The eigenvalues and inclusion masks are the representative's own arrays.
        """
        grid = self._grid
        labels = grid.full.labels
        denominator = grid.full.denominator
        lookup = {tuple(int(value) for value in label): index for index, label in enumerate(labels)}
        members = []
        for index in range(len(labels)):
            star = int(grid.full_to_irreducible[index])
            record = self._stars[star]
            unitary, antiunitary = star_member_operator(self._symmetry, grid, index)
            gauge = star_member_gauge(self._symmetry, grid, index, self._positions)
            source = record.eigenvectors.conj() if antiunitary else record.eigenvectors
            label = tuple(int(value) for value in labels[index])
            negative = tuple(int(value) for value in np.mod(-np.asarray(label), denominator))
            members.append(
                _MemberModes(
                    index,
                    label,
                    grid.full.points[index],
                    star,
                    lookup[negative],
                    bool(antiunitary),
                    record.eigenvalues,
                    gauge[:, None] * (unitary @ source),
                    record.frequencies_thz,
                    record.included,
                    record.translations,
                )
            )
        return tuple(members)

    def _project_gamma(self, matrix: np.ndarray, bases: np.ndarray) -> np.ndarray:
        """Return the Gamma matrix with the uniform translations projected out.

        The three acoustic zero modes of a translation-invariant model are not sampled modes,
        so the representative that carries Gamma is diagonalized in the orthogonal complement
        of the mass-weighted translations.  Every later comparison has to use the same
        projected matrix, otherwise it would compare a reconstruction with a matrix that was
        never diagonalized.
        """
        projected = np.asarray(matrix).real
        projected = projected - bases @ (bases.T @ projected)
        projected = projected - (projected @ bases) @ bases.T
        return projected

    def _validate_expansion(self) -> None:
        r"""Raise unless every expanded eigenbasis reproduces its member's own matrix.

        The sampler synthesizes a displacement field from ``w(gq) = Gamma_G U_g w(q_s)``, so
        the whole construction is only as sound as that identity.  Rebuilding
        ``E diag(lambda) E^dagger`` from the stored member basis and comparing it with the
        member's own dynamical matrix validates the rotation, the site permutation, the
        positional gauge and the antiunitary conjugation at once, on every full q point --
        including the members whose label reduction bites and the pairs whose partner is
        represented by a conjugate amplitude.  A reconstruction that fails it would still
        produce a plausible-looking spectrum, which is why it is checked rather than assumed.
        """
        tolerance = self.symmetry_tolerance
        if tolerance is None:
            return
        gamma = self._gamma_star()
        bases = self._translation_basis()
        matrices = [self._dynamical_matrix(member.qpoint) for member in self._members]
        for index, member in enumerate(self._members):
            if member.star == gamma:
                matrices[index] = self._project_gamma(matrices[index], bases)
        # The scale is the largest element over the whole mesh, not per member: a Gamma
        # matrix of a translation-invariant model is zero by construction, so a per-member
        # scale would turn its rounding noise into a violation.
        scale = max((float(np.max(np.abs(matrix))) for matrix in matrices), default=0.0)
        allowed = tolerance * scale if scale > 0 else tolerance
        for member, direct in zip(self._members, matrices, strict=True):
            star = self._stars[member.star]
            reconstructed = (member.eigenvectors * star.eigenvalues[None, :]) @ (
                member.eigenvectors.conj().T
            )
            residual = float(np.max(np.abs(reconstructed - direct)))
            if residual > allowed:
                raise SymmetryViolationError(
                    f"HarmonicSampler: the modes expanded for label {list(member.label)} do "
                    f"not reproduce that member's dynamical matrix: residual {residual:.6e} "
                    f"against an allowed {allowed:.6e} (mesh scale {scale:.6e}). The star "
                    "expansion, the positional gauge or the antiunitary conjugation is "
                    "inconsistent with the force constants."
                )

    def _translation_basis(self) -> np.ndarray:
        """Return the orthonormal mass-weighted uniform translations, one per column."""
        basis = np.zeros((3 * self._n_primitive, _TRANSLATIONS))
        for axis in range(_TRANSLATIONS):
            basis[axis::3, axis] = np.sqrt(self._masses)
        return basis / np.linalg.norm(basis, axis=0)

    def _gamma_star(self) -> int:
        """Return the star index that owns the Gamma point, which is always irreducible."""
        zero = np.flatnonzero(np.all(self._grid.full.labels == 0, axis=1))
        if len(zero) != 1:
            raise RuntimeError("the reciprocal grid must contain the Gamma point exactly once")
        star = int(self._grid.full_to_irreducible[int(zero[0])])
        if int(self._grid.representatives[star]) != int(zero[0]):
            raise RuntimeError("the Gamma point must be the representative of its own star")
        return star

    def _dynamical_matrix(self, qpoint: np.ndarray) -> np.ndarray:
        """Return the mass-weighted dynamical matrix in the positional gauge.

        The kernel lives in :mod:`mlfcs.reciprocal.fourier`, so this sampler and the SCPH
        path build the same matrix and can share one symmetry representation.
        """
        return compact_dynamical_matrix(
            self._compact,
            self._cell_atoms,
            self._cell_translations,
            self._positions,
            self._masses,
            qpoint,
        )

    def _mode_sigma(self, eigenvalues: np.ndarray) -> np.ndarray:
        return mode_sigma(
            eigenvalues,
            temperature=self.temperature,
            statistics=self.statistics,
        )

    def _label_stream(
        self, root: np.random.SeedSequence, label: tuple[int, int, int]
    ) -> np.random.SeedSequence:
        """Return the stream of one q point: the global seed plus the exact grid label."""
        entropy = np.atleast_1d(np.asarray(root.entropy, dtype=np.int64)).ravel()
        return np.random.SeedSequence([*entropy.tolist(), *label])

    def _state(
        self, maximum_sampled: float, clipped_atoms: int, affected_snapshots: int
    ) -> SamplingState:
        weights = np.asarray(self._grid.weights, dtype=np.int64)
        represented = np.asarray(
            [int(np.count_nonzero(~star.translations)) for star in self._stars], dtype=np.int64
        )
        included = np.asarray(
            [int(np.count_nonzero(star.included)) for star in self._stars], dtype=np.int64
        )
        total = int(np.sum(weights * represented))
        sampled = int(np.sum(weights * included))
        # The projected translations are already outside ``total``: the sampler represents
        # the internal modes of Gamma, not the three uniform translations of the supercell.
        frequencies = np.concatenate(
            [star.frequencies_thz[~star.translations] for star in self._stars]
        )
        return SamplingState(
            int(self._grid.n_qpoints),
            int(self._grid.n_irreducible),
            total,
            sampled,
            int(total - sampled),
            int(self._imaginary_count),
            float(np.min(frequencies)) if len(frequencies) else float("nan"),
            self.max_displacement,
            maximum_sampled,
            clipped_atoms,
            affected_snapshots,
            float(self.symprec),
            self.symmetry_tolerance,
        )


__all__ = ["HarmonicSampler", "SamplingState"]
