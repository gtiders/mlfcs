r"""Full-grid oracle for every reciprocal path, pinned before symmetry reduction exists.

This is stage A of
``docs/zh/development/research/reciprocal-space-symmetry-reduction-plan.md``: it records
what the *current*, unreduced implementation computes, so that stages B-G can prove the
irreducible Brillouin-zone reduction and the star expansion reproduce it.  Every expected
value below was produced by the full-grid code; a change in a full-grid number must fail
this module instead of passing silently.

Models and grids
================

Force constants are built in this module (``pair_bond_force_constants``) from one
anharmonic pair-bond potential,

.. math::

    V = \frac{k}{2}\delta_l^2 + \frac{k_t}{2}|\delta_t|^2 + \frac{\lambda}{24}\delta_l^4,
    \qquad \delta_l = \hat r \cdot (u_b - u_a), \quad
    \delta_t = (u_b - u_a) - \hat r \delta_l,

so FC2 and FC4 are physically consistent by construction and carry non-trivial
site/translation labels.  The model is positive semi-definite (an energy sum of squares),
satisfies the acoustic sum rule with exactly three zero modes at Gamma, and keeps non-zero
transverse branches through the ``bend`` stiffness that a purely central pair force
lacks.  Coverage required by plan section 9.2/9.3:

================================  ======================================================
``cubic_2x1x1``                   monoatomic cubic cell, Gamma and the boundary label
                                  ``q = -q + G`` at ``q = (1/2, 0, 0)``
``cubic_3x1x1``                   monoatomic cubic cell, anisotropic grid with a general
                                  ``q/-q`` pair ``(1/3, 0, 0)``/``(2/3, 0, 0)``
``cubic_nondiagonal``             monoatomic cell, sheared supercell matrix
                                  ``[[2, 1, 0], [0, 1, 0], [0, 0, 1]]``
``cubic_multiplier``              monoatomic cell, interpolation multiplier 2, the
                                  eight-point half-integer mesh
``fcc_2x1x1``                     fcc primitive cell
``hcp_2x1x1``                     hexagonal two-atom primitive cell (also drives the SCPH
                                  covariance and the harmonic sampler)
``rhombohedral_2x1x1``            one-atom tilted rhombohedral cell
``diamond_2x1x1``                 non-symmorphic two-atom diamond cell, diagonal supercell
``diamond_nondiagonal``           the same cell with the sheared supercell matrix
================================  ======================================================

Stability of the pinned arrays
==============================

Bitwise-stable, compared exactly (``==``, ``np.testing.assert_array_equal``):

* integer q labels, their order, the grid denominator and the label count;
* the exact relation ``q = label / denominator`` between labels and float q points;
* the key set of the SCPH covariance dictionary, which must equal
  ``_needed_covariances(fc4.sparse[4])``;
* integer mode and snapshot counts, the iteration indices, q-point arrays, and the
  ``converged`` flag of the pinned trajectory.

Tolerance-compared, each tolerance tied to the scale of its quantity:

* frequencies: ``rtol=1e-9`` with ``atol=1e-6`` THz.  The three Gamma acoustic modes are
  diagonalization noise (``|omega| < 3e-7`` THz) whose sign is not fixed by the physics,
  so an absolute floor at that numerical-zero scale is required;
* SCPH covariance blocks: ``rtol=1e-9`` with ``atol=1e-15``.  The blocks are Hermitian
  with ``|block| ~ 1e-3`` in mass-weighted displacement units, and the imaginary part is
  pinned to zero with ``atol=1e-15``;
* sampler free energies, SCPH correction norms and correction history: ``rtol=1e-9``;
* fixed-seed sampled statistics: ``rtol=1e-9`` (they are reproducible for a fixed seed);
* the Monte-Carlo displacement covariance: a statistical bound derived from the theory
  scale and the snapshot count (eight standard errors), never a fixed relative tolerance.

Gauge
=====

Every dynamical matrix uses the positional gauge
``exp[2 pi i q . (R + tau_b - tau_a)]`` owned by :mod:`mlfcs.reciprocal.fourier`, so one
symmetry representation can serve both the SCPH and the sampling paths.
``test_sampler_and_shared_fourier_kernel_agree`` pins that the sampler's compact kernel
and the exact-lattice kernel are the same matrix, and
``test_scph_covariance_matches_the_sampler_kernel`` pins that the SCPH covariance assembly
and the sampler's mode ensemble are the same real-space covariance, which in turn equals
the exact Cartesian supercell covariance.

Observations about the current behaviour
========================================

These are reported, not fixed.  They are properties of the unreduced code that stage E
must be aware of:

#. ``LoopSCPH._covariance`` keeps the three Gamma acoustic modes whenever
   ``frequency_cutoff_thz`` is left at its default ``0.0``.  For a physical FC2 the
   acoustic eigenvalues are rounding noise (``omega ~ 1e-8`` THz), so ``mode_sigma``
   scales like ``omega**-2`` and those modes dominate: on ``hcp_2x1x1`` the
   default-cutoff blocks reach ``5e12`` while the cutoff-guarded blocks are ``2e-3``, and
   the first loop iteration then returns frequencies of ``2.5e6`` THz.  The existing SCPH
   tests never see this because their FC2 carries no zero modes.
   ``test_default_cutoff_covariance_is_dominated_by_the_gamma_zero_modes`` pins the
   hazard; every other pinned covariance and trajectory value here passes an explicit
   ``frequency_cutoff_thz``.
#. The quartic loop correction does not preserve the acoustic sum rule: on
   ``cubic_2x1x1`` the bare Gamma acoustic frequencies are ``~2e-8`` THz and the corrected
   ones are ``~0.14`` THz.
#. ``SamplingState.total_modes`` counts the modes the sampler actually represents, so the
   three uniform translations projected out at Gamma are missing from it: four atoms (12
   modes) report ``total_modes == 9``, exactly the number of non-zero modes.
#. The harmonic sampler diagonalizes the ``N_irr`` representatives in one batched ``eigh``
   call and expands their modes onto every full q point, so ``SamplingState`` reports
   ``n_qpoints`` and ``n_irreducible`` separately and ``_sampler_bloch_covariance`` adds one
   contribution per full q point: the old per-``±``-pair multiplicity of two is now carried
   by the pair's two members, whose expanded covariance kernels are equal.  The sampler
   itself draws a single complex amplitude per q/-q pair -- the conjugate partner lives
   inside the real part of that one field term -- so the reduction removes repeated
   diagonalizations and never an amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.geometry import cellpar_to_cell
from ase.neighborlist import neighbor_list

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.reciprocal.fourier import dynamical_matrix, fourier_terms
from mlfcs.reciprocal.grid import reciprocal_quotient_grid
from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler
from mlfcs.reciprocal.scph.fourier import _needed_covariances, harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.symmetry import expand_star_values
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.supercell_mapping import PeriodicIndex
from mlfcs.tools.supercell import build_supercell

TEMPERATURE = 300.0
FREQUENCY_CUTOFF_THZ = 1.0
# Far below any reachable frequency change, so a pinned trajectory keeps its length.
SCPH_TOLERANCE = 1e-30
FREQUENCY_RTOL = 1e-9
FREQUENCY_ATOL = 1e-6
COVARIANCE_RTOL = 1e-9
COVARIANCE_ATOL = 1e-15
STATISTICS_RTOL = 1e-9
# Standard errors of a covariance entry for independent snapshots.
SAMPLING_STANDARD_ERRORS = 8.0
SAMPLER_SNAPSHOTS = 200_000
SAMPLER_SEED = 815


def cubic_argon() -> Atoms:
    """Return a monoatomic simple-cubic cell."""
    return Atoms("Ar", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)


def fcc_argon() -> Atoms:
    """Return the fcc primitive cell."""
    return bulk("Ar", "fcc", a=4.0)


def hcp_magnesium() -> Atoms:
    """Return a hexagonal close-packed two-atom primitive cell."""
    return bulk("Mg", "hcp", a=3.2, c=5.2)


def rhombohedral_silicon() -> Atoms:
    """Return a one-atom tilted (rhombohedral) cell."""
    return Atoms(
        "Si",
        positions=[[0, 0, 0]],
        cell=cellpar_to_cell([4.2, 4.2, 4.2, 70.0, 70.0, 70.0]),
        pbc=True,
    )


def diamond_silicon() -> Atoms:
    """Return the non-symmorphic two-atom diamond primitive cell."""
    return bulk("Si", "diamond", a=5.43)


def periodic_bonds(
    atoms: Atoms, cutoff: float
) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    """Return the canonical ``(site_a, site_b, shift)`` bonds of a periodic cell.

    Bonds from an atom to its own periodic image are kept once, in the orientation whose
    integer shift is lexicographically smaller.
    """
    first, second, shifts = neighbor_list("ijS", atoms, cutoff)
    bonds: dict[tuple[int, int, tuple[int, int, int]], None] = {}
    for atom_a, atom_b, shift in zip(first.tolist(), second.tolist(), shifts.tolist(), strict=True):
        forward = tuple(int(value) for value in shift)
        backward = tuple(-value for value in forward)
        if atom_a < atom_b:
            key = (atom_a, atom_b, forward)
        elif atom_a > atom_b:
            key = (atom_b, atom_a, backward)
        else:
            key = (atom_a, atom_a, min(forward, backward))
        bonds.setdefault(key, None)
    return tuple(sorted(bonds))


def pair_bond_force_constants(
    primitive: Atoms,
    matrix: object,
    *,
    cutoff: float,
    spring: float = 1.0,
    bend: float = 0.0,
    quartic: float = 1.0,
) -> tuple[ForceConstants, ForceConstants]:
    """Return ``(fc2, fc4)`` sparse IFCs of the anharmonic pair-bond model.

    ``matrix`` is the reference supercell matrix of the structure relation.
    """
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference, symprec=1e-5)
    cell = np.asarray(primitive.cell, dtype=float)
    positions = np.asarray(primitive.get_scaled_positions(wrap=False), dtype=float)
    second: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    fourth: dict[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]], np.ndarray] = {}
    for site_a, site_b, shift in periodic_bonds(primitive, cutoff):
        vector = (positions[site_b] + np.asarray(shift) - positions[site_a]) @ cell
        direction = vector / np.linalg.norm(vector)
        projection = np.outer(direction, direction)
        harmonic = spring * projection + bend * (np.eye(3) - projection)
        anharmonic = quartic * np.einsum(
            "a,b,c,d->abcd", direction, direction, direction, direction
        )
        zero = (0, 0, 0)
        backward = tuple(-value for value in shift)
        for key, value in (
            ((site_a, site_a, zero), harmonic),
            ((site_b, site_b, zero), harmonic),
            ((site_a, site_b, shift), -harmonic),
            ((site_b, site_a, backward), -harmonic),
        ):
            second[key] = second.get(key, 0.0) + value
        for sites, translations, sign in (
            ((site_a, site_a, site_a, site_a), (zero, zero, zero), 1.0),
            ((site_b, site_b, site_b, site_b), (zero, zero, zero), 1.0),
            ((site_a, site_a, site_a, site_b), (zero, zero, shift), -1.0),
            ((site_a, site_a, site_b, site_b), (zero, shift, shift), 1.0),
            ((site_a, site_b, site_b, site_b), (shift, shift, shift), -1.0),
        ):
            key = (sites, translations)
            fourth[key] = fourth.get(key, 0.0) + sign * anharmonic

    second_keys = sorted(second)
    fourth_keys = sorted(fourth)
    fc2 = SparseOrderForceConstants(
        2,
        np.asarray([[a, b] for a, b, _ in second_keys], dtype=np.int32),
        np.asarray([[shift] for _, _, shift in second_keys], dtype=np.int32),
        np.asarray([second[key] for key in second_keys], dtype=float),
    )
    fc4 = SparseOrderForceConstants(
        4,
        np.asarray([sites for sites, _ in fourth_keys], dtype=np.int32),
        np.asarray([translations for _, translations in fourth_keys], dtype=np.int32),
        np.asarray([fourth[key] for key in fourth_keys], dtype=float),
    )
    return (
        ForceConstants({}, relation.reference, sparse={2: fc2}, relation=relation),
        ForceConstants({}, relation.reference, sparse={4: fc4}, relation=relation),
    )


def _supercell_force_constants(
    primitive: Atoms,
    reference: Atoms,
    *,
    cutoff: float,
    spring: float = 1.0,
    bend: float = 0.0,
) -> np.ndarray:
    """Return the Cartesian supercell FC2 ``(N, N, 3, 3)`` of the same bond model.

    Every bond is placed for each supercell cell, so the result is exactly the periodic
    image sum of the pair potential, with no minimum-image approximation.
    """
    index = PeriodicIndex(
        np.asarray(reference.arrays["primitive_index"], dtype=np.int64),
        np.asarray(reference.arrays["cell_translation"], dtype=np.int64),
        np.asarray(reference.info["mlfcs_supercell_matrix"]),
    )
    cells = {}
    for translation in np.asarray(reference.arrays["cell_translation"], dtype=np.int64):
        cells.setdefault(index.residue(translation), translation)
    cell = np.asarray(primitive.cell, dtype=float)
    positions = np.asarray(primitive.get_scaled_positions(wrap=False), dtype=float)
    values = np.zeros((len(reference), len(reference), 3, 3))
    for translation in np.asarray(list(cells.values()), dtype=np.int64):
        for site_a, site_b, shift in periodic_bonds(primitive, cutoff):
            target = translation + np.asarray(shift, dtype=np.int64)
            atom_a = index.atom(site_a, translation)
            atom_b = index.atom(site_b, target)
            vector = (positions[site_b] + target - positions[site_a] - translation) @ cell
            direction = vector / np.linalg.norm(vector)
            projection = np.outer(direction, direction)
            harmonic = spring * projection + bend * (np.eye(3) - projection)
            values[atom_a, atom_a] += harmonic
            values[atom_b, atom_b] += harmonic
            values[atom_a, atom_b] -= harmonic
            values[atom_b, atom_a] -= harmonic
    return values


def _classical_supercell_covariance(
    primitive: Atoms, reference: Atoms, values: np.ndarray, temperature: float
) -> np.ndarray:
    """Return the exact classical covariance ``kT (M^-1/2 Phi M^-1/2)^+ / M^1/2``."""
    primitive_index = np.asarray(reference.arrays["primitive_index"], dtype=np.int64)
    masses = np.asarray(primitive.get_masses(), dtype=float)[primitive_index]
    n_atoms = len(reference)
    mass = np.sqrt(masses[:, None] * masses[None, :])
    weighted = values / mass[..., None, None]
    matrix = (
        ((weighted + weighted.transpose(1, 0, 3, 2)) / 2)
        .transpose(0, 2, 1, 3)
        .reshape(3 * n_atoms, 3 * n_atoms)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # The three uniform translations are zero modes; the sampler projects them out at
    # Gamma, so the pseudo-inverse must drop them as well.
    keep = np.abs(eigenvalues) > 1e-8
    covariance = (
        eigenvectors[:, keep] * (units.kB * temperature / eigenvalues[keep])[None, :]
    ) @ eigenvectors[:, keep].T
    return covariance / np.kron(mass, np.ones((3, 3)))


def _sampler_bloch_covariance(sampler: HarmonicSampler) -> np.ndarray:
    """Return the full-grid covariance implied by the sampler's member records.

    The kernel is built only from the eigenvalues, the star-expanded eigenvectors,
    ``mode_sigma`` and the positional phases of the sampler's own grid, so it is an
    independent statement of what the sampler must produce, not a copy of its random draw.

    Every full q point contributes exactly once.  The sampler draws a single complex
    amplitude for a q/-q pair and reads the conjugate partner inside the real part of that
    one field term, so the pair's two members are both accounted for by the two equal
    per-label contributions this sum adds for them; a label with ``q = -q + G`` is its own
    partner and contributes once, with real degrees of freedom.
    """
    n_cells = sampler._n_cells
    n_primitive = sampler._n_primitive
    n_atoms = len(sampler.supercell)
    masses = sampler._masses
    positions = sampler._positions
    covariance = np.zeros((3 * n_atoms, 3 * n_atoms))
    for record in sampler._members:
        sigma2 = np.where(record.included, sampler._mode_sigma(record.eigenvalues), 0.0) ** 2
        weighted = (record.eigenvectors * sigma2[None, :]) @ record.eigenvectors.conj().T
        phase = np.exp(
            2j
            * np.pi
            * np.einsum(
                "cad,d->ca",
                sampler._cell_translations[:, None, :] + positions[None, :, :],
                record.qpoint,
            )
        )
        for first in range(n_primitive):
            for second in range(n_primitive):
                block = weighted[3 * first : 3 * first + 3, 3 * second : 3 * second + 3] / np.sqrt(
                    masses[first] * masses[second]
                )
                for cell in range(n_cells):
                    for other in range(n_cells):
                        atom = sampler._cell_atoms[cell, first]
                        partner = sampler._cell_atoms[other, second]
                        covariance[3 * atom : 3 * atom + 3, 3 * partner : 3 * partner + 3] += (
                            1.0
                            / n_cells
                            * np.real(block * phase[cell, first] * np.conj(phase[other, second]))
                        )
    return covariance


# ``(model, supercell matrix, bond cutoff, model parameters, interpolation multiplier)``.
_CASES = {
    "cubic_2x1x1": (cubic_argon, (2, 1, 1), 4.5, {"spring": 1.0, "bend": 0.5, "quartic": 1.0}, 1),
    "cubic_3x1x1": (cubic_argon, (3, 1, 1), 4.5, {"spring": 1.0, "bend": 0.5, "quartic": 1.0}, 1),
    "cubic_nondiagonal": (
        cubic_argon,
        [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
        4.5,
        {"spring": 1.0, "bend": 0.5, "quartic": 1.0},
        1,
    ),
    "cubic_multiplier": (
        cubic_argon,
        (1, 1, 1),
        4.5,
        {"spring": 1.0, "bend": 0.5, "quartic": 1.0},
        2,
    ),
    "fcc_2x1x1": (fcc_argon, (2, 1, 1), 3.0, {"spring": 1.0, "bend": 0.5, "quartic": 1.0}, 1),
    "hcp_2x1x1": (hcp_magnesium, (2, 1, 1), 3.25, {"spring": 1.0, "bend": 0.5, "quartic": 0.2}, 1),
    "rhombohedral_2x1x1": (
        rhombohedral_silicon,
        (2, 1, 1),
        4.5,
        {"spring": 1.0, "bend": 0.5, "quartic": 1.0},
        1,
    ),
    "diamond_2x1x1": (
        diamond_silicon,
        (2, 1, 1),
        2.4,
        {"spring": 4.0, "bend": 2.0, "quartic": 2.0},
        1,
    ),
    "diamond_nondiagonal": (
        diamond_silicon,
        [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
        2.4,
        {"spring": 4.0, "bend": 2.0, "quartic": 2.0},
        1,
    ),
}

_CASE_NAMES = sorted(_CASES)


def _force_constants(name: str):
    model, matrix, cutoff, parameters, _ = _CASES[name]
    return pair_bond_force_constants(model(), matrix, cutoff=cutoff, **parameters)


def scph_case(name: str) -> tuple[ForceConstants, ForceConstants]:
    """Return the ``(fc2, fc4)`` pair of one pinned case, for other test modules."""
    return _force_constants(name)


def _multiplier(name: str) -> int:
    return _CASES[name][4]


def _sampler(name: str, *, statistics: str, temperature: float, **options) -> HarmonicSampler:
    fc2, _ = _force_constants(name)
    return HarmonicSampler(
        fc2.relation.primitive,
        fc2.relation.reference,
        fc2.materialize(2),
        temperature=temperature,
        statistics=statistics,
        **options,
    )


def _solver(name: str, *, mixing: float, max_iterations: int, **options) -> LoopSCPH:
    fc2, fc4 = _force_constants(name)
    return LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        statistics="classical",
        mixing=mixing,
        frequency_cutoff_thz=FREQUENCY_CUTOFF_THZ,
        tolerance=SCPH_TOLERANCE,
        max_iterations=max_iterations,
        **options,
    )


_FREQUENCIES = {
    "cubic_2x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [0.0, 2.059045337765843e-08, 2.9119298422095457e-08],
            [3.4979875442582604, 3.4979875442582604, 4.9469014261021895],
        ],
    },
    "cubic_3x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        "denominator": 3,
        "qpoints": [
            [0.0, 0.0, 0.0],
            [0.3333333333333333, 0.0, 0.0],
            [0.6666666666666666, 0.0, 0.0],
        ],
        "frequencies": [
            [0.0, 2.059045337765843e-08, 2.9119298422095457e-08],
            [3.029346075449197, 3.029346075449197, 4.284142305021963],
            [3.029346075449198, 3.029346075449198, 4.284142305021964],
        ],
    },
    "cubic_multiplier": {
        "labels": [
            [0, 0, 0],
            [0, 0, 4],
            [0, 4, 0],
            [0, 4, 4],
            [4, 0, 0],
            [4, 0, 4],
            [4, 4, 0],
            [4, 4, 4],
        ],
        "denominator": 8,
        "qpoints": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ],
        "frequencies": [
            [0.0, 2.059045337765843e-08, 2.9119298422095457e-08],
            [3.49798754425826, 3.49798754425826, 4.9469014261021895],
            [3.4979875442582604, 3.4979875442582604, 4.9469014261021895],
            [4.9469014261021895, 6.058692150898394, 6.058692150898394],
            [3.4979875442582604, 3.4979875442582604, 4.9469014261021895],
            [4.9469014261021895, 6.058692150898394, 6.058692150898394],
            [4.9469014261021895, 6.058692150898394, 6.058692150898394],
            [6.995975088516521, 6.995975088516521, 6.995975088516521],
        ],
    },
    "cubic_nondiagonal": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [0.0, 2.059045337765843e-08, 2.9119298422095457e-08],
            [3.4979875442582604, 3.4979875442582604, 4.9469014261021895],
        ],
    },
    "diamond_2x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [0.0, 0.0, 0.0, 13.625201290243298, 13.625201290243298, 13.625201290243298],
            [
                5.899885224513601,
                5.899885224513604,
                8.34369770095177,
                10.771667413858442,
                12.281590472622577,
                12.281590472622579,
            ],
        ],
    },
    "diamond_nondiagonal": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [0.0, 0.0, 0.0, 13.625201290243298, 13.625201290243298, 13.625201290243298],
            [
                5.899885224513601,
                5.899885224513604,
                8.34369770095177,
                10.771667413858442,
                12.281590472622577,
                12.281590472622579,
            ],
        ],
    },
    "fcc_2x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [-7.974648302253136e-08, -7.704242197754012e-08, -4.118090675531686e-08],
            [6.544135466908579, 6.544135466908579, 7.821737933409024],
        ],
    },
    "hcp_2x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [
                -1.6472362702126752e-07,
                -1.426548456039304e-07,
                -8.236181351063372e-08,
                8.393733513444873,
                8.393733513444873,
                10.021164366347643,
            ],
            [
                8.38980735507794,
                8.584683360123838,
                9.86141997878115,
                10.03102286786243,
                10.352360446425044,
                10.676370788215598,
            ],
        ],
    },
    "rhombohedral_2x1x1": {
        "labels": [[0, 0, 0], [1, 0, 0]],
        "denominator": 2,
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "frequencies": [
            [-9.221071989790077e-08, -2.8713273628349102e-08, 0.0],
            [4.171848850475885, 4.171848850475885, 5.8998852245136035],
        ],
    },
}
_SCPH_FREQUENCIES = {
    "cubic_2x1x1": {
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "bare": [
            [0.0, 2.059045337765843e-08, 2.9119298422095457e-08],
            [3.4979875442582604, 3.4979875442582604, 4.9469014261021895],
        ],
        "corrected": [
            [0.1405033064760811, 0.14050330647608866, 0.17204920403940943],
            [3.500808197961815, 3.500808197961815, 4.951885358183237],
        ],
    },
    "hcp_2x1x1": {
        "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "bare": [
            [
                -1.6472362702126752e-07,
                -1.426548456039304e-07,
                -8.236181351063372e-08,
                8.393733513444873,
                8.393733513444873,
                10.021164366347643,
            ],
            [
                8.38980735507794,
                8.584683360123838,
                9.86141997878115,
                10.03102286786243,
                10.352360446425044,
                10.676370788215598,
            ],
        ],
        "corrected": [
            [
                0.08635492647460895,
                0.09984481910130769,
                0.10046643867847485,
                8.394311967534673,
                8.394458235656542,
                10.022107846258221,
            ],
            [
                8.390386181045965,
                8.585438012139509,
                9.862341828003476,
                10.031619340782827,
                10.353135432127717,
                10.677189736972968,
            ],
        ],
    },
}
_SCPH_MULTIPLIER_GRID_LABELS = [
    [0, 0, 0],
    [0, 0, 8],
    [0, 8, 0],
    [0, 8, 8],
    [4, 0, 0],
    [4, 0, 8],
    [4, 8, 0],
    [4, 8, 8],
    [8, 0, 0],
    [8, 0, 8],
    [8, 8, 0],
    [8, 8, 8],
    [12, 0, 0],
    [12, 0, 8],
    [12, 8, 0],
    [12, 8, 8],
]
_COVARIANCES = {
    "diamond_2x1x1": {
        (0, 1, (0, 0, 1)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
        (1, 1, (0, 0, 0)): [
            [0.002059562665331302, 0.000232005047540464, 0.00023200504754046372],
            [0.00023200504754046391, 0.0020595626653313, -0.00023200504754046345],
            [0.00023200504754046367, -0.00023200504754046345, 0.0020595626653313014],
        ],
        (0, 1, (0, 1, 0)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
        (0, 1, (1, 0, 0)): [
            [-0.0012392591043846638, -0.0003065780985356128, -0.00030657809853561267],
            [-0.00030657809853561256, -0.0012392591043846625, 0.0003065780985356123],
            [-0.0003065780985356129, 0.0003065780985356119, -0.001239259104384663],
        ],
        (0, 0, (0, 0, 0)): [
            [0.0020595626653313022, 0.00023200504754046394, 0.000232005047540464],
            [0.00023200504754046394, 0.0020595626653313022, -0.00023200504754046402],
            [0.00023200504754046405, -0.00023200504754046402, 0.002059562665331301],
        ],
        (0, 1, (0, 0, 0)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
    },
    "diamond_nondiagonal": {
        (0, 1, (0, 0, 1)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
        (1, 1, (0, 0, 0)): [
            [0.002059562665331302, 0.000232005047540464, 0.00023200504754046372],
            [0.00023200504754046391, 0.0020595626653313, -0.00023200504754046345],
            [0.00023200504754046367, -0.00023200504754046345, 0.0020595626653313014],
        ],
        (0, 1, (0, 1, 0)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
        (0, 1, (1, 0, 0)): [
            [-0.0012392591043846638, -0.0003065780985356128, -0.00030657809853561267],
            [-0.00030657809853561256, -0.0012392591043846625, 0.0003065780985356123],
            [-0.0003065780985356129, 0.0003065780985356119, -0.001239259104384663],
        ],
        (0, 0, (0, 0, 0)): [
            [0.0020595626653313022, 0.00023200504754046394, 0.000232005047540464],
            [0.00023200504754046394, 0.0020595626653313022, -0.00023200504754046402],
            [0.00023200504754046405, -0.00023200504754046402, 0.002059562665331301],
        ],
        (0, 1, (0, 0, 0)): [
            [0.0006333530650490787, 0.0003065780985356128, 0.0003065780985356127],
            [0.00030657809853561256, 0.0006333530650490774, -0.0003065780985356123],
            [0.0003065780985356129, -0.0003065780985356119, 0.0006333530650490778],
        ],
    },
}
_TRAJECTORY = {
    "history": [
        (1, 0.10742166851251157, 0.012198640747504836),
        (2, 4.830962927850124e-05, 0.012188069586631285),
        (3, 2.411893899503189e-05, 0.012182793292644283),
    ],
    "qpoints": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    "frequencies": [
        [0.14052142022747385, 0.14052142022747685, 0.17207691167819025],
        [3.5008089249955763, 3.5008089249955763, 4.951886962777561],
    ],
}
_SAMPLER = {
    ("classical", 300.0): {
        "n_qpoints": 2,
        "n_irreducible": 2,
        "total_modes": 9,
        "sampled_modes": 9,
        "excluded_modes": 0,
        "imaginary_modes": 0,
        "minimum_frequency_thz": 8.38980735507794,
        "free_energy": 0.04707102114182576,
        "frequencies": [
            [0.0, 0.0, 0.0, 8.393733513444873, 8.393733513444873, 10.021164366347644],
            [
                8.38980735507794,
                8.58468336012384,
                9.86141997878115,
                10.031022867862433,
                10.352360446425044,
                10.676370788215598,
            ],
        ],
    },
    ("quantum", 300.0): {
        "n_qpoints": 2,
        "n_irreducible": 2,
        "total_modes": 9,
        "sampled_modes": 9,
        "excluded_modes": 0,
        "imaginary_modes": 0,
        "minimum_frequency_thz": 8.38980735507794,
        "free_energy": 0.057948227615850015,
        "frequencies": [
            [0.0, 0.0, 0.0, 8.393733513444873, 8.393733513444873, 10.021164366347644],
            [
                8.38980735507794,
                8.58468336012384,
                9.86141997878115,
                10.031022867862433,
                10.352360446425044,
                10.676370788215598,
            ],
        ],
    },
    ("quantum", 0.0): {
        "n_qpoints": 2,
        "n_irreducible": 2,
        "total_modes": 9,
        "sampled_modes": 9,
        "excluded_modes": 0,
        "imaginary_modes": 0,
        "minimum_frequency_thz": 8.38980735507794,
        "free_energy": 0.08757720465328056,
        "frequencies": [
            [0.0, 0.0, 0.0, 8.393733513444873, 8.393733513444873, 10.021164366347644],
            [
                8.38980735507794,
                8.58468336012384,
                9.86141997878115,
                10.031022867862433,
                10.352360446425044,
                10.676370788215598,
            ],
        ],
    },
}
# Regenerated when the Gamma representative started being diagonalized in the internal subspace:
# the deterministic internal basis changes which random number multiplies which mode, so the
# fixed-seed realization moves while its statistics stay the same to within a percent.
_SAMPLER_SAMPLE = {
    "snapshots": 4096,
    "seed": 20240921,
    "maximum_mean": 0.0009157678879362259,
    "second_moment": [
        [0.00226874648354031, 0.0024540754452243, 0.00215390931765951],
        [0.00225878525279043, 0.0023362158611287, 0.00216539163462085],
        [0.00220495278895579, 0.0023915769311813, 0.00213717155208757],
        [0.00219242196922192, 0.00237021422515582, 0.00213122405102878],
    ],
    "maximum_sampled_displacement": 0.24390772248363585,
}


@pytest.mark.parametrize("name", _CASE_NAMES)
def test_q_grid_labels_and_points_are_exact(name):
    multiplier = _multiplier(name)
    force_constants = _force_constants(name)[0]
    grid = reciprocal_quotient_grid(multiplier * force_constants.relation.supercell_matrix)
    expected = _FREQUENCIES[name]
    assert grid.labels.tolist() == expected["labels"]
    assert grid.denominator == expected["denominator"]
    assert len(grid.labels) == grid.denominator
    np.testing.assert_array_equal(grid.points, grid.labels / grid.denominator)
    np.testing.assert_array_equal(grid.points, np.asarray(expected["qpoints"]))


@pytest.mark.parametrize("name", _CASE_NAMES)
def test_harmonic_frequencies_pin_every_qpoint(name):
    multiplier = _multiplier(name)
    force_constants = _force_constants(name)[0]
    expected = _FREQUENCIES[name]
    mesh = harmonic_frequencies(force_constants, multiplier)
    qpoints = mesh.full_qpoints()
    frequencies = mesh.expand_frequencies()

    np.testing.assert_array_equal(qpoints, np.asarray(expected["qpoints"]))
    n_modes = 3 * len(force_constants.relation.primitive)
    assert frequencies.shape == (len(expected["labels"]), n_modes)
    np.testing.assert_array_equal(frequencies, np.sort(frequencies, axis=1))
    np.testing.assert_allclose(
        frequencies, expected["frequencies"], rtol=FREQUENCY_RTOL, atol=FREQUENCY_ATOL
    )


def test_full_grid_contains_gamma_boundary_and_general_pair():
    """Gamma, a boundary point with ``q = -q + G``, and a general ``q/-q`` pair."""
    gamma = (0, 0, 0)

    # 2x1x1: the only non-Gamma label is half a reciprocal lattice vector, so it maps
    # onto its own negative and carries a real standing-wave degree of freedom.
    boundary_grid = reciprocal_quotient_grid((2, 1, 1))
    boundary = [
        tuple(label)
        for label in boundary_grid.labels.tolist()
        if tuple(label) != gamma and boundary_grid.negative_label(label) == tuple(label)
    ]
    assert boundary == [(1, 0, 0)]
    boundary_frequencies = harmonic_frequencies(
        _force_constants("cubic_2x1x1")[0], 1
    ).expand_frequencies()
    # Gamma: the acoustic sum rule leaves three numerical-zero frequencies.
    np.testing.assert_allclose(boundary_frequencies[0], 0.0, atol=FREQUENCY_ATOL)

    # 3x1x1: a general pair of distinct labels related by q -> -q, which time reversal
    # forces onto equal frequencies.
    pair_grid = reciprocal_quotient_grid((3, 1, 1))
    pairs = [
        (index, pair_grid.negative_label(label))
        for index, label in enumerate(pair_grid.labels.tolist())
        if tuple(label) != gamma and pair_grid.negative_label(label) != tuple(label)
    ]
    assert pairs == [(1, (2, 0, 0)), (2, (1, 0, 0))]
    pair_frequencies = harmonic_frequencies(
        _force_constants("cubic_3x1x1")[0], 1
    ).expand_frequencies()
    np.testing.assert_allclose(
        pair_frequencies[1], pair_frequencies[2], rtol=FREQUENCY_RTOL, atol=FREQUENCY_ATOL
    )


@pytest.mark.parametrize("name", ["cubic_2x1x1", "hcp_2x1x1"])
def test_scph_frequencies_are_pinned_for_bare_and_corrected_lattice(name):
    expected = _SCPH_FREQUENCIES[name]
    force_constants = _force_constants(name)[0]
    solver = _solver(name, mixing=0.7, max_iterations=3)

    qpoints, irreducible_bare = solver._irreducible_frequencies(lattice_fc2(force_constants), 1)
    result = solver._run_single(TEMPERATURE, None)
    corrected_qpoints, irreducible_corrected = solver._irreducible_frequencies(
        lattice_fc2(result.force_constants), 1
    )

    # The irreducible arrays are expanded before they are compared with the full-grid
    # numbers this module pinned, so the pins still measure the physics and not the
    # bookkeeping.
    bare = expand_star_values(irreducible_bare, solver._mesh(1))
    corrected = expand_star_values(irreducible_corrected, solver._mesh(1))
    np.testing.assert_array_equal(qpoints, np.asarray(expected["qpoints"])[: len(qpoints)])
    np.testing.assert_array_equal(corrected_qpoints, np.asarray(expected["qpoints"])[: len(qpoints)])
    np.testing.assert_array_equal(solver._qpoints(1), np.asarray(expected["qpoints"]))
    np.testing.assert_allclose(bare, expected["bare"], rtol=FREQUENCY_RTOL, atol=FREQUENCY_ATOL)
    np.testing.assert_allclose(
        corrected, expected["corrected"], rtol=FREQUENCY_RTOL, atol=FREQUENCY_ATOL
    )


def test_scph_multiplier_grid_labels_are_exact():
    force_constants = _force_constants("cubic_2x1x1")[0]
    grid = reciprocal_quotient_grid(2 * force_constants.relation.supercell_matrix)
    assert grid.labels.tolist() == _SCPH_MULTIPLIER_GRID_LABELS
    assert grid.denominator == 16
    solver = _solver("cubic_2x1x1", mixing=1.0, max_iterations=1)
    np.testing.assert_array_equal(solver._qpoints(2), grid.points)


@pytest.mark.parametrize("name", ["diamond_2x1x1", "diamond_nondiagonal"])
def test_scph_covariance_blocks_match_the_full_grid_sum(name):
    force_constants, fc4 = _force_constants(name)
    solver = _solver(name, mixing=1.0, max_iterations=1)
    covariance = solver._covariance(lattice_fc2(force_constants), 1, TEMPERATURE)

    expected = _COVARIANCES[name]
    assert set(covariance) == set(expected)
    assert set(covariance) == _needed_covariances(fc4.sparse[4])
    for key, block in expected.items():
        np.testing.assert_allclose(covariance[key].imag, 0.0, rtol=0.0, atol=COVARIANCE_ATOL)
        np.testing.assert_allclose(
            covariance[key].real, block, rtol=COVARIANCE_RTOL, atol=COVARIANCE_ATOL
        )


def test_scph_covariance_is_independent_of_the_supercell_shape():
    diagonal = _solver("diamond_2x1x1", mixing=1.0, max_iterations=1)._covariance(
        lattice_fc2(_force_constants("diamond_2x1x1")[0]), 1, TEMPERATURE
    )
    nondiagonal = _solver("diamond_nondiagonal", mixing=1.0, max_iterations=1)._covariance(
        lattice_fc2(_force_constants("diamond_nondiagonal")[0]), 1, TEMPERATURE
    )
    assert set(diagonal) == set(nondiagonal)
    for key in sorted(diagonal):
        np.testing.assert_allclose(
            nondiagonal[key], diagonal[key], rtol=COVARIANCE_RTOL, atol=COVARIANCE_ATOL
        )


def test_scph_single_trajectory_is_pinned():
    solver = _solver("cubic_2x1x1", mixing=0.5, max_iterations=3)
    result = solver._run_single(TEMPERATURE, None)

    assert [iteration.index for iteration in result.history] == [
        row[0] for row in _TRAJECTORY["history"]
    ]
    np.testing.assert_allclose(
        [iteration.frequency_change_thz for iteration in result.history],
        [row[1] for row in _TRAJECTORY["history"]],
        rtol=STATISTICS_RTOL,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [iteration.correction_norm for iteration in result.history],
        [row[2] for row in _TRAJECTORY["history"]],
        rtol=STATISTICS_RTOL,
        atol=1e-15,
    )
    np.testing.assert_array_equal(result.full_qpoints(), np.asarray(_TRAJECTORY["qpoints"]))
    np.testing.assert_allclose(
        result.expand_frequencies(),
        _TRAJECTORY["frequencies"],
        rtol=FREQUENCY_RTOL,
        atol=FREQUENCY_ATOL,
    )
    assert int(result.weights.sum()) == result.n_qpoints
    assert result.n_irreducible <= result.n_qpoints
    assert result.converged is False


def test_default_cutoff_covariance_is_not_dominated_by_the_gamma_zero_modes():
    """Without an explicit cutoff the covariance stays physical.

    The Gamma translations used to be weighted like modes, which put ``1e12``-sized blocks into
    the covariance of a cell whose physical blocks are ``~1e-3``.  They are now excluded
    structurally, so the default-cutoff covariance and the cutoff-guarded one agree in scale and
    no longer depend on a cutoff to stay finite.
    """
    force_constants, fc4 = _force_constants("hcp_2x1x1")
    solver = LoopSCPH(
        fc2=force_constants,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        statistics="classical",
        mixing=1.0,
        max_iterations=1,
    )
    covariance = solver._covariance(lattice_fc2(force_constants), 1, TEMPERATURE)
    largest = max(float(np.abs(block).max()) for block in covariance.values())
    assert largest < 1e-2, largest
    guarded = LoopSCPH(
        fc2=force_constants,
        fc4=fc4,
        temperature=TEMPERATURE,
        interpolation_multiplier=1,
        scph_multiplier=1,
        statistics="classical",
        mixing=1.0,
        max_iterations=1,
        frequency_cutoff_thz=1.0,
    )._covariance(lattice_fc2(force_constants), 1, TEMPERATURE)
    for key, block in covariance.items():
        np.testing.assert_allclose(block, guarded[key], rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize(
    ("statistics", "temperature"),
    [("classical", 300.0), ("quantum", 300.0), ("quantum", 0.0)],
)
def test_harmonic_sampler_state_and_free_energy_are_pinned(statistics, temperature):
    expected = _SAMPLER[(statistics, temperature)]
    sampler = _sampler("hcp_2x1x1", statistics=statistics, temperature=temperature)
    state = sampler.state

    assert state.n_qpoints == expected["n_qpoints"]
    assert state.n_irreducible == expected["n_irreducible"]
    assert state.total_modes == expected["total_modes"]
    assert state.sampled_modes == expected["sampled_modes"]
    assert state.excluded_modes == expected["excluded_modes"]
    assert state.imaginary_modes == expected["imaginary_modes"]
    assert state.n_qpoints == len(sampler.full_qpoints())
    assert state.n_irreducible == len(sampler.irreducible_qpoints)
    assert int(sampler.weights.sum()) == state.n_qpoints
    np.testing.assert_allclose(
        state.minimum_frequency_thz, expected["minimum_frequency_thz"], rtol=FREQUENCY_RTOL
    )
    np.testing.assert_allclose(
        sampler.harmonic_free_energy(), expected["free_energy"], rtol=STATISTICS_RTOL
    )
    # The expansion carries the representative spectrum onto every member of its star, so
    # the full-grid array is pinned in full-grid order.
    frequencies = sampler.expand_frequencies()
    assert frequencies.shape == (state.n_qpoints, 3 * len(sampler.primitive))
    # The order *within* one star is a gauge: the representative is diagonalized in the internal
    # subspace at Gamma and in the full space elsewhere, so the pinned spectrum is compared as
    # the set each q point carries, not as a particular column order.
    np.testing.assert_allclose(
        np.sort(frequencies, axis=1),
        np.sort(np.asarray(expected["frequencies"], dtype=float), axis=1),
        rtol=FREQUENCY_RTOL,
        atol=FREQUENCY_ATOL,
    )
    np.testing.assert_allclose(
        sampler.irreducible_frequencies,
        frequencies[np.asarray(sampler.grid.representatives)],
        rtol=0.0,
        atol=0.0,
    )


def test_harmonic_sampler_fixed_seed_statistics_are_pinned():
    expected = _SAMPLER_SAMPLE
    sampler = _sampler("hcp_2x1x1", statistics="classical", temperature=300.0)
    samples = sampler.sample(expected["snapshots"], random_seed=expected["seed"])

    second_moment = np.mean(samples**2, axis=0)
    np.testing.assert_allclose(
        second_moment, expected["second_moment"], rtol=STATISTICS_RTOL, atol=1e-15
    )
    mean = samples.mean(axis=0)
    np.testing.assert_allclose(np.abs(mean).max(), expected["maximum_mean"], rtol=STATISTICS_RTOL)
    bound = SAMPLING_STANDARD_ERRORS * np.sqrt(second_moment.max() / expected["snapshots"])
    np.testing.assert_allclose(mean, 0.0, rtol=0.0, atol=bound)
    np.testing.assert_allclose(
        sampler.state.maximum_sampled_displacement,
        expected["maximum_sampled_displacement"],
        rtol=STATISTICS_RTOL,
    )
    assert sampler.state.clipped_atoms == 0


def test_sampler_and_shared_fourier_kernel_agree():
    """The sampler's compact kernel and the exact-lattice kernel are one gauge."""
    for name in ("hcp_2x1x1", "diamond_nondiagonal"):
        force_constants = _force_constants(name)[0]
        primitive = force_constants.relation.primitive
        sampler = _sampler(name, statistics="classical", temperature=TEMPERATURE)
        terms = fourier_terms(lattice_fc2(force_constants), primitive)
        masses = np.asarray(primitive.get_masses(), dtype=float)
        grid = reciprocal_quotient_grid(force_constants.relation.supercell_matrix)
        qpoints = sampler.full_qpoints()
        assert len(qpoints) == len(grid.labels)
        scale = 0.0
        for qpoint in qpoints:
            reference = dynamical_matrix(terms, masses, qpoint)
            scale = max(scale, float(np.abs(reference).max()))
            np.testing.assert_allclose(
                sampler._dynamical_matrix(qpoint), reference, rtol=1e-10, atol=1e-14
            )
        assert scale > 0.1


def test_scph_covariance_matches_the_sampler_kernel():
    """SCPH block assembly and the sampler's mode ensemble give one covariance."""
    force_constants, fc4 = _force_constants("hcp_2x1x1")
    sampler = _sampler("hcp_2x1x1", statistics="classical", temperature=TEMPERATURE)
    solver = _solver("hcp_2x1x1", mixing=1.0, max_iterations=1)
    covariance = solver._covariance(lattice_fc2(force_constants), 1, TEMPERATURE)
    theory = _sampler_bloch_covariance(sampler)
    index = sampler._index

    assert set(covariance) == _needed_covariances(fc4.sparse[4])
    for (first, second, shift), block in covariance.items():
        residue = index.residue(np.asarray(shift, dtype=np.int64))
        cells = sampler._cell_translations
        cell = int(np.flatnonzero((cells == np.asarray(residue)).all(axis=1))[0])
        atom = sampler._cell_atoms[cell, first]
        partner = sampler._cell_atoms[0, second]
        np.testing.assert_allclose(
            block.real,
            theory[3 * atom : 3 * atom + 3, 3 * partner : 3 * partner + 3],
            rtol=COVARIANCE_RTOL,
            atol=COVARIANCE_ATOL,
        )


def test_sampler_full_grid_covariance_matches_theory_and_exact_supercell():
    """Every ``E[u_a u_b]`` block: sample, Bloch theory, and exact Cartesian supercell."""
    force_constants, _ = _force_constants("hcp_2x1x1")
    reference = force_constants.relation.reference
    sampler = _sampler("hcp_2x1x1", statistics="classical", temperature=TEMPERATURE)
    theory = _sampler_bloch_covariance(sampler)
    exact = _classical_supercell_covariance(
        force_constants.relation.primitive,
        reference,
        _supercell_force_constants(
            force_constants.relation.primitive,
            reference,
            cutoff=_CASES["hcp_2x1x1"][2],
            spring=_CASES["hcp_2x1x1"][3]["spring"],
            bend=_CASES["hcp_2x1x1"][3]["bend"],
        ),
        TEMPERATURE,
    )
    np.testing.assert_allclose(theory, exact, rtol=COVARIANCE_RTOL, atol=COVARIANCE_ATOL)

    samples = sampler.sample(SAMPLER_SNAPSHOTS, random_seed=SAMPLER_SEED)
    flat = samples.reshape(len(samples), -1)
    empirical = flat.T @ flat / len(flat)
    bound = SAMPLING_STANDARD_ERRORS * float(np.abs(theory).max()) / np.sqrt(SAMPLER_SNAPSHOTS)
    assert bound < 0.1 * float(np.abs(theory).max())
    np.testing.assert_allclose(empirical, theory, rtol=0.0, atol=bound)
