"""Quartic loop self-consistent phonons.

This module deliberately implements only the static quartic loop diagram.  The
result is a temperature-dependent real-space FC2 that can be handed to the
existing IO backends; it is not a frequency-dependent bubble self-energy.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from mlfcs.force_constants.dense import lattice_fc2, replace_lattice_fc2
from mlfcs.force_constants.representation import (
    ForceConstants,
)
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.grid import (
    IrreducibleReciprocalGrid,
    irreducible_reciprocal_grid,
)
from mlfcs.reciprocal.kernels import dynamical_matrices_compiled
from mlfcs.reciprocal.plan import FourierPlan, ReciprocalExpansionPlan
from mlfcs.reciprocal.scph.fourier import (
    _multiplier,
    _needed_covariances,
    _validate_relation,
)
from mlfcs.reciprocal.statistics import HBAR_ASE, OMEGA_TO_THZ, mode_sigma
from mlfcs.reciprocal.symmetry import (
    expand_star_values,
    require_hermitian,
    require_star_covariance,
    validate_site_masses,
    validate_symmetry_tolerance,
    validate_symprec,
)
from mlfcs.reciprocal.temperature import TemperatureSeriesResult, normalize_temperature_schedule
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

_HBAR_ASE = HBAR_ASE
_OMEGA_TO_THZ = OMEGA_TO_THZ
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoopSCPHIteration:
    index: int
    frequency_change_thz: float
    correction_norm: float


@dataclass(slots=True)
class LoopSCPHResult:
    """One converged SCPH temperature on the irreducible wedge of its q grid.

    The frequencies are stored per star representative.  ``expand_frequencies`` returns
    the full mesh exactly, without another diagonalization, because a star carries the
    eigenvalues of its representative.
    """

    temperature: float
    irreducible_qpoints: np.ndarray
    irreducible_frequencies: np.ndarray
    weights: np.ndarray
    grid: IrreducibleReciprocalGrid
    symprec: float
    symmetry_tolerance: float | None
    time_reversal: bool
    force_constants: ForceConstants
    history: tuple[LoopSCPHIteration, ...]
    converged: bool

    @property
    def iterations(self) -> int:
        return len(self.history)

    @property
    def n_qpoints(self) -> int:
        """Number of q points of the full grid."""
        return len(self.grid.full.labels)

    @property
    def n_irreducible(self) -> int:
        """Number of irreducible representative q points."""
        return len(self.grid.representatives)

    @property
    def reduction_ratio(self) -> float:
        """How many full q points one representative stands for."""
        return self.n_qpoints / self.n_irreducible

    def full_qpoints(self) -> np.ndarray:
        """Return the q points of the full grid, in full-grid order."""
        return self.grid.full.points

    def expand_frequencies(self) -> np.ndarray:
        """Return the full-grid frequencies, in full-grid order."""
        return expand_star_values(self.irreducible_frequencies, self.grid)


class LoopSCPH:
    """Self-consistent quartic-loop renormalization of FC2.

    ``fc2`` and ``fc4`` are intentionally separate objects.  They may come
    from different calculations, but their primitive/reference structure
    relations must be identical.
    """

    def __init__(
        self,
        *,
        fc2: ForceConstants,
        fc4: ForceConstants,
        temperature: float | Sequence[float],
        interpolation_multiplier: int = 1,
        scph_multiplier: int = 2,
        statistics: str = "quantum",
        mixing: float = 0.1,
        tolerance: float = 1e-10,
        max_iterations: int = 100,
        frequency_cutoff_thz: float = 0.0,
        warm_start: ForceConstants | None = None,
        continuation: bool = True,
        symprec: float = 1e-5,
        time_reversal: bool = True,
        symmetry_tolerance: float | None = 1e-6,
    ) -> None:
        if not isinstance(fc2, ForceConstants) or not isinstance(fc4, ForceConstants):
            raise TypeError("fc2 and fc4 must be ForceConstants objects")
        if 2 not in fc2.orders:
            raise ValueError("fc2 does not contain order-2 force constants")
        if 4 not in fc4.orders:
            raise ValueError("fc4 does not contain order-4 force constants")
        if statistics not in {"quantum", "classical"}:
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if not 0 < mixing <= 1:
            raise ValueError("mixing must be in (0, 1]")
        if tolerance <= 0 or max_iterations < 1:
            raise ValueError("tolerance must be positive and max_iterations >= 1")
        self.fc2 = fc2
        self.fc4 = fc4
        self.temperatures = normalize_temperature_schedule(temperature)
        self.interpolation_multiplier = _multiplier(
            interpolation_multiplier, "interpolation_multiplier"
        )
        self.scph_multiplier = _multiplier(scph_multiplier, "scph_multiplier")
        if self.scph_multiplier % self.interpolation_multiplier:
            raise ValueError("scph_multiplier must be a multiple of interpolation_multiplier")
        self.statistics = statistics
        self.mixing = float(mixing)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.frequency_cutoff_thz = float(frequency_cutoff_thz)
        if self.frequency_cutoff_thz < 0:
            raise ValueError("frequency_cutoff_thz must be non-negative")
        _validate_relation(fc2, fc4)
        if warm_start is not None:
            if not isinstance(warm_start, ForceConstants) or 2 not in warm_start.orders:
                raise TypeError("warm_start must be a ForceConstants object containing FC2")
            _validate_relation(fc2, warm_start)
        self.warm_start = warm_start
        self.continuation = bool(continuation)
        self._primitive = fc2.relation.primitive if fc2.relation is not None else None
        if self._primitive is None:
            raise ValueError("fc2 must contain an explicit StructureRelation")
        # The geometric tolerance that identifies the primitive symmetry is a different
        # quantity from any dynamical-matrix tolerance, so it is a public argument and is
        # recorded in every result instead of being a module constant.
        self.time_reversal = bool(time_reversal)
        self.symmetry_tolerance = validate_symmetry_tolerance(
            symmetry_tolerance, context="LoopSCPH"
        )
        self.symprec = validate_symprec(symprec, context="LoopSCPH")
        self._symmetry = PrimitiveSymmetryOperations.from_atoms(
            self._primitive, symprec=self.symprec
        )
        validate_site_masses(
            self._symmetry,
            np.asarray(self._primitive.get_masses(), dtype=float),
            context="LoopSCPH",
        )
        self._meshes: dict[int, IrreducibleReciprocalGrid] = {}
        # One plan per grid: the integer inverse, the permutation and the gauge of every member
        # are computed once and shared by the validation gate and the covariance.
        self._plans: dict[int, ReciprocalExpansionPlan] = {}
        # Content-addressed certificates: a lattice mapping that has already been validated
        # for a multiplier is never validated twice, and an object identity is never used as
        # the key, because the same content has to be reusable across temperatures.
        self._certificates: set[tuple[bytes, int]] = set()

    def run(self) -> LoopSCPHResult | TemperatureSeriesResult[LoopSCPHResult]:
        """Run one temperature or an ascending temperature schedule."""
        if len(self.temperatures) == 1:
            return self._run_single(self.temperatures[0], self.warm_start)
        previous = self.warm_start
        results: list[LoopSCPHResult] = []
        for temperature in self.temperatures:
            result = self._run_single(temperature, previous)
            results.append(result)
            if self.continuation:
                previous = result.force_constants
            else:
                previous = self.warm_start
        return TemperatureSeriesResult(self.temperatures, tuple(results), self.continuation)

    def _run_single(self, temperature: float, warm_start: ForceConstants | None) -> LoopSCPHResult:
        base = self._copy_order(self.fc2, 2)
        bare = lattice_fc2(base)
        current = (
            {key: value.copy() for key, value in bare.items()}
            if warm_start is None
            else lattice_fc2(warm_start)
        )
        history: list[LoopSCPHIteration] = []
        multipliers = tuple(
            dict.fromkeys((self.scph_multiplier, self.interpolation_multiplier))
        )
        for multiplier in multipliers:
            self._require_covariant(
                current,
                multiplier,
                stage="initial",
                iteration=None,
                temperature=temperature,
            )
        mesh = self._mesh(self.interpolation_multiplier)
        diagonalizations = 0
        qpoints, previous_frequencies = self._irreducible_frequencies(
            current, self.interpolation_multiplier
        )
        diagonalizations += mesh.n_irreducible
        converged = False
        previous_covariance: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] | None = None
        for iteration in range(1, self.max_iterations + 1):
            covariance = self._covariance(current, self.scph_multiplier, temperature)
            if previous_covariance is not None:
                covariance = {
                    key: self.mixing * value + (1.0 - self.mixing) * previous_covariance[key]
                    for key, value in covariance.items()
                }
            correction = self._loop_correction(covariance)
            correction_norm = float(
                np.sqrt(sum(np.vdot(value, value).real for value in correction.values()))
            )
            keys = bare.keys() | correction.keys()
            updated = {key: bare.get(key, 0.0) + correction.get(key, 0.0) for key in keys}
            # The updated iterate is what the next iteration expands and what the run finally
            # returns, so it has to pass the gate before anything else touches it.
            for multiplier in multipliers:
                self._require_covariant(
                    updated,
                    multiplier,
                    stage="updated",
                    iteration=iteration,
                    temperature=temperature,
                )
            last_qpoints, frequencies = self._irreducible_frequencies(
                updated, self.interpolation_multiplier
            )
            diagonalizations += mesh.n_irreducible
            frequency_change = self._frequency_change(frequencies, previous_frequencies, mesh)
            history.append(LoopSCPHIteration(iteration, frequency_change, correction_norm))
            logger.info(
                f"SCPH iteration {iteration}: delta_omega={frequency_change:.6e} THz, "
                f"frequency_min={np.min(frequencies):.6e} THz, "
                f"frequency_max={np.max(frequencies):.6e} THz, "
                f"correction_norm={correction_norm:.6e}, "
                f"q_points={mesh.n_qpoints}, irreducible={mesh.n_irreducible}, "
                f"reduction={mesh.reduction_ratio:.2f}, diagonalizations={diagonalizations}",
            )
            current = updated
            qpoints, previous_frequencies = last_qpoints, frequencies
            previous_covariance = covariance
            # Convergence is a fixed-point criterion.  An imaginary mode is a
            # physical diagnostic of the current solution, not an additional
            # stopping condition.
            if frequency_change < self.tolerance:
                converged = True
                break

        if not converged:
            logger.warning(
                "SCPH reached %d iterations without meeting tolerance %.3e THz; "
                "returning the final iterate",
                self.max_iterations,
                self.tolerance,
            )
        effective = replace_lattice_fc2(
            base,
            current,
            metadata={
                "method": "loop_scph",
                "temperature": temperature,
                "symprec": self.symprec,
                "symmetry_tolerance": self.symmetry_tolerance,
                "time_reversal": self.time_reversal,
            },
        )
        # The returned iterate is the last updated one; re-check that its certificate still
        # matches its content, so a result can never carry an unvalidated force-constant set.
        for multiplier in multipliers:
            self._require_covariant(
                current,
                multiplier,
                stage="returned",
                iteration=None,
                temperature=temperature,
            )
        # The final iterate's frequencies were already computed by the last sweep, so the
        # result reuses them instead of paying another ``N_irr`` diagonalizations.
        return LoopSCPHResult(
            temperature=temperature,
            irreducible_qpoints=qpoints,
            irreducible_frequencies=frequencies,
            weights=np.asarray(mesh.weights, dtype=np.int64),
            grid=mesh,
            symprec=self.symprec,
            symmetry_tolerance=self.symmetry_tolerance,
            time_reversal=self.time_reversal,
            force_constants=effective,
            history=tuple(history),
            converged=converged,
        )

    def _covariance(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
        temperature: float,
    ) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
        r"""Return the real-space displacement covariance of one loop sweep.

        The covariance is built on the irreducible wedge and expanded per star member:

        1. diagonalize ``D(q_s)`` at every star representative only;
        2. form the basis-independent ``W(q_s) = V_s diag(sigma_s^2) V_s^dagger``;
        3. carry it onto each member with ``U_g`` and, on an antiunitary member, a complex
           conjugation first: ``W(gq_s) = U_g W(q_s) U_g^dagger``;
        4. take the atom blocks the quartic force constants actually need and multiply them
           by the exact Fourier phase of that member;
        5. sum over the whole star and divide by the number of full q points.

        Step 5 is a genuine sum over every full q point: a star weight never multiplies a
        representative's matrix, because the members are related by a rotation and not by a
        scalar.  What the reduction removes is the repeated diagonalization, not the
        physical summation.
        """
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = fourier_terms(lattice, relation.primitive)
        primitive_positions = relation.primitive.get_scaled_positions(wrap=False)
        grid = self._mesh(multiplier)
        n_q = grid.n_qpoints
        needed = sorted(_needed_covariances(self.fc4.sparse[4]))
        representatives = np.asarray(grid.representatives, dtype=np.int64)
        members = tuple(star.members for star in grid.stars)
        # One composite action per member: the operation, the antiunitary flag and the
        # positional gauge that brings the result onto the label the grid stores.
        plan = self._plan(multiplier)
        def covariance_of_stars(star_chunk: np.ndarray) -> dict[tuple, np.ndarray]:
            result: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
            points = grid.full.points[representatives[star_chunk]]
            eigenvalues, vectors = np.linalg.eigh(dynamical_matrices(terms, masses, points))
            sigma2 = (
                mode_sigma(
                    eigenvalues,
                    temperature=temperature,
                    statistics=self.statistics,
                    cutoff_frequency_thz=self.frequency_cutoff_thz,
                )
                ** 2
            )
            weighted = (vectors * sigma2[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
            for local, star in enumerate(star_chunk.tolist()):
                for member in members[star].tolist():
                    expanded = plan.apply_to_matrix(int(member), weighted[local])
                    require_hermitian(
                        expanded,
                        scale=float(np.max(np.abs(weighted[local]))),
                        tolerance=self.symmetry_tolerance,
                        label=tuple(int(value) for value in grid.full.labels[member]),
                        operation=int(grid.full_operations[member]),
                        context="LoopSCPH covariance",
                    )
                    qpoint = grid.full.points[member]
                    for a, b, r in needed:
                        block = expanded[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] / np.sqrt(
                            masses[a] * masses[b]
                        )
                        displacement = (
                            primitive_positions[a] - primitive_positions[b] + np.asarray(r)
                        )
                        phase = np.exp(2j * np.pi * float(qpoint @ displacement))
                        key = (a, b, r)
                        result[key] = result.get(key, 0.0) + block * phase / n_q
            return result

        # One pass over the stars: a thread pool over them measured slower than the serial
        # loop (the per-star work is numpy under the GIL, so the threads only queue and add
        # overhead), and the plan forbids keeping a knob whose only effect is a regression.
        return covariance_of_stars(np.arange(len(grid.stars), dtype=np.int64))

    def _loop_correction(
        self, covariance: dict[tuple[int, int, tuple[int, int, int]], np.ndarray]
    ) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
        result: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
        for sites, translations, tensor in zip(
            self.fc4.sparse[4].sites,
            self.fc4.sparse[4].translations,
            self.fc4.sparse[4].tensors,
            strict=True,
        ):
            s1, s2, s3, s4 = map(int, sites)
            r2, r3, r4 = (tuple(map(int, row)) for row in translations)
            cov = covariance.get((s3, s4, tuple(np.asarray(r3) - np.asarray(r4))))
            if cov is None:
                continue
            value = 0.5 * np.einsum("abcd,cd->ab", tensor, cov, optimize=True)
            key = (s1, s2, r2)
            result[key] = result.get(key, 0.0) + value.real
        return result

    def _irreducible_frequencies(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the frequencies of the star representatives of one grid.

        The dynamical matrices are built and diagonalized once per representative; a star
        member never reaches an eigensolver, because its spectrum is its representative's.
        """
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = fourier_terms(lattice, relation.primitive)
        grid = self._mesh(multiplier)
        qpoints = grid.full.points[grid.representatives]
        plan = FourierPlan.from_terms(terms, masses)
        eigenvalues = np.linalg.eigvalsh(dynamical_matrices_compiled(plan, qpoints))
        values = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * _OMEGA_TO_THZ
        return np.asarray(qpoints), np.asarray(values)

    def _frequency_change(
        self,
        frequencies: np.ndarray,
        previous: np.ndarray,
        grid: IrreducibleReciprocalGrid,
    ) -> float:
        r"""Return the star-weighted RMS frequency change over the full grid.

        .. math::

            \Delta\omega = \sqrt{\frac{1}{N_q N_b}\sum_s w_s
                \lVert\omega_s^{(n)} - \omega_s^{(n-1)}\rVert_2^2},

        which is the full-grid RMS of the expanded change: every member of a star carries
        the same change as its representative, so summing with the star weights is the
        full sum without materializing it.
        """
        delta = np.asarray(frequencies, dtype=float) - np.asarray(previous, dtype=float)
        n_q = len(grid.full.labels)
        n_b = delta.shape[1]
        weights = np.asarray(grid.weights, dtype=float)
        return float(np.sqrt(np.sum(weights[:, None] * delta**2) / (n_q * n_b)))

    def _qpoints(self, multiplier: int) -> np.ndarray:
        """Return the full-grid q points of one multiplier, in full-grid order."""
        return self._mesh(multiplier).full.points

    def _mesh(self, multiplier: int) -> IrreducibleReciprocalGrid:
        """Return the cached star decomposition of the grid of one multiplier."""
        if multiplier not in self._meshes:
            relation = self.fc2.relation
            assert relation is not None
            self._meshes[multiplier] = irreducible_reciprocal_grid(
                multiplier * relation.supercell_matrix,
                self._symmetry,
                time_reversal=self.time_reversal,
            )
        return self._meshes[multiplier]

    def _plan(self, multiplier: int) -> ReciprocalExpansionPlan:
        """Return the cached expansion plan of one grid."""
        if multiplier not in self._plans:
            relation = self.fc2.relation
            assert relation is not None
            self._plans[multiplier] = ReciprocalExpansionPlan.from_grid(
                self._symmetry,
                self._mesh(multiplier),
                np.asarray(relation.primitive.get_scaled_positions(wrap=False), dtype=float),
            )
        return self._plans[multiplier]

    def _lattice_fingerprint(
        self, lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray]
    ) -> bytes:
        """Return a content digest of one lattice mapping."""
        digest = hashlib.blake2b(digest_size=16)
        for key in sorted(lattice):
            digest.update(repr(key).encode("ascii"))
            digest.update(np.ascontiguousarray(lattice[key], dtype=float).tobytes())
        return digest.digest()

    def _require_covariant(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
        *,
        stage: str,
        iteration: int | None,
        temperature: float,
    ) -> None:
        """Validate one FC2 iterate on one grid before it is expanded, or reuse its certificate.

        Every iterate that the irreducible path expands has to pass the full-star gate first:
        the initial FC2, the updated FC2 of each iteration, and the FC2 that is finally
        returned.  The certificate is content-addressed, so the same iterate is never
        validated twice within one run.
        """
        if self.symmetry_tolerance is None:
            return
        cache_key = (self._lattice_fingerprint(lattice), multiplier)
        if cache_key in self._certificates:
            return
        where = (
            f"{stage} FC2"
            if iteration is None
            else f"{stage} FC2 at iteration {iteration}"
        )
        self._check_symmetry(
            lattice,
            multiplier,
            f"LoopSCPH {where} ({temperature} K, multiplier {multiplier})",
        )
        self._certificates.add(cache_key)

    def _check_symmetry(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
        context: str,
    ) -> None:
        """Raise unless the given force constants are covariant on every star member.

        Every consumer of a star decomposition relies on that covariance, so it is checked on
        the lattice that is actually being used, and the failure names the representative, the
        member, the operation and the residual.
        """
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = fourier_terms(lattice, relation.primitive)
        require_star_covariance(
            partial(dynamical_matrices_compiled, FourierPlan.from_terms(terms, masses)),
            self._symmetry,
            self._mesh(multiplier),
            np.asarray(relation.primitive.get_scaled_positions(wrap=False), dtype=float),
            tolerance=self.symmetry_tolerance,
            context=context,
            plan=self._plan(multiplier),
        )

    @staticmethod
    def _copy_order(source: ForceConstants, order: int) -> ForceConstants:
        reference = source.relation.reference if source.relation is not None else source.supercell
        return ForceConstants(
            {},
            reference.copy(),
            dict(source.metadata),
            {order: source.sparse[order]},
            source.relation,
        )
