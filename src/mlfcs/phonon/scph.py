"""Quartic-loop self-consistent phonons on symmetry-reduced reciprocal grids."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numba import njit
from scipy.constants import Boltzmann, angstrom, atomic_mass, electron_volt, hbar

from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.expansion import expand_lattice_tensors
from mlfcs.foundation.log import get_logger
from mlfcs.phonon.dynamics import (
    THZ_PER_SQRT_EV_PER_A2_AMU,
    accumulate_dynamical_matrices,
    prepare_dynamical_terms,
    prepare_fc2_bases,
    translation_complement,
)
from mlfcs.phonon.grid import QStars, as_qgrid, mass_preserving_symmetry
from mlfcs.phonon.stars import StarPlan

_OMEGA = np.sqrt(electron_volt / (angstrom**2 * atomic_mass))
_VARIANCE = hbar / (2.0 * atomic_mass * angstrom**2 * _OMEGA)
logger = get_logger(__name__)


@njit(cache=True, nogil=True)
def _add_covariance(
    result: np.ndarray,
    matrix: np.ndarray,
    qpoint: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    distances: np.ndarray,
    mass_weights: np.ndarray,
    n_qpoints: int,
) -> None:
    """Accumulate one reciprocal-space covariance contribution into pair blocks.

    Fourier transform the mass-weighted displacement covariance at one q point
    using the positional-gauge phase, average over the full grid, and restore
    the physical displacement normalization for each atomic pair.
    """
    for key in range(len(first)):
        angle = 0.0
        for axis in range(3):
            angle += qpoint[axis] * distances[key, axis]
        angle *= 2.0 * np.pi
        phase = complex(np.cos(angle), np.sin(angle)) * mass_weights[key] / n_qpoints
        row = 3 * first[key]
        column = 3 * second[key]
        for alpha in range(3):
            for beta in range(3):
                result[key, alpha, beta] += matrix[row + alpha, column + beta] * phase


@njit(cache=True, nogil=True)
def _contract(
    tensors: np.ndarray,
    covariance: np.ndarray,
    covariance_index: np.ndarray,
    orbit_index: np.ndarray,
    n_orbits: int,
) -> np.ndarray:
    """Contract FC4 with displacement covariance to obtain a quartic-loop FC2 correction.

    The correction is ``Delta Phi2_ab = 1/2 * sum_cd Phi4_abcd <u_c u_d>``.
    Pair terms outside the retained FC2 cluster space are omitted.
    """
    correction = np.zeros((n_orbits, 3, 3), dtype=np.float64)
    for term in range(len(tensors)):
        orbit = orbit_index[term]
        if orbit < 0:
            continue
        key = covariance_index[term]
        for alpha in range(3):
            for beta in range(3):
                value = 0.0j
                for gamma in range(3):
                    for delta in range(3):
                        value += (
                            tensors[term, alpha, beta, gamma, delta] * covariance[key, gamma, delta]
                        )
                correction[orbit, alpha, beta] += 0.5 * value.real
    return correction


@njit(cache=True, nogil=True)
def _fc2_tensors(
    parameters: np.ndarray,
    bases: np.ndarray,
    offsets: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    """Expand an FC2 parameter vector into Cartesian tensors for all Fourier terms."""
    result = np.empty((len(offsets), 3, 3), dtype=np.float64)
    for term in range(len(offsets)):
        for component in range(9):
            value = 0.0
            for parameter in range(dimensions[term]):
                value += bases[term, component, parameter] * parameters[offsets[term] + parameter]
            result[term, component // 3, component % 3] = value
    return result


def _modal_covariance(
    matrix: np.ndarray,
    masses: np.ndarray,
    temperature: float,
    *,
    gamma: bool,
    statistics: Literal["quantum", "classical"],
) -> np.ndarray:
    """Return the harmonic displacement covariance of one dynamical matrix.

    The matrix is diagonalized in mass-weighted coordinates. A mode of angular
    frequency ``omega`` has quantum variance

        <Q**2> = hbar/(2*omega) * coth(hbar*omega/(2*k_B*T)),

    including zero-point motion. Classical statistics use the equipartition
    variance ``k_B*T/omega**2``. At Gamma, rigid translations are projected
    out before diagonalization. Negative curvatures use their absolute values
    to form a trial covariance; this does not classify the structure as stable.
    A non-translational zero mode has divergent covariance and raises ValueError.
    """
    if gamma:
        internal = translation_complement(masses)
        if internal.shape[1] == 0:
            return np.zeros_like(matrix)
        eigenvalues, vectors = np.linalg.eigh(internal.T @ matrix @ internal)
        vectors = internal @ vectors
    else:
        eigenvalues, vectors = np.linalg.eigh(matrix)
    # A negative bare curvature is a physical instability, not a tolerance
    # violation.  Its magnitude supplies a positive trial covariance; the
    # signed frequencies remain available for diagnosing the iterate.
    curvatures = np.abs(eigenvalues)
    if np.any(curvatures == 0.0):
        raise ValueError("a non-translational zero mode has divergent harmonic covariance")
    if statistics == "classical":
        variances = Boltzmann * temperature / (atomic_mass * angstrom**2 * _OMEGA**2 * curvatures)
    else:
        omega = _OMEGA * np.sqrt(curvatures)
        variances = _VARIANCE / np.sqrt(curvatures)
        if temperature > 0.0:
            variances /= np.tanh(hbar * omega / (2.0 * Boltzmann * temperature))
    if not np.all(np.isfinite(variances)):
        raise ValueError("modal covariance is not finite")
    return (vectors * variances) @ vectors.conj().T


@dataclass(frozen=True, slots=True)
class SCPHStep:
    """Convergence diagnostics recorded for one SCPH iteration.

    ``frequency_change_thz`` is the star-weighted RMS change in signed
    frequencies. ``minimum_frequency_thz`` is the smallest signed frequency
    on the mesh, including Gamma translations. Both are in THz.
    """

    index: int
    frequency_change_thz: float
    minimum_frequency_thz: float


@dataclass(frozen=True, slots=True)
class SCPHResult:
    """Result of an SCPH calculation at one temperature.

    ``fc2`` is the final effective second-order model. ``frequencies`` contains
    signed THz frequencies on the irreducible mesh, and ``history`` records the
    iteration diagnostics. Convergence is based only on frequency change; a
    converged result may still have physically imaginary modes.
    """

    temperature: float
    fc2: ForceConstants
    frequencies: np.ndarray
    stars: QStars
    history: tuple[SCPHStep, ...]
    converged: bool
    minimum_mode_thz: float | None

    @property
    def iterations(self) -> int:
        """Number of recorded self-consistency iterations."""
        return len(self.history)

    @property
    def has_imaginary_modes(self) -> bool:
        """Whether the final mesh has a negative non-translational curvature.

        This is a physical diagnostic, not part of the iteration stopping rule.
        """
        return self.minimum_mode_thz is not None and self.minimum_mode_thz < 0.0


class SCPH:
    """Static quartic-loop self-consistent phonon model.

    The effective FC2 is updated according to

        Phi2_eff = Phi2_bare + 1/2 * Phi4 : <u u>,

    where the displacement covariance ``<u u>`` is evaluated from the current
    effective harmonic dynamical matrix using quantum or classical statistics.
    The self-consistency cycle uses a symmetry-reduced reciprocal mesh. Matrix-
    valued covariances are transformed to full-grid members before the real-
    space quartic contraction.

    When the cluster space has prepared acoustic coordinates, bare FC2,
    initial FC2 and every quartic-loop target are projected onto that subspace
    in the physical parameter Euclidean metric before mixing. The projection
    uses lift and adjoint actions without forming a dense nullspace.

    Parameters
    ----------
    model : ForceConstants
        Primitive model containing FC2 and FC4 on the same cluster space.
        FC3 does not enter this approximation.
    mesh : QGrid or array_like of integers, shape (3,) or (3, 3)
        Positive diagonal mesh sizes or a supercell row matrix/grid.
    statistics : {'quantum', 'classical'}, default 'quantum'
        Modal covariance includes quantum zero-point motion or the classical limit.
    time_reversal : bool, default True
        Include time reversal when building irreducible reciprocal stars.

    """

    def __init__(
        self,
        model: ForceConstants,
        mesh: object,
        *,
        statistics: Literal["quantum", "classical"] = "quantum",
        time_reversal: bool = True,
    ) -> None:
        """Construct the static quartic-loop SCPH problem on the requested mesh."""
        if not isinstance(model, ForceConstants) or not {2, 4} <= set(model.orders):
            raise ValueError("SCPH requires one ForceConstants model containing FC2 and FC4")
        if statistics not in ("quantum", "classical"):
            raise ValueError("statistics must be 'quantum' or 'classical'")
        grid = as_qgrid(mesh)
        stars = QStars(
            grid,
            mass_preserving_symmetry(model.cluster_space.symmetry, model.cluster_space.masses),
            time_reversal=time_reversal,
        )
        self.model = model
        self.stars = stars
        self.plan = StarPlan(stars, model.cluster_space)
        self._points = grid.points
        self.statistics = statistics
        self._acoustic_coordinates = model.cluster_space.acoustic_coordinates(2)
        self._bare = self._constrain_fc2(np.asarray(model.coefficients[2]))
        self._masses = model.cluster_space.masses
        self._dynamical_terms, _ = prepare_dynamical_terms(self._fc2(self._bare))
        self._gamma = next(
            index
            for index, representative in enumerate(stars.representatives)
            if not np.any(grid.labels[representative])
        )
        quartic = expand_lattice_tensors(model, 4)
        self._orbits = model.cluster_space.orbits[model.cluster_space.block(2).orbits]
        self._fc2_bases, self._fc2_offsets, self._fc2_dimensions = prepare_fc2_bases(
            model.cluster_space, self._dynamical_terms
        )
        output = {
            (
                orbit.representative.sites[0].site,
                orbit.representative.sites[1].site,
                orbit.representative.sites[1].translation,
            ): index
            for index, orbit in enumerate(self._orbits)
        }
        covariance_keys = sorted(
            {
                (
                    sites[2],
                    sites[3],
                    tuple(a - b for a, b in zip(shifts[1], shifts[2], strict=True)),
                )
                for sites, shifts in zip(quartic.sites, quartic.translations, strict=True)
            }
        )
        covariance_lookup = {key: index for index, key in enumerate(covariance_keys)}
        self._first = np.asarray([key[0] for key in covariance_keys], dtype=np.int32)
        self._second = np.asarray([key[1] for key in covariance_keys], dtype=np.int32)
        self._distances = np.asarray(
            [
                np.asarray(key[2])
                + model.cluster_space.scaled_positions[key[0]]
                - model.cluster_space.scaled_positions[key[1]]
                for key in covariance_keys
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        self._mass_weights = 1.0 / np.sqrt(self._masses[self._first] * self._masses[self._second])
        self._tensors = np.asarray(quartic.tensors, dtype=np.float64).reshape(-1, 3, 3, 3, 3)
        self._covariance_index = np.asarray(
            [
                covariance_lookup[
                    (
                        sites[2],
                        sites[3],
                        tuple(a - b for a, b in zip(shifts[1], shifts[2], strict=True)),
                    )
                ]
                for sites, shifts in zip(quartic.sites, quartic.translations, strict=True)
            ],
            dtype=np.int32,
        )
        self._orbit_index = np.asarray(
            [
                output.get((sites[0], sites[1], shifts[0]), -1)
                for sites, shifts in zip(quartic.sites, quartic.translations, strict=True)
            ],
            dtype=np.int32,
        )
        logger.info(
            "SCPH prepared: mesh=%s qpoints=%d irreducible=%d FC2 terms=%d "
            "FC4 terms=%d statistics=%s",
            tuple(tuple(int(value) for value in row) for row in grid.matrix),
            grid.size,
            len(stars.representatives),
            len(self._dynamical_terms.first_sites),
            len(self._tensors),
            statistics,
        )

    def _fc2(self, parameters: np.ndarray) -> ForceConstants:
        """Construct an FC2-only model from the current parameter vector."""
        return ForceConstants(self.model.cluster_space, {2: parameters})

    def _constrain_fc2(self, parameters: np.ndarray) -> np.ndarray:
        """Project an FC2 target into the prepared acoustic subspace when enabled."""
        if self._acoustic_coordinates is None:
            return parameters
        return self._acoustic_coordinates.project(parameters)

    def _matrices(self, parameters: np.ndarray) -> np.ndarray:
        """Evaluate effective FC2 dynamical matrices only at irreducible star representatives."""
        return accumulate_dynamical_matrices(
            np.ascontiguousarray(self.stars.points),
            self._dynamical_terms,
            _fc2_tensors(parameters, self._fc2_bases, self._fc2_offsets, self._fc2_dimensions),
            len(self._masses),
        )

    def _frequencies(self, parameters: np.ndarray) -> np.ndarray:
        """Return signed THz frequencies of the current representative dynamical matrices."""
        eigenvalues = np.linalg.eigvalsh(self._matrices(parameters))
        return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * THZ_PER_SQRT_EV_PER_A2_AMU

    def _minimum_mode(self, parameters: np.ndarray) -> float | None:
        """Return the minimum signed physical mode frequency on the mesh.

        The three rigid Gamma translations are excluded.
        """
        minimum = None
        internal = translation_complement(self._masses)
        for index, matrix in enumerate(self._matrices(parameters)):
            if index == self._gamma:
                if internal.shape[1] == 0:
                    continue
                matrix = internal.T @ matrix @ internal
            eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
            minimum = eigenvalue if minimum is None else min(minimum, eigenvalue)
        if minimum is None:
            return None
        return float(np.sign(minimum) * np.sqrt(abs(minimum)) * THZ_PER_SQRT_EV_PER_A2_AMU)

    def _correction(self, parameters: np.ndarray, temperature: float) -> np.ndarray:
        """Evaluate the quartic-loop correction to the FC2 parameter vector.

        The current effective FC2 defines harmonic covariances on the
        irreducible mesh. Transform the matrix-valued covariances to full-grid
        members, Fourier transform them to the real-space pairs required by
        FC4, and contract ``Delta Phi2 = 1/2 * Phi4 : <u u>``. Convert the
        representative FC2 observation components to the physical parameter
        coordinates of the original cluster space. This component selection
        is not a global Frobenius projection of the correction tensors.
        """
        matrices = self._matrices(parameters)
        covariance = np.asarray(
            [
                _modal_covariance(
                    matrix,
                    self._masses,
                    temperature,
                    gamma=index == self._gamma,
                    statistics=self.statistics,
                )
                for index, matrix in enumerate(matrices)
            ]
        )
        values = np.zeros((len(self._first), 3, 3), dtype=np.complex128)
        # Covariances are matrices in positional gauge, so scalar star weights
        # cannot replace rotating/phasing each member before Fourier accumulation.
        for member, matrix in self.plan.iter_matrices(covariance):
            _add_covariance(
                values,
                matrix,
                self._points[member],
                self._first,
                self._second,
                self._distances,
                self._mass_weights,
                self.stars.grid.size,
            )
        tensors = _contract(
            self._tensors,
            values,
            self._covariance_index,
            self._orbit_index,
            len(self._orbits),
        )
        correction = np.empty_like(self._bare)
        offset = 0
        for orbit, tensor in zip(self._orbits, tensors, strict=True):
            stop = offset + orbit.dimension
            correction[offset:stop] = tensor.reshape(-1)[orbit.observation_rows]
            offset = stop
        if not np.all(np.isfinite(correction)):
            raise ValueError("SCPH loop correction is not finite")
        return correction

    def run(
        self,
        temperature: float,
        *,
        start: ForceConstants | None = None,
        mixing: float = 0.2,
        tolerance: float = 1e-9,
        max_iterations: int = 200,
    ) -> SCPHResult:
        """Run the SCPH fixed-point iteration at one temperature.

        At iteration ``n``, evaluate the quartic-loop correction from the
        current effective FC2 and mix toward the updated model:

            Phi2[n+1] = (1 - mixing) * Phi2[n]
                + mixing * (Phi2_bare + Delta Phi2[Phi2[n]]).

        Convergence is reached when the star-weighted RMS change in signed
        phonon frequencies falls below ``tolerance``. The result is returned
        even when ``max_iterations`` is reached; inspect ``result.converged``.

        If the model's cluster space enables ASR, the initial FC2 and each
        target ``Phi2_bare + Delta Phi2`` are projected into its acoustic
        subspace before mixing. Projection failure raises ``RuntimeError``.

        Parameters
        ----------
        temperature : float
            Finite nonnegative temperature in kelvin.
        start : ForceConstants, optional
            Initial FC2 model; defaults to bare FC2. The caller must provide matching
            physical parameter layout and masses; compatibility is not checked.
        mixing : float, default 0.2
            Target fraction in (0, 1] for linear parameter mixing.
        tolerance : float, default 1e-9
            Positive star-weighted RMS frequency-change tolerance in THz.
        max_iterations : int, default 200
            Positive maximum iteration count.

        Returns
        -------
        result : SCPHResult
            Final FC2, representative frequencies, iteration history and stability
            diagnostic. Check converged explicitly; exhaustion does not raise.

        """
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative in kelvin")
        if not np.isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("mixing must be in (0, 1]")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite in THz")
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if start is None:
            current = self._bare.copy()
        else:
            if not isinstance(start, ForceConstants) or 2 not in start.orders:
                raise ValueError("start must be a ForceConstants model containing FC2")
            current = np.asarray(start.coefficients[2]).copy()
            if current.shape != self._bare.shape:
                raise ValueError(f"start FC2 parameters must have shape {self._bare.shape}")
            current = self._constrain_fc2(current)
        run_started = perf_counter()
        logger.info(
            "SCPH run started: temperature=%.6g K statistics=%s mixing=%.6g "
            "tolerance=%.6g THz max_iterations=%d warm_start=%s",
            temperature,
            self.statistics,
            mixing,
            tolerance,
            max_iterations,
            start is not None,
        )
        previous = self._frequencies(current)
        history = []
        converged = False
        for iteration in range(1, max_iterations + 1):
            started = perf_counter()
            correction = self._correction(current, temperature)
            # With ASR enabled, both endpoints and their linear mixture satisfy ASR.
            target = self._constrain_fc2(self._bare + correction)
            current = (1.0 - mixing) * current + mixing * target
            frequencies = self._frequencies(current)
            delta = frequencies - previous
            change = float(
                np.sqrt(
                    np.sum(self.stars.weights[:, None] * delta**2)
                    / (self.stars.grid.size * frequencies.shape[1])
                )
            )
            history.append(SCPHStep(iteration, change, float(np.min(frequencies))))
            previous = frequencies
            logger.info(
                "SCPH %.6g K iteration %d: frequency_change=%.10g THz "
                "minimum_frequency=%.10g THz elapsed=%.2f s",
                temperature,
                iteration,
                change,
                float(np.min(frequencies)),
                perf_counter() - started,
            )
            if change < tolerance:
                converged = True
                break
        result = SCPHResult(
            float(temperature),
            self._fc2(current),
            previous,
            self.stars,
            tuple(history),
            converged,
            self._minimum_mode(current),
        )
        logger.info(
            "SCPH run complete: temperature=%.6g K iterations=%d converged=%s "
            "minimum_internal_mode=%.10g THz elapsed_s=%.2f",
            result.temperature,
            result.iterations,
            result.converged,
            result.minimum_mode_thz if result.minimum_mode_thz is not None else float("nan"),
            perf_counter() - run_started,
        )
        return result

    def run_many(self, temperatures: object, **kwargs: object) -> list[SCPHResult]:
        """Solve a temperature sequence by cooling from high to low temperature.

        Each converged solution seeds the next lower temperature. Results are
        returned in ascending temperature order. Dynamical instability and
        numerical convergence remain separate diagnostics.
        """
        values = np.asarray(temperatures, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("temperatures must be a nonempty finite sequence")
        if values[0] < 0.0 or np.any(np.diff(values) <= 0.0):
            raise ValueError("temperatures must be non-negative and strictly increasing")
        if "start" in kwargs:
            raise TypeError("run_many manages the start from the previous temperature")
        results = []
        start = None
        logger.info("SCPH temperature schedule: %s K (high to low execution)", tuple(values))
        for temperature in values[::-1]:
            result = self.run(float(temperature), start=start, **kwargs)
            results.append(result)
            start = result.fc2 if result.converged else None
        results.reverse()
        return results


__all__ = ["SCPH", "SCPHResult", "SCPHStep"]
