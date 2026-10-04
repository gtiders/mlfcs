"""Quartic-loop self-consistent phonons on an irreducible reciprocal grid."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numba import njit
from scipy.constants import Boltzmann, angstrom, atomic_mass, electron_volt, hbar

from mlfcs.core.log import get_logger
from mlfcs.core.tensors import rotate_basis
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.lattice import expand
from mlfcs.reciprocal.grid import QStars, _grid, _mass_symmetry
from mlfcs.reciprocal.harmonic import _THZ, Harmonic, _dynamical
from mlfcs.reciprocal.harmonic import internal_basis as _internal_basis
from mlfcs.reciprocal.stars import StarPlan

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
    """Accumulate one q member's Fourier covariance into selected pair blocks in place.

    result is complex128 (pairs, 3, 3), matrix (3*N, 3*N) in the mass-weighted
    modal basis, and qpoint a fractional reciprocal triple. first/second index
    sites; distances stores the corresponding fractional Fourier separation.
    Divide by n_qpoints and undo pair mass weighting. Caller supplies compatible
    indices; the resulting physical displacement covariance is in angstrom**2.
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
    """Contract quartic tensors with pair covariance into representative FC2 corrections.

    Inputs are tensors (terms, 3, 3, 3, 3), covariance (pairs, 3, 3), and
    per-term covariance/orbit indices. Negative orbit indices skip terms absent
    from the FC2 truncation. Sum half the real contraction over the last two
    axes; return float64 (n_orbits, 3, 3), in eV/angstrom**2.
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
    """Evaluate cached, symmetry-rotated FC2 bases without Python tensor work."""
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
    """Return mass-weighted modal displacement covariance at temperature in kelvin.

    matrix is a Hermitian (3*N, 3*N) dynamical matrix; masses are positive atomic
    mass units. Gamma translations are removed by an internal orthonormal
    basis. Quantum statistics include zero-point motion; classical statistics
    use the thermal limit. Negative curvature uses its magnitude as a trial
    covariance, without claiming physical stability. Nontranslational zero
    modes or nonfinite variance raise ValueError.
    """
    if gamma:
        internal = _internal_basis(masses)
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
    """One iteration's index, RMS frequency change and minimum signed frequency.

    Both frequency fields are in THz. minimum_frequency_thz includes the
    Gamma translations; the final result has a separate internal-mode diagnostic.
    """

    index: int
    frequency_change_thz: float
    minimum_frequency_thz: float


@dataclass(frozen=True, slots=True)
class SCPHResult:
    """SCPH outcome at a temperature in kelvin, including convergence diagnostics.

    fc2 is the final effective FC2-only model. frequencies has shape
    (n_stars, 3*n_atoms) in signed THz; stars provides full-grid expansion.
    history records iterations. converged tests frequency change only;
    minimum_mode_thz excludes Gamma translations and may be None if no internal
    mode exists. A converged result may still have imaginary physical modes.
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
    """Static quartic-loop self-consistent phonons on one exact reciprocal mesh.

    Parameters
    ----------
    model : ForceConstants
        Primitive model containing FC2 and FC4 on the same cluster space.
        Retained by reference; optional FC3 does not enter this approximation.
    mesh : QGrid or array_like of integers, shape (3,) or (3, 3)
        Positive diagonal mesh sizes or exact row-cell supercell matrix/grid.
    statistics : {'quantum', 'classical'}, default 'quantum'
        Modal covariance includes quantum zero-point motion or the classical limit.
    time_reversal : bool, default True
        Include time reversal when building irreducible reciprocal stars.

    Notes
    -----
    Initialization expands quartic terms and caches FC2 basis/Fourier metadata.
    Each iteration evaluates representatives and streams symmetry-related
    covariances through StarPlan, without storing a second full-grid stack.
    No fitted coefficients or reference geometry are modified.
    Invalid input model or statistics raises ValueError.
    """

    def __init__(
        self,
        model: ForceConstants,
        mesh: object,
        *,
        statistics: Literal["quantum", "classical"] = "quantum",
        time_reversal: bool = True,
    ) -> None:
        """Build the exact mesh/stars and cache harmonic, quartic and covariance-index metadata."""
        if not isinstance(model, ForceConstants) or not {2, 4} <= set(model.orders):
            raise ValueError("SCPH requires one ForceConstants model containing FC2 and FC4")
        if statistics not in ("quantum", "classical"):
            raise ValueError("statistics must be 'quantum' or 'classical'")
        grid = _grid(mesh)
        stars = QStars.from_symmetry(
            grid,
            _mass_symmetry(model.cluster_space.symmetry, model.cluster_space.masses),
            time_reversal=time_reversal,
        )
        self.model = model
        self.stars = stars
        self.plan = StarPlan.from_stars(stars, model.cluster_space)
        self._points = grid.points
        self.statistics = statistics
        self._bare = np.asarray(model.coefficients[2])
        self._masses = model.cluster_space.masses
        harmonic = Harmonic(self._fc2(self._bare))
        self._harmonic_first = harmonic._first
        self._harmonic_second = harmonic._second
        self._harmonic_images = harmonic._images
        self._harmonic_mass_weights = harmonic._mass_weights
        self._gamma = next(
            index
            for index, representative in enumerate(stars.representatives)
            if not np.any(grid.labels[representative])
        )
        quartic = expand(model, 4)
        block = model.cluster_space.block(2)
        self._orbits = model.cluster_space.orbits[block.orbits]
        maximum_dimension = max(orbit.dimension for orbit in self._orbits)
        bases = []
        offsets = []
        dimensions = []
        parameter_offset = 0
        for orbit in self._orbits:
            for operation, permutation in zip(orbit.operations, orbit.permutations, strict=True):
                rotation = model.cluster_space.symmetry.cartesian_rotations[operation].T
                basis = np.zeros((9, maximum_dimension), dtype=np.float64)
                basis[:, : orbit.dimension] = rotate_basis(
                    orbit.component_basis, rotation, permutation
                )
                bases.append(basis)
                offsets.append(parameter_offset)
                dimensions.append(orbit.dimension)
            parameter_offset += orbit.dimension
        self._fc2_bases = np.asarray(bases, dtype=np.float64)
        self._fc2_offsets = np.asarray(offsets, dtype=np.int32)
        self._fc2_dimensions = np.asarray(dimensions, dtype=np.int32)
        if len(self._fc2_offsets) != len(self._harmonic_first):
            raise RuntimeError("cached FC2 bases do not match the Fourier term order")
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
            len(self._harmonic_first),
            len(self._tensors),
            statistics,
        )

    def _fc2(self, parameters: np.ndarray) -> ForceConstants:
        """Bind a physical FC2 parameter vector to the original schema, omitting other orders."""
        return ForceConstants(self.model.cluster_space, {2: parameters})

    def _matrices(self, parameters: np.ndarray) -> np.ndarray:
        """Evaluate effective FC2 dynamical matrices only at irreducible star representatives."""
        return _dynamical(
            np.ascontiguousarray(self.stars.points),
            self._harmonic_first,
            self._harmonic_second,
            self._harmonic_images,
            _fc2_tensors(parameters, self._fc2_bases, self._fc2_offsets, self._fc2_dimensions),
            self._harmonic_mass_weights,
            len(self._masses),
        )

    def _frequencies(self, parameters: np.ndarray) -> np.ndarray:
        """Return signed THz frequencies of the current representative dynamical matrices."""
        eigenvalues = np.linalg.eigvalsh(self._matrices(parameters))
        return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _THZ

    def _minimum_mode(self, parameters: np.ndarray) -> float | None:
        """Smallest signed mesh frequency, excluding Gamma translations by construction."""
        minimum = None
        internal = _internal_basis(self._masses)
        for index, matrix in enumerate(self._matrices(parameters)):
            if index == self._gamma:
                if internal.shape[1] == 0:
                    continue
                matrix = internal.T @ matrix @ internal
            eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
            minimum = eigenvalue if minimum is None else min(minimum, eigenvalue)
        if minimum is None:
            return None
        return float(np.sign(minimum) * np.sqrt(abs(minimum)) * _THZ)

    def _correction(self, parameters: np.ndarray, temperature: float) -> np.ndarray:
        """Build the physical quartic-loop FC2 parameter correction for one temperature.

        Compute representative covariances, stream full-grid member actions, then
        contract FC4 into representative FC2 tensors. Observation rows recover
        physical component parameters. Nonfinite correction raises ValueError;
        input parameters and bare model are unchanged.
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
        """Iterate one temperature and return an SCPHResult, even if iteration limits are reached.

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

        Notes
        -----
        Update toward bare FC2 plus the current quartic-loop correction. Convergence
        is numerical and independent of imaginary-mode diagnostics. Invalid options
        or nonfinite/zero-mode covariances raise ValueError.
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
            target = self._bare + correction
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
        """Solve from high to low temperature; return results in ascending order.

        Only a converged solution seeds the next temperature. An imaginary mode
        is allowed and reported separately from numerical convergence.
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
