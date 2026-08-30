"""Quartic loop self-consistent phonons.

This module deliberately implements only the static quartic loop diagram.  The
result is a temperature-dependent real-space FC2 that can be handed to the
existing IO backends; it is not a frequency-dependent bubble self-energy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from mlfcs.force_constants.dense import lattice_fc2, replace_lattice_fc2
from mlfcs.force_constants.representation import (
    ForceConstants,
)
from mlfcs.phonon.reciprocal import quotient_qpoints
from mlfcs.phonon.scph.fourier import (
    _dynamical,
    _dynamical_batch,
    _fourier_terms,
    _multiplier,
    _needed_covariances,
    _validate_relation,
)
from mlfcs.phonon.statistics import HBAR_ASE, OMEGA_TO_THZ, mode_sigma
from mlfcs.physics.temperature import TemperatureSeriesResult, normalize_temperature_schedule

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
    temperature: float
    qpoints: np.ndarray
    frequencies: np.ndarray
    force_constants: ForceConstants
    history: tuple[LoopSCPHIteration, ...]
    converged: bool

    @property
    def iterations(self) -> int:
        return len(self.history)


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
        qpoint_workers: int = 1,
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
        if qpoint_workers < 1:
            raise ValueError("qpoint_workers must be positive")
        self.qpoint_workers = int(qpoint_workers)
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
        previous_frequencies = self._frequencies(current, self.interpolation_multiplier)[1]
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
            frequencies = self._frequencies(updated, self.interpolation_multiplier)[1]
            frequency_change = float(np.sqrt(np.mean((frequencies - previous_frequencies) ** 2)))
            history.append(LoopSCPHIteration(iteration, frequency_change, correction_norm))
            logger.info(
                f"SCPH iteration {iteration}: delta_omega={frequency_change:.6e} THz, "
                f"frequency_min={np.min(frequencies):.6e} THz, "
                f"frequency_max={np.max(frequencies):.6e} THz, "
                f"correction_norm={correction_norm:.6e}",
            )
            current = updated
            previous_frequencies = frequencies
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
            metadata={"method": "loop_scph", "temperature": temperature},
        )
        qpoints, frequencies = self._frequencies(current, self.interpolation_multiplier)
        return LoopSCPHResult(
            temperature,
            qpoints,
            frequencies,
            effective,
            tuple(history),
            converged,
        )

    def _covariance(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
        temperature: float,
    ) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(lattice, relation.primitive)
        primitive_positions = relation.primitive.get_scaled_positions(wrap=False)
        qpoints = self._qpoints(multiplier)
        n = len(qpoints)
        covariance: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
        needed = _needed_covariances(self.fc4.sparse[4])

        def covariance_at_q(q_chunk):
            result = {}
            q_chunk = np.asarray(q_chunk, dtype=float)
            dynamical = _dynamical_batch(terms, masses, q_chunk)
            values, vectors = np.linalg.eigh(dynamical)
            sigma2 = (
                mode_sigma(
                    values,
                    temperature=temperature,
                    statistics=self.statistics,
                    cutoff_frequency_thz=self.frequency_cutoff_thz,
                )
                ** 2
            )
            weighted = (vectors * sigma2[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
            for a, b, r in needed:
                block = weighted[:, 3 * a : 3 * a + 3, 3 * b : 3 * b + 3] / np.sqrt(
                    masses[a] * masses[b]
                )
                displacement = primitive_positions[a] - primitive_positions[b] + np.asarray(r)
                phase = np.exp(2j * np.pi * (q_chunk @ displacement))
                result[(a, b, r)] = np.sum(block * phase[:, None, None], axis=0)
            return result

        chunks = tuple(np.array_split(np.asarray(qpoints), min(self.qpoint_workers, len(qpoints))))
        if self.qpoint_workers == 1 or len(chunks) == 1:
            parts = (covariance_at_q(chunk) for chunk in chunks)
        else:
            with ThreadPoolExecutor(max_workers=self.qpoint_workers) as pool:
                parts = pool.map(covariance_at_q, chunks)
        for part in parts:
            for key, value in part.items():
                covariance[key] = covariance.get(key, 0.0) + value / n
        return covariance

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

    def _frequencies(
        self,
        lattice: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
        multiplier: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(lattice, relation.primitive)
        values = []
        qpoints = self._qpoints(multiplier)
        for q in qpoints:
            eigenvalues = np.linalg.eigvalsh(_dynamical(terms, masses, q))
            values.append(np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * _OMEGA_TO_THZ)
        return np.asarray(qpoints), np.asarray(values)

    def _qpoints(self, multiplier: int) -> np.ndarray:
        relation = self.fc2.relation
        assert relation is not None
        return quotient_qpoints(multiplier * relation.supercell_matrix)

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
