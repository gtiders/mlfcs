"""Quartic loop self-consistent phonons.

This module deliberately implements only the static quartic loop diagram.  The
result is a temperature-dependent real-space FC2 that can be handed to the
existing IO backends; it is not a frequency-dependent bubble self-energy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlfcs.anharmonic.common.thermodynamics import (
    HBAR_ASE,
    OMEGA_TO_THZ,
    mode_sigma,
    quotient_qpoints,
)
from mlfcs.anharmonic.common.schedule import TemperatureSeriesResult, normalize_temperature_schedule
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants

_HBAR_ASE = HBAR_ASE
_OMEGA_TO_THZ = OMEGA_TO_THZ


@dataclass(frozen=True, slots=True)
class LoopSCPHIteration:
    index: int
    frequency_change_thz: float


@dataclass(slots=True)
class LoopSCPHResult:
    temperature: float
    qpoints: np.ndarray
    frequencies: np.ndarray
    base_force_constants: ForceConstants
    loop_correction: ForceConstants
    effective_force_constants: ForceConstants
    history: tuple[LoopSCPHIteration, ...]
    converged: bool

    @property
    def iterations(self) -> int:
        return len(self.history)

    def write(self, target: str | Path, *, format: str, order: int = 2) -> None:
        """Write the temperature-dependent effective FC2 through normal IO."""
        self.effective_force_constants.write(target, format=format, order=order)


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
        verbose: bool = True,
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
        self.interpolation_multiplier = _multiplier(interpolation_multiplier, "interpolation_multiplier")
        self.scph_multiplier = _multiplier(scph_multiplier, "scph_multiplier")
        if self.scph_multiplier % self.interpolation_multiplier:
            raise ValueError("scph_multiplier must be a multiple of interpolation_multiplier")
        self.statistics = statistics
        self.mixing = float(mixing)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.frequency_cutoff_thz = float(frequency_cutoff_thz)
        self.verbose = bool(verbose)
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
                previous = result.effective_force_constants
            else:
                previous = self.warm_start
        return TemperatureSeriesResult(self.temperatures, tuple(results), self.continuation)

    def _run_single(
        self, temperature: float, warm_start: ForceConstants | None
    ) -> LoopSCPHResult:
        base = self._copy_order(self.fc2, 2)
        bare = _compact_fc2(base)
        current = bare.copy() if warm_start is None else _compact_fc2(warm_start)
        history: list[LoopSCPHIteration] = []
        previous_frequencies = self._frequencies(current, self.interpolation_multiplier)[1]
        converged = False
        last_correction = np.zeros_like(current)
        previous_covariance: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] | None = None
        for iteration in range(1, self.max_iterations + 1):
            covariance = self._covariance(current, self.scph_multiplier, temperature)
            if previous_covariance is not None:
                covariance = {
                    key: self.mixing * value
                    + (1.0 - self.mixing) * previous_covariance[key]
                    for key, value in covariance.items()
                }
            correction = self._loop_correction(covariance)
            target = bare + correction
            updated = target
            frequencies = self._frequencies(updated, self.interpolation_multiplier)[1]
            frequency_change = float(np.sqrt(np.mean((frequencies - previous_frequencies) ** 2)))
            history.append(LoopSCPHIteration(iteration, frequency_change))
            if self.verbose:
                print(
                    f"SCPH iteration {iteration}: delta_omega={frequency_change:.6e} THz, "
                    f"frequency_min={np.min(frequencies):.6e} THz, "
                    f"frequency_max={np.max(frequencies):.6e} THz, "
                    f"correction_norm={np.linalg.norm(correction):.6e}",
                    flush=True,
                )
            current = updated
            last_correction = updated - bare
            previous_frequencies = frequencies
            previous_covariance = covariance
            # Convergence is a fixed-point criterion.  An imaginary mode is a
            # physical diagnostic of the current solution, not an additional
            # stopping condition.
            if frequency_change < self.tolerance:
                converged = True
                break

        support = _fc2_support(base.sparse[2], self.fc4.sparse[4], base.relation)
        effective = _replace_fc2(base, current, support=support)
        correction_fc = _replace_fc2(base, last_correction, support=support)
        qpoints, frequencies = self._frequencies(current, self.interpolation_multiplier)
        return LoopSCPHResult(
            temperature,
            qpoints,
            frequencies,
            base,
            correction_fc,
            effective,
            tuple(history),
            converged,
        )

    def _covariance(
        self, compact: np.ndarray, multiplier: int, temperature: float
    ) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(compact, relation)
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
    ) -> np.ndarray:
        sparse = self.fc2.sparse[2]
        result = np.zeros((sparse.n_primitive, sparse.n_supercell, 3, 3), dtype=float)
        index = self.fc2.relation.index
        for sites, translations, tensor in zip(
            self.fc4.sparse[4].sites,
            self.fc4.sparse[4].translation_representatives,
            self.fc4.sparse[4].tensors,
            strict=True,
        ):
            s1, s2, s3, s4 = map(int, sites)
            r2, r3, r4 = (tuple(map(int, row)) for row in translations)
            cov = covariance.get((s3, s4, tuple(np.asarray(r3) - np.asarray(r4))))
            if cov is None:
                continue
            value = 0.5 * np.einsum("abcd,cd->ab", tensor, cov, optimize=True)
            atom2 = index.atom(s2, r2)
            result[s1, atom2] += value.real
        return result

    def _frequencies(
        self, compact: np.ndarray, multiplier: int
    ) -> tuple[np.ndarray, np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(compact, relation)
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


def harmonic_frequencies(
    fc2: ForceConstants, interpolation_multiplier: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic frequencies on a reference-supercell-derived q grid."""
    if 2 not in fc2.orders or fc2.relation is None:
        raise ValueError("fc2 must contain order-2 force constants and a structure relation")
    compact = _compact_fc2(fc2)
    masses = np.asarray(fc2.relation.primitive.get_masses(), dtype=float)
    terms = _fourier_terms(compact, fc2.relation)
    multiplier = _multiplier(interpolation_multiplier, "interpolation_multiplier")
    qpoints = quotient_qpoints(multiplier * fc2.relation.supercell_matrix)
    frequencies = []
    for q in qpoints:
        values = np.linalg.eigvalsh(_dynamical(terms, masses, q))
        frequencies.append(np.sqrt(np.abs(values)) * np.sign(values) * _OMEGA_TO_THZ)
    return qpoints, np.asarray(frequencies)


def _multiplier(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_relation(fc2: ForceConstants, fc4: ForceConstants) -> None:
    r2, r4 = fc2.relation, fc4.relation
    if r2 is None or r4 is None:
        raise ValueError("fc2 and fc4 must contain explicit StructureRelation objects")
    if not np.array_equal(r2.supercell_matrix, r4.supercell_matrix):
        raise ValueError("fc2 and fc4 supercell matrices differ")
    if (
        len(r2.reference) != len(r4.reference)
        or not np.array_equal(r2.reference.numbers, r4.reference.numbers)
        or not np.allclose(r2.primitive.cell, r4.primitive.cell, atol=1e-8, rtol=1e-10)
        or not np.allclose(r2.reference.cell, r4.reference.cell, atol=1e-8, rtol=1e-10)
        or not np.allclose(r2.reference.positions, r4.reference.positions, atol=1e-8, rtol=1e-10)
    ):
        raise ValueError("fc2 and fc4 reference supercells differ")


def _compact_fc2(fc: ForceConstants) -> np.ndarray:
    sparse = fc.sparse[2]
    if fc.relation is not None:
        primitive_index = np.asarray(fc.relation.primitive_index, dtype=np.int64)
    else:
        primitive_index = np.asarray(fc.supercell.arrays["primitive_index"], dtype=np.int64)
    result = np.zeros((sparse.n_primitive, sparse.n_supercell, 3, 3), dtype=float)
    counts = np.zeros((sparse.n_primitive, sparse.n_supercell), dtype=np.int32)
    for cluster, tensor in zip(sparse.clusters, sparse.tensors, strict=True):
        key = (int(primitive_index[cluster[0]]), int(cluster[1]))
        result[key] += tensor
        counts[key] += 1
    mask = counts > 0
    result[mask] /= counts[mask, None, None]
    return result


def _replace_fc2(
    base: ForceConstants,
    compact: np.ndarray,
    *,
    support: set[tuple[int, int, tuple[int, int, int]]] | None = None,
) -> ForceConstants:
    sparse_base = base.sparse[2]
    relation = base.relation
    if relation is None:
        raise ValueError("FC2 replacement requires an explicit StructureRelation")
    index = relation.index
    primitive_index = np.asarray(relation.primitive_index, dtype=np.int64)
    rows: list[tuple[int, int]] = [tuple(map(int, cluster)) for cluster in sparse_base.clusters]
    row_keys = {
        (int(primitive_index[cluster[0]]), int(cluster[1]), tuple(map(int, translation)))
        for cluster, translation in zip(
            sparse_base.clusters,
            sparse_base.translation_representatives[:, 0, :],
            strict=True,
        )
    }
    if support is not None:
        for site, other, translation in sorted(support):
            key = (site, other, translation)
            if key in row_keys:
                continue
            anchor = index.representative(site)
            atom = index.atom(other, translation)
            rows.append((anchor, atom))
            row_keys.add(key)
    clusters = np.asarray(rows, dtype=np.int32).reshape((-1, 2))
    tensors = np.asarray(
        [compact[int(primitive_index[c[0]]), int(c[1])] for c in clusters], dtype=float
    )
    sites = index.primitive[clusters]
    translations = np.asarray(
        [
            [index.canonical_translation(index.translations[c[1]] - index.translations[c[0]])]
            for c in clusters
        ],
        dtype=np.int32,
    )
    sparse = SparseOrderForceConstants(
        2,
        sparse_base.n_primitive,
        sparse_base.n_supercell,
        clusters,
        tensors,
        sites,
        translations,
    )
    reference = base.relation.reference if base.relation is not None else base.supercell
    return ForceConstants({}, reference.copy(), dict(base.metadata), {2: sparse}, base.relation)


def _fc2_support(
    fc2: SparseOrderForceConstants,
    fc4: SparseOrderForceConstants,
    relation,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    """Return all pair labels required by the bare and loop FC2 supports."""
    if fc2.sites is None or fc2.translation_representatives is None:
        raise ValueError("SCPH requires lattice-labelled FC2")
    if fc4.sites is None or fc4.translation_representatives is None:
        raise ValueError("SCPH requires lattice-labelled FC4")
    result = {
        (
            int(sites[0]),
            int(sites[1]),
            tuple(map(int, translations[0])),
        )
        for sites, translations in zip(
            fc2.sites, fc2.translation_representatives, strict=True
        )
    }
    result.update(
        (
            int(sites[0]),
            int(sites[1]),
            tuple(map(int, translations[0])),
        )
        for sites, translations in zip(
            fc4.sites, fc4.translation_representatives, strict=True
        )
    )
    for site, other, translation in result:
        relation.index.atom(site, (0, 0, 0))
        relation.index.atom(other, translation)
    return result


def _lattice_fc2(
    sparse: SparseOrderForceConstants,
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    result: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    counts: dict[tuple[int, int, tuple[int, int, int]], int] = {}
    for sites, translations, tensor in zip(
        sparse.sites, sparse.translation_representatives, sparse.tensors, strict=True
    ):
        key = (int(sites[0]), int(sites[1]), tuple(map(int, translations[0])))
        result[key] = result.get(key, 0.0) + tensor
        counts[key] = counts.get(key, 0) + 1
    for key in result:
        result[key] /= counts[key]
    return result


def _fourier_terms(compact: np.ndarray, relation):
    """Build Fourier terms from the exact reference lattice labels.

    The SCPH mesh need not be commensurate with the force-constant reference
    supercell.  Replacing a labelled translation by a nearest image of that
    supercell therefore changes the phase at a general q point.  Use the
    primitive-site positions and integer translation labels recovered by the
    structure relation instead.
    """
    index = relation.index
    primitive = relation.primitive
    inverse_primitive = np.linalg.inv(np.asarray(primitive.cell))
    terms = []
    for first in range(index.n_primitive):
        anchor = index.representative(first)
        for atom in range(len(relation.reference)):
            site = int(index.primitive[atom])
            translation = index.translations[atom] - index.translations[anchor]
            vector = (
                primitive.positions[site]
                - primitive.positions[first]
                + translation @ np.asarray(primitive.cell)
            )
            terms.append(
                (
                    first,
                    site,
                    np.asarray(vector) @ inverse_primitive,
                    compact[first, atom],
                )
            )
    return terms


def _dynamical(terms, masses: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = len(masses)
    matrix = np.zeros((3 * n, 3 * n), dtype=complex)
    for a, b, images, tensor in terms:
        phase = np.exp(2j * np.pi * float(images @ q))
        matrix[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += (
            tensor * phase / np.sqrt(masses[a] * masses[b])
        )
    return (matrix + matrix.conj().T) / 2


def _dynamical_batch(terms, masses: np.ndarray, qpoints: np.ndarray) -> np.ndarray:
    """Build all q-point dynamical matrices in one bounded batch."""
    qpoints = np.asarray(qpoints, dtype=float).reshape((-1, 3))
    n = len(masses)
    matrix = np.zeros((len(qpoints), 3 * n, 3 * n), dtype=complex)
    for first, second, images, tensor in terms:
        phase = np.exp(2j * np.pi * (qpoints @ images))
        matrix[:, 3 * first : 3 * first + 3, 3 * second : 3 * second + 3] += (
            phase[:, None, None] * tensor / np.sqrt(masses[first] * masses[second])
        )
    return (matrix + matrix.conj().swapaxes(-1, -2)) / 2


def _compact_to_lattice(
    compact: np.ndarray, sparse: SparseOrderForceConstants, index
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    result = _lattice_fc2(sparse)
    for sites, translations in zip(sparse.sites, sparse.translation_representatives, strict=True):
        key = (int(sites[0]), int(sites[1]), tuple(map(int, translations[0])))
        atom = index.atom(int(sites[1]), tuple(map(int, translations[0])))
        result[key] = compact[int(sites[0]), atom]
    return result


def _needed_covariances(
    sparse: SparseOrderForceConstants,
) -> set[tuple[int, int, tuple[int, int, int]]]:
    result: set[tuple[int, int, tuple[int, int, int]]] = set()
    for sites, translations in zip(sparse.sites, sparse.translation_representatives, strict=True):
        result.add(
            (
                int(sites[2]),
                int(sites[3]),
            tuple((np.asarray(translations[1]) - np.asarray(translations[2])).tolist()),
            )
        )
    return result


__all__ = ["LoopSCPH", "LoopSCPHIteration", "LoopSCPHResult", "harmonic_frequencies"]
