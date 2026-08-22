"""Quartic loop self-consistent phonons.

This module deliberately implements only the static quartic loop diagram.  The
result is a temperature-dependent real-space FC2 that can be handed to the
existing IO backends; it is not a frequency-dependent bubble self-energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlfcs.anharmonic.core import HBAR_ASE, OMEGA_TO_THZ, mode_sigma, regular_qpoints
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.structure.geometry import PeriodicGeometry

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
        temperature: float,
        interpolation_mesh: tuple[int, int, int],
        scph_mesh: tuple[int, int, int],
        statistics: str = "quantum",
        mixing: float = 0.1,
        tolerance: float = 1e-10,
        max_iterations: int = 100,
        frequency_cutoff_thz: float = 0.01,
        warm_start: ForceConstants | None = None,
    ) -> None:
        if not isinstance(fc2, ForceConstants) or not isinstance(fc4, ForceConstants):
            raise TypeError("fc2 and fc4 must be ForceConstants objects")
        if 2 not in fc2.orders:
            raise ValueError("fc2 does not contain order-2 force constants")
        if 4 not in fc4.orders:
            raise ValueError("fc4 does not contain order-4 force constants")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if statistics not in {"quantum", "classical"}:
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if not 0 < mixing <= 1:
            raise ValueError("mixing must be in (0, 1]")
        if tolerance <= 0 or max_iterations < 1:
            raise ValueError("tolerance must be positive and max_iterations >= 1")
        self.fc2 = fc2
        self.fc4 = fc4
        self.temperature = float(temperature)
        self.interpolation_mesh = _mesh(interpolation_mesh, "interpolation_mesh")
        self.scph_mesh = _mesh(scph_mesh, "scph_mesh")
        if any(s % i for s, i in zip(self.scph_mesh, self.interpolation_mesh, strict=True)):
            raise ValueError("each scph_mesh value must be a multiple of interpolation_mesh")
        self.statistics = statistics
        self.mixing = float(mixing)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.frequency_cutoff_thz = float(frequency_cutoff_thz)
        _validate_relation(fc2, fc4)
        if warm_start is not None:
            if not isinstance(warm_start, ForceConstants) or 2 not in warm_start.orders:
                raise TypeError("warm_start must be a ForceConstants object containing FC2")
            _validate_relation(fc2, warm_start)
        self.warm_start = warm_start
        self._primitive = fc2.relation.primitive if fc2.relation is not None else None
        if self._primitive is None:
            raise ValueError("fc2 must contain an explicit StructureRelation")

    def run(self) -> LoopSCPHResult:
        base = self._copy_order(self.fc2, 2)
        bare = _compact_fc2(base)
        current = bare.copy() if self.warm_start is None else _compact_fc2(self.warm_start)
        history: list[LoopSCPHIteration] = []
        previous_frequencies = self._frequencies(current, self.interpolation_mesh)[1]
        converged = False
        last_correction = np.zeros_like(current)
        for iteration in range(1, self.max_iterations + 1):
            covariance = self._covariance(current, self.scph_mesh)
            correction = self._loop_correction(covariance)
            target = bare + correction
            residual = target - current
            updated = current + self.mixing * residual
            frequencies = self._frequencies(updated, self.interpolation_mesh)[1]
            frequency_change = float(np.sqrt(np.mean((frequencies - previous_frequencies) ** 2)))
            history.append(LoopSCPHIteration(iteration, frequency_change))
            current = updated
            last_correction = updated - bare
            previous_frequencies = frequencies
            if frequency_change < self.tolerance and np.min(frequencies) >= 0.0:
                converged = True
                break

        effective = _replace_fc2(base, current)
        correction_fc = _replace_fc2(base, last_correction)
        qpoints, frequencies = self._frequencies(current, self.interpolation_mesh)
        return LoopSCPHResult(
            self.temperature,
            qpoints,
            frequencies,
            base,
            correction_fc,
            effective,
            tuple(history),
            converged,
        )

    def _covariance(
        self, compact: np.ndarray, mesh: tuple[int, int, int]
    ) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(compact, relation)
        primitive_positions = relation.primitive.get_scaled_positions(wrap=False)
        n = np.prod(mesh)
        covariance: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
        needed = _needed_covariances(self.fc4.sparse[4])
        for q in regular_qpoints(mesh):
            dynamical = _dynamical(terms, masses, q)
            values, vectors = np.linalg.eigh(dynamical)
            sigma2 = (
                mode_sigma(
                    values,
                    temperature=self.temperature,
                    statistics=self.statistics,
                    cutoff_frequency_thz=self.frequency_cutoff_thz,
                )
                ** 2
            )
            weighted = (vectors * sigma2[None, :]) @ vectors.conj().T
            for a, b, r in needed:
                block = weighted[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] / np.sqrt(
                    masses[a] * masses[b]
                )
                displacement = primitive_positions[a] - primitive_positions[b] + np.asarray(r)
                phase = np.exp(2j * np.pi * np.dot(q, displacement))
                key = (a, b, r)
                covariance[key] = covariance.get(key, 0.0) + block * phase / n
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
        self, compact: np.ndarray, mesh: tuple[int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        relation = self.fc2.relation
        assert relation is not None
        masses = np.asarray(relation.primitive.get_masses(), dtype=float)
        terms = _fourier_terms(compact, relation)
        values = []
        qpoints = list(regular_qpoints(mesh))
        for q in qpoints:
            eigenvalues = np.linalg.eigvalsh(_dynamical(terms, masses, q))
            values.append(np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * _OMEGA_TO_THZ)
        return np.asarray(qpoints), np.asarray(values)

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
    fc2: ForceConstants, mesh: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic frequencies on a regular fractional q-point mesh."""
    if 2 not in fc2.orders or fc2.relation is None:
        raise ValueError("fc2 must contain order-2 force constants and a structure relation")
    compact = _compact_fc2(fc2)
    masses = np.asarray(fc2.relation.primitive.get_masses(), dtype=float)
    terms = _fourier_terms(compact, fc2.relation)
    qpoints = np.asarray(list(regular_qpoints(_mesh(mesh, "mesh"))))
    frequencies = []
    for q in qpoints:
        values = np.linalg.eigvalsh(_dynamical(terms, masses, q))
        frequencies.append(np.sqrt(np.abs(values)) * np.sign(values) * _OMEGA_TO_THZ)
    return qpoints, np.asarray(frequencies)


def _mesh(value: tuple[int, int, int], name: str) -> tuple[int, int, int]:
    result = tuple(int(x) for x in value)
    if len(result) != 3 or any(x < 1 for x in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result


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


def _replace_fc2(base: ForceConstants, compact: np.ndarray) -> ForceConstants:
    sparse_base = base.sparse[2]
    primitive_index = np.asarray(base.relation.primitive_index, dtype=np.int64)
    tensors = np.asarray(
        [compact[int(primitive_index[c[0]]), int(c[1])] for c in sparse_base.clusters],
        dtype=float,
    )
    sparse = SparseOrderForceConstants(
        2,
        sparse_base.n_primitive,
        sparse_base.n_supercell,
        sparse_base.clusters.copy(),
        tensors,
        sparse_base.sites.copy() if sparse_base.sites is not None else None,
        sparse_base.translation_representatives.copy()
        if sparse_base.translation_representatives is not None
        else None,
    )
    reference = base.relation.reference if base.relation is not None else base.supercell
    return ForceConstants({}, reference.copy(), dict(base.metadata), {2: sparse}, base.relation)


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
    """Build Wigner--Seitz Fourier terms with equal degenerate-image weights."""
    index = relation.index
    geometry = PeriodicGeometry(relation.reference.cell, relation.reference.pbc)
    inverse_primitive = np.linalg.inv(np.asarray(relation.primitive.cell))
    terms = []
    for first in range(index.n_primitive):
        anchor = index.representative(first)
        for atom in range(len(relation.reference)):
            images, _ = geometry.closest_images(
                relation.reference.positions[atom] - relation.reference.positions[anchor]
            )
            terms.append(
                (
                    first,
                    int(index.primitive[atom]),
                    images @ inverse_primitive,
                    compact[first, atom],
                )
            )
    return terms


def _dynamical(terms, masses: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = len(masses)
    matrix = np.zeros((3 * n, 3 * n), dtype=complex)
    for a, b, images, tensor in terms:
        phase = np.mean(np.exp(2j * np.pi * (images @ q)))
        matrix[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += (
            tensor * phase / np.sqrt(masses[a] * masses[b])
        )
    return (matrix + matrix.conj().T) / 2


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
