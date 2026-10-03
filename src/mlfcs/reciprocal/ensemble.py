"""Stable harmonic Gaussian on one irreducible supercell reciprocal grid."""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np
from scipy.constants import Boltzmann, angstrom, atomic_mass, electron_volt, hbar

from mlfcs.force_constants import ForceConstants
from mlfcs.reciprocal.grid import QStars
from mlfcs.reciprocal.harmonic import _THZ, Harmonic
from mlfcs.reciprocal.stars import StarPlan
from mlfcs.mapping import ClusterMap

_OMEGA = np.sqrt(electron_volt / (angstrom**2 * atomic_mass))
_VARIANCE = hbar / (2.0 * atomic_mass * angstrom**2 * _OMEGA)


class UnstableTrialError(ValueError):
    """The non-translational trial spectrum cannot define a Gaussian."""


def internal_basis(masses: np.ndarray) -> np.ndarray:
    """Orthogonal complement of the three mass-weighted Gamma translations."""
    count = len(masses)
    if count == 1:
        return np.empty((3, 0))
    weights = np.sqrt(masses)
    weights /= np.linalg.norm(weights)
    direction = weights.copy()
    direction[-1] -= 1.0
    direction /= np.linalg.norm(direction)
    householder = np.eye(count) - 2.0 * np.outer(direction, direction)
    basis = np.zeros((3 * count, 3 * (count - 1)))
    for axis in range(3):
        basis[axis::3, axis::3] = householder[:, :-1]
    return basis


def _stream(
    seed: int, temperature: float, iteration: int, label: np.ndarray
) -> np.random.Generator:
    bits = int.from_bytes(struct.pack(">d", temperature), "big")
    words = [seed & 0xFFFFFFFF, seed >> 32, bits & 0xFFFFFFFF, bits >> 32, iteration]
    words.extend(int(value) for value in label)
    return np.random.default_rng(np.random.SeedSequence(words))


def _expand_vectors(plan: StarPlan, member: int, source: np.ndarray) -> np.ndarray:
    """Apply exactly the member matrix action to each modal vector."""
    stars = plan.stars
    operation = int(stars.operations[member])
    if stars.antiunitary[member]:
        source = source.conj()
    result = np.empty_like(source)
    permutation = stars.symmetry.site_permutations[operation]
    rotation = stars.symmetry.cartesian_rotations[operation]
    for site, target in enumerate(permutation):
        result[3 * target : 3 * target + 3] = (
            plan.phases[member, target] * rotation.T @ source[3 * site : 3 * site + 3]
        )
    return result


class HarmonicEnsemble:
    """Diagonalize star representatives, then draw every real-space q degree of freedom."""

    def __init__(
        self,
        model: ForceConstants,
        mapping: ClusterMap,
        stars: QStars,
        temperature: float,
        *,
        statistics: Literal["quantum", "classical"],
    ) -> None:
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and nonnegative")
        if statistics not in ("quantum", "classical"):
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if statistics == "classical" and temperature == 0.0:
            raise ValueError("classical sampling at zero temperature has no identifiable FC2")
        if model.space.fingerprint != mapping.cluster_space.fingerprint:
            raise ValueError("trial FC2 and cluster map use different primitive structures")
        if not np.array_equal(
            stars.grid.matrix, np.asarray(mapping.supercell_matrix, dtype=np.int64)
        ):
            raise ValueError("q grid and supercell use different integer matrices")
        self.model = model
        self.mapping = mapping
        self.stars = stars
        self.temperature = temperature
        self.statistics = statistics
        self.plan = StarPlan.from_stars(stars, model.space)
        self.masses = np.asarray(model.space.primitive_atoms.masses)
        self._basis = internal_basis(self.masses)
        self._gamma = next(
            index
            for index, member in enumerate(stars.representatives)
            if not np.any(stars.grid.labels[member])
        )
        matrices = Harmonic(model).matrices(stars)
        self.eigenvalues: tuple[np.ndarray, ...]
        self.vectors: tuple[np.ndarray, ...]
        eigenvalues = []
        vectors = []
        for star, matrix in enumerate(matrices):
            if star == self._gamma:
                values, columns = np.linalg.eigh(self._basis.T @ matrix @ self._basis)
                columns = self._basis @ columns
            else:
                values, columns = np.linalg.eigh(matrix)
            if not np.all(np.isfinite(values)):
                raise ValueError("trial dynamical matrix has a non-finite spectrum")
            if np.any(values <= 0.0):
                label = stars.grid.labels[stars.representatives[star]].tolist()
                raise UnstableTrialError(
                    f"trial FC2 is not a positive Gaussian: representative {label}, "
                    f"minimum non-translational curvature {float(np.min(values)):.10g} "
                    "eV/(angstrom^2 u); supply a stable trial or enable Cartesian bootstrap"
                )
            eigenvalues.append(values)
            vectors.append(columns)
        self.eigenvalues = tuple(eigenvalues)
        self.vectors = tuple(vectors)
        self.minimum_frequency_thz = float(
            min(
                (float(np.sqrt(np.min(values)) * _THZ) for values in eigenvalues if len(values)),
                default=float("inf"),
            )
        )

    def _variances(self, values: np.ndarray) -> np.ndarray:
        if self.statistics == "classical":
            return Boltzmann * self.temperature / (atomic_mass * angstrom**2 * _OMEGA**2 * values)
        result = _VARIANCE / np.sqrt(values)
        if self.temperature > 0.0:
            omega = _OMEGA * np.sqrt(values)
            result = result / np.tanh(hbar * omega / (2.0 * Boltzmann * self.temperature))
        return result

    def _member_vectors(self, member: int) -> np.ndarray:
        star = int(self.stars.star_of[member])
        return _expand_vectors(self.plan, member, self.vectors[star])

    def sample(self, pairs: int, *, seed: int, iteration: int) -> np.ndarray:
        """Return positive displacements; the caller evaluates each sign separately."""
        if not isinstance(pairs, int) or pairs < 1:
            raise ValueError("pairs must be a positive integer")
        grid = self.stars.grid
        labels = grid.labels
        lookup = {tuple(int(value) for value in label): index for index, label in enumerate(labels)}
        n_atoms = len(self.mapping.atomic_numbers)
        field = np.zeros((pairs, n_atoms, 3))
        sites = self.mapping.primitive_site_indices
        positions = self.model.space.scaled_positions[sites]
        translations = self.mapping.lattice_translations
        root_mass = np.sqrt(self.masses[sites])
        for member, label in enumerate(labels):
            partner = lookup[tuple(-int(value) % grid.denominator for value in label)]
            if partner < member:
                continue
            star = int(self.stars.star_of[member])
            values = self.eigenvalues[star]
            if not len(values):
                continue
            generator = _stream(seed, self.temperature, iteration, label)
            draw = generator.standard_normal((2, pairs, len(values)))
            amplitudes = (draw[0] + 1j * draw[1]) * np.sqrt(self._variances(values))
            modes = self._member_vectors(member)
            wave = amplitudes @ modes.T
            phases = np.exp(2j * np.pi * ((translations + positions) @ grid.points[member]))
            factor = np.sqrt(2.0 / grid.size) if partner != member else 1.0 / np.sqrt(grid.size)
            field += (
                factor
                * np.real(
                    wave.reshape(pairs, len(self.masses), 3)[:, sites, :] * phases[None, :, None]
                )
                / root_mass[None, :, None]
            )
        if not np.all(np.isfinite(field)):
            raise ValueError("harmonic sampling produced non-finite displacements")
        return field

    def frequency_change_thz(self, other: HarmonicEnsemble) -> float:
        if self.stars is not other.stars:
            raise ValueError("frequency comparison requires the same reciprocal stars")
        total = 0.0
        count = 0
        for weight, before, after in zip(
            self.stars.weights, self.eigenvalues, other.eigenvalues, strict=True
        ):
            difference = _THZ * (np.sqrt(before) - np.sqrt(after))
            total += int(weight) * float(difference @ difference)
            count += int(weight) * len(difference)
        return float(np.sqrt(total / count)) if count else 0.0

    def free_energy(self) -> float:
        """Harmonic free energy in eV per primitive cell."""
        total = 0.0
        thermal = Boltzmann * self.temperature / electron_volt
        for weight, values in zip(self.stars.weights, self.eigenvalues, strict=True):
            energy = hbar * _OMEGA * np.sqrt(values) / electron_volt
            if self.statistics == "classical":
                contribution = thermal * np.log(energy / thermal)
            elif self.temperature == 0.0:
                contribution = 0.5 * energy
            else:
                contribution = 0.5 * energy + thermal * np.log(-np.expm1(-energy / thermal))
            total += int(weight) * float(np.sum(contribution))
        return total / self.stars.grid.size


__all__ = ["HarmonicEnsemble", "UnstableTrialError", "internal_basis"]
