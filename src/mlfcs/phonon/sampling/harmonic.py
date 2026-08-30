from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms, units

from mlfcs.phonon.reciprocal import reciprocal_quotient_grid
from mlfcs.phonon.statistics import HBAR_ASE, OMEGA_TO_THZ, mode_sigma
from mlfcs.structure.supercell_mapping import PeriodicIndex

Statistics = Literal["quantum", "classical"]
ImaginaryModePolicy = Literal["error", "absolute", "exclude"]

_HBAR_ASE = HBAR_ASE
_OMEGA_TO_THZ = OMEGA_TO_THZ
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingState:
    qpoints: int
    total_modes: int
    sampled_modes: int
    excluded_modes: int
    imaginary_modes: int
    minimum_frequency_thz: float
    maximum_displacement: float | None
    maximum_sampled_displacement: float
    clipped_atoms: int
    affected_snapshots: int


@dataclass(frozen=True, slots=True)
class _Modes:
    qpoint: np.ndarray
    qlabel: tuple[int, int, int]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    paired: bool
    frequencies_thz: np.ndarray
    included: np.ndarray


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
        self._qgrid = reciprocal_quotient_grid(self._index.supercell_matrix)
        self._modes = self._prepare_modes()
        self._last_state: SamplingState | None = None

    @property
    def qpoints(self) -> np.ndarray:
        return np.asarray([modes.qpoint for modes in self._modes])

    @property
    def frequencies(self) -> tuple[np.ndarray, ...]:
        return tuple(modes.frequencies_thz.copy() for modes in self._modes)

    @property
    def state(self) -> SamplingState:
        if self._last_state is None:
            return self._state(0.0, 0, 0)
        return self._last_state

    def sample(self, snapshots: int, *, random_seed: int | None = None) -> np.ndarray:
        if snapshots < 1:
            raise ValueError("snapshots must be positive")
        rng = np.random.default_rng(random_seed)
        displacement = np.zeros((snapshots, self._n_cells, self._n_primitive, 3), dtype=float)
        inverse_root_mass = 1.0 / np.sqrt(self._masses)[None, :, None]
        for modes in self._modes:
            sigma = self._mode_sigma(modes.eigenvalues)
            sigma = np.where(modes.included, sigma, 0.0)
            phase = np.exp(2j * np.pi * (self._cell_translations @ modes.qpoint))
            if modes.paired:
                normal = (
                    rng.standard_normal((snapshots, len(sigma)))
                    + 1j * rng.standard_normal((snapshots, len(sigma)))
                ) / np.sqrt(2.0)
                reduced = (normal * sigma) @ modes.eigenvectors.T
                field = np.sqrt(2.0 / self._n_cells) * np.real(
                    reduced[:, None, :] * phase[None, :, None]
                )
            else:
                normal = rng.standard_normal((snapshots, len(sigma)))
                reduced = (normal * sigma) @ modes.eigenvectors.T
                field = np.real(reduced[:, None, :] * phase[None, :, None]) / np.sqrt(self._n_cells)
            displacement += (
                field.reshape(snapshots, self._n_cells, self._n_primitive, 3) * inverse_root_mass
            )

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
        """Return harmonic free energy per primitive cell in eV."""
        total = 0.0
        for modes in self._modes:
            omega = np.sqrt(np.abs(modes.eigenvalues[modes.included]))
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
            weight = 2 if modes.paired else 1
            total += weight * float(np.sum(contribution))
        return total / self._n_cells

    def _prepare_modes(self) -> tuple[_Modes, ...]:
        modes = []
        visited: set[tuple[int, int, int]] = set()
        imaginary_count = 0
        for label_values, qpoint in zip(self._qgrid.labels, self._qgrid.points, strict=True):
            grid_index = tuple(int(value) for value in label_values)
            if grid_index in visited:
                continue
            partner = self._qgrid.negative_label(label_values)
            paired = partner != grid_index
            visited.add(grid_index)
            visited.add(partner)
            dynamical = self._dynamical_matrix(qpoint)
            if grid_index == (0, 0, 0):
                eigenvalues, eigenvectors = self._gamma_internal_modes(dynamical)
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
            frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * _OMEGA_TO_THZ
            imaginary = frequencies < -self.imaginary_tolerance
            imaginary_count += (2 if paired else 1) * int(np.count_nonzero(imaginary))
            if self.imaginary_modes == "error" and np.any(imaginary):
                minimum = float(np.min(frequencies))
                raise ValueError(
                    f"imaginary harmonic modes detected (minimum {minimum:.8f} THz); "
                    "choose imaginary_modes='absolute' or 'exclude' explicitly"
                )
            included = np.abs(frequencies) > self.cutoff_frequency
            if self.imaginary_modes == "exclude":
                included &= ~imaginary
            modes.append(
                _Modes(
                    qpoint,
                    grid_index,
                    eigenvalues,
                    eigenvectors,
                    paired,
                    frequencies,
                    included,
                )
            )
        self._imaginary_count = imaginary_count
        return tuple(modes)

    def _dynamical_matrix(self, qpoint: np.ndarray) -> np.ndarray:
        values = self._compact[:, self._cell_atoms.reshape(-1)].reshape(
            self._n_primitive, self._n_cells, self._n_primitive, 3, 3
        )
        phase = np.exp(2j * np.pi * (self._cell_translations @ qpoint))
        blocks = np.einsum("c,kclab->kalb", phase, values, optimize=True)
        mass = np.sqrt(self._masses[:, None] * self._masses[None, :])
        blocks /= mass[:, None, :, None]
        matrix = blocks.transpose(0, 1, 2, 3).reshape(3 * self._n_primitive, 3 * self._n_primitive)
        return (matrix + matrix.conj().T) / 2

    def _gamma_internal_modes(self, dynamical: np.ndarray):
        translations = np.zeros((3 * self._n_primitive, 3))
        for atom, mass in enumerate(self._masses):
            translations[3 * atom : 3 * atom + 3] = np.eye(3) * np.sqrt(mass)
        complete = np.linalg.qr(translations, mode="complete")[0]
        internal = complete[:, 3:]
        reduced = internal.T @ dynamical.real @ internal
        eigenvalues, eigenvectors = np.linalg.eigh((reduced + reduced.T) / 2)
        return eigenvalues, internal @ eigenvectors

    def _mode_sigma(self, eigenvalues: np.ndarray) -> np.ndarray:
        return mode_sigma(
            eigenvalues,
            temperature=self.temperature,
            statistics=self.statistics,
        )

    def _state(
        self, maximum_sampled: float, clipped_atoms: int, affected_snapshots: int
    ) -> SamplingState:
        frequencies = np.concatenate([modes.frequencies_thz for modes in self._modes])
        total = sum((2 if modes.paired else 1) * len(modes.included) for modes in self._modes)
        sampled = sum(
            (2 if modes.paired else 1) * int(np.count_nonzero(modes.included))
            for modes in self._modes
        )
        return SamplingState(
            int(self._n_cells),
            int(total),
            int(sampled),
            int(total - sampled),
            int(self._imaginary_count),
            float(np.min(frequencies)) if len(frequencies) else float("nan"),
            self.max_displacement,
            maximum_sampled,
            clipped_atoms,
            affected_snapshots,
        )
