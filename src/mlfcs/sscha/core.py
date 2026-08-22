"""Self-consistent harmonic sampling and FC2 fitting with native MLFCS components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from numpy.typing import ArrayLike, NDArray

from mlfcs.core.fc2 import compact_fc2, expand_compact_fc2
from mlfcs.core.geometry import make_supercell
from mlfcs.fitting import ForceConstantFitter
from mlfcs.model import ForceConstants
from mlfcs.sscha.ensemble import EnsembleDiagnostics, HarmonicEnsemble

Progress = Callable[[int, int], None]
ForceInput = NDArray[np.floating] | Sequence[ArrayLike] | Mapping[int, ArrayLike]


@dataclass(frozen=True, slots=True)
class SSCHAIteration:
    """Immutable summary of one fitted effective harmonic Hamiltonian."""

    index: int
    sampling: Literal["cartesian", "canonical"]
    force_constants: NDArray[np.float64]
    free_energy: float | None
    free_energy_error: float | None
    potential_energy: float | None
    harmonic_potential_energy: float
    ensemble: EnsembleDiagnostics | None
    fitting_relative_force_error: float


class SSCHA:
    """Iterative effective-harmonic sampling driven by arbitrary ASE forces."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        supercell: Sequence[int] = (2, 2, 2),
        temperature: float = 300.0,
        statistics: Literal["quantum", "classical"] = "quantum",
        snapshots: int | Literal["auto"] = 1000,
        max_iterations: int = 10,
        initial_displacement: float = 0.01,
        random_seed: int | None = None,
        symprec: float = 1e-5,
        cutoff_frequency: float = 0.01,
        imaginary_modes: Literal["error", "absolute", "exclude"] = "error",
        imaginary_tolerance: float = 1e-6,
        max_displacement: float | None = None,
        initial_force_constants: ArrayLike | None = None,
        acoustic_sum_rule: bool = True,
        rotational_invariance: int = 0,
        log_level: int = 0,
    ) -> None:
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        repeats = tuple(int(value) for value in supercell)
        if len(repeats) != 3 or any(value < 1 for value in repeats):
            raise ValueError("supercell must contain three positive diagonal repeats")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if snapshots != "auto" and snapshots < 1:
            raise ValueError("snapshots must be positive or 'auto'")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if initial_displacement <= 0:
            raise ValueError("initial_displacement must be positive")
        if statistics not in {"quantum", "classical"}:
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if imaginary_modes not in {"error", "absolute", "exclude"}:
            raise ValueError("imaginary_modes must be 'error', 'absolute', or 'exclude'")
        if cutoff_frequency < 0 or imaginary_tolerance < 0:
            raise ValueError("frequency tolerances must be non-negative")
        if max_displacement is not None and max_displacement <= 0:
            raise ValueError("max_displacement must be positive or None")

        self.primitive = atoms.copy()
        self.primitive.wrap()
        self.supercell = repeats
        self._reference, self._index = make_supercell(self.primitive, repeats)
        self.temperature = float(temperature)
        self.statistics = statistics
        self.snapshots = snapshots
        self.max_iterations = max_iterations
        self.initial_displacement = float(initial_displacement)
        self.random_seed = random_seed
        self.symprec = symprec
        self.cutoff_frequency = float(cutoff_frequency)
        self.imaginary_modes = imaginary_modes
        self.imaginary_tolerance = float(imaginary_tolerance)
        self.max_displacement = max_displacement
        self.acoustic_sum_rule = acoustic_sum_rule
        self.rotational_invariance = rotational_invariance
        self.log_level = log_level
        self.history: list[SSCHAIteration] = []
        self._prepared_index: int | None = None
        self._prepared_structures: list[Atoms] | None = None
        self._sampling_compact: NDArray[np.float64] | None = None
        self._sampling_ensemble: HarmonicEnsemble | None = None
        self._reference_energy: float | None = None
        self._fitter = ForceConstantFitter(
            self.primitive,
            self._reference,
            supercell=repeats,
            orders=(2,),
            cutoffs={2: None},
            symprec=symprec,
            verbose=log_level > 1,
        )
        self._active_compact: NDArray[np.float64] | None = None
        if initial_force_constants is not None:
            values = np.asarray(initial_force_constants, dtype=float)
            compact_shape = (len(self.primitive), len(self._reference), 3, 3)
            full_shape = (len(self._reference), len(self._reference), 3, 3)
            if values.shape == compact_shape:
                self._active_compact = values.copy()
            elif values.shape == full_shape:
                self._active_compact = compact_fc2(values, self._reference)
            else:
                raise ValueError(
                    "initial_force_constants must have compact shape "
                    f"{compact_shape} or full shape {full_shape}"
                )

    @property
    def force_constants(self) -> NDArray[np.float64] | None:
        """Return the active FC2 in full internal-supercell atom order."""
        if self._active_compact is None:
            return None
        return expand_compact_fc2(self._active_compact, self._reference)

    @property
    def compact_force_constants(self) -> NDArray[np.float64] | None:
        return None if self._active_compact is None else self._active_compact.copy()

    @property
    def supercell_atoms(self) -> Atoms:
        return self._reference.copy()

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    def sow(self) -> list[Atoms]:
        """Create snapshots for the next iteration in deterministic reap order."""
        index = self.current_iteration
        if index > self.max_iterations:
            raise StopIteration("all requested SSCHA iterations are complete")
        if self._prepared_index == index:
            assert self._prepared_structures is not None
            return [atoms.copy() for atoms in self._prepared_structures]

        count = self._snapshot_count()
        self._sampling_compact = (
            None if self._active_compact is None else self._active_compact.copy()
        )
        self._sampling_ensemble = None
        if self._sampling_compact is None:
            rng = np.random.default_rng(self.random_seed)
            displacement = rng.normal(
                scale=self.initial_displacement,
                size=(count, len(self._reference), 3),
            )
            displacement -= displacement.mean(axis=1, keepdims=True)
            sampling = "cartesian"
        else:
            self._sampling_ensemble = self._make_ensemble(self._sampling_compact)
            displacement = self._sampling_ensemble.sample(
                count, random_seed=self._sampling_seed(index)
            )
            sampling = "canonical"

        structures = []
        for configuration, values in enumerate(displacement):
            atoms = self._reference.copy()
            atoms.positions += values
            atoms.info.update(
                mlfcs_sscha_iteration=index,
                mlfcs_configuration_id=configuration,
                mlfcs_sscha_sampling=sampling,
            )
            structures.append(atoms)
        self._prepared_index = index
        self._prepared_structures = structures
        if self.log_level:
            print(f"[SSCHA {index}/{self.max_iterations}] {sampling} sampling")
            if self._sampling_ensemble is not None:
                self._report_ensemble(self._sampling_ensemble.diagnostics)
        return [atoms.copy() for atoms in structures]

    def reap(
        self,
        forces: ForceInput,
        *,
        energies: ArrayLike | Mapping[int, float] | None = None,
        reference_energy: float | None = None,
    ) -> SSCHAIteration:
        """Fit the next effective FC2 using the native streamed-Gram fitter."""
        snapshots = self.sow()
        n_snapshots, n_atoms = len(snapshots), len(self._reference)
        force_array = self._ordered(forces, n_snapshots, "forces")
        if force_array.shape != (n_snapshots, n_atoms, 3):
            raise ValueError(
                f"forces must have shape {(n_snapshots, n_atoms, 3)}, got {force_array.shape}"
            )
        energy_array = None
        if energies is not None:
            energy_array = self._ordered(energies, n_snapshots, "energies").reshape(-1)
            if energy_array.shape != (n_snapshots,):
                raise ValueError(f"energies must have shape {(n_snapshots,)}")
        if reference_energy is not None:
            self._reference_energy = float(reference_energy)

        for atoms, values in zip(snapshots, force_array, strict=True):
            atoms.new_array("forces", np.asarray(values, dtype=float))
        fit = self._fitter.fit(
            snapshots,
            validation_split=0.0,
            acoustic_sum_rule=self.acoustic_sum_rule,
            rotational_invariance=self.rotational_invariance,
        )
        fitted_compact = fit.force_constants.materialize(2, max_bytes=None)
        fitted_full = expand_compact_fc2(fitted_compact, self._reference)
        displacement = np.asarray(
            [atoms.positions - self._reference.positions for atoms in snapshots]
        )
        trial_compact = fitted_compact if self._sampling_compact is None else self._sampling_compact
        trial_full = expand_compact_fc2(trial_compact, self._reference)
        harmonic_each = (
            np.einsum("ijab,mia,mjb->m", trial_full, displacement, displacement, optimize=True) / 2
        )
        free_energy = free_energy_error = potential_energy = None
        ensemble = self._sampling_ensemble or self._make_ensemble(trial_compact)
        if energy_array is not None and self._reference_energy is not None:
            potential_each = energy_array - self._reference_energy
            potential_energy = float(np.mean(potential_each))
            if self._sampling_compact is not None:
                correction = (potential_each - harmonic_each) / int(np.prod(self.supercell))
                free_energy = float(ensemble.harmonic_free_energy() + np.mean(correction))
                free_energy_error = (
                    float(np.std(correction, ddof=1) / np.sqrt(len(correction)))
                    if len(correction) > 1
                    else float("nan")
                )
        result = SSCHAIteration(
            index=self.current_iteration,
            sampling="cartesian" if self._sampling_compact is None else "canonical",
            force_constants=fitted_full,
            free_energy=free_energy,
            free_energy_error=free_energy_error,
            potential_energy=potential_energy,
            harmonic_potential_energy=float(np.mean(harmonic_each)),
            ensemble=None if self._sampling_ensemble is None else ensemble.diagnostics,
            fitting_relative_force_error=fit.diagnostics.training_relative_force_error,
        )
        self._active_compact = fitted_compact.copy()
        self.history.append(result)
        self._prepared_index = None
        self._prepared_structures = None
        self._sampling_compact = None
        self._sampling_ensemble = None
        return result

    def step(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHAIteration:
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        if calculate_free_energy and self._reference_energy is None:
            equilibrium = self.supercell_atoms
            equilibrium.calc = calculator
            self._reference_energy = float(equilibrium.get_potential_energy())
        structures = self.sow()
        forces = np.empty((len(structures), len(structures[0]), 3))
        energies = np.empty(len(structures)) if calculate_free_energy else None
        for index, atoms in enumerate(structures):
            atoms.calc = calculator
            forces[index] = atoms.get_forces()
            if energies is not None:
                energies[index] = atoms.get_potential_energy()
            if progress is not None:
                progress(index + 1, len(structures))
        return self.reap(forces, energies=energies)

    def run(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHA:
        while self.current_iteration <= self.max_iterations:
            self.step(
                calculator,
                progress=progress,
                calculate_free_energy=calculate_free_energy,
            )
        return self

    def averaged_force_constants(self, last: int) -> NDArray[np.float64]:
        if last < 1 or not self.history:
            raise ValueError("last must be positive and at least one iteration must exist")
        return np.mean([item.force_constants for item in self.history[-last:]], axis=0)

    def use_average(self, last: int) -> NDArray[np.float64]:
        full = self.averaged_force_constants(last)
        self._active_compact = compact_fc2(full, self._reference)
        return full

    def write(self, target: str | Path, *, format: Literal["text", "hdf5"] = "hdf5") -> None:
        if self._active_compact is None:
            raise RuntimeError("no force constants are available")
        values = ForceConstants(
            {2: self._active_compact.copy()},
            self._reference.copy(),
            metadata={"method": "sscha", "temperature": self.temperature},
        )
        if format == "text":
            values.write(target, format="phonopy", order=2)
        elif format == "hdf5":
            values.write(target, format="phonopy_hdf5", order=2)
        else:
            raise ValueError("format must be 'text' or 'hdf5'")

    def _snapshot_count(self) -> int:
        if self.snapshots != "auto":
            return self.snapshots
        equations = max(3 * len(self._reference) - 3, 1)
        return max(1, int(np.ceil(4 * self._fitter.n_parameters / equations)))

    def _make_ensemble(self, compact: np.ndarray) -> HarmonicEnsemble:
        return HarmonicEnsemble(
            self.primitive,
            self._reference,
            compact,
            temperature=self.temperature,
            statistics=self.statistics,
            cutoff_frequency=self.cutoff_frequency,
            imaginary_modes=self.imaginary_modes,
            imaginary_tolerance=self.imaginary_tolerance,
            max_displacement=self.max_displacement,
        )

    def _sampling_seed(self, iteration: int) -> int | None:
        """Derive an independent reproducible seed for one canonical iteration."""
        if self.random_seed is None:
            return None
        sequence = np.random.SeedSequence([self.random_seed, iteration])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _report_ensemble(self, diagnostics: EnsembleDiagnostics) -> None:
        print(f"- q points: {diagnostics.qpoints}")
        print(
            f"- sampled modes: {diagnostics.sampled_modes}/{diagnostics.total_modes}, "
            f"imaginary={diagnostics.imaginary_modes}, excluded={diagnostics.excluded_modes}"
        )
        print(
            f"- maximum sampled atomic displacement: "
            f"{diagnostics.maximum_sampled_displacement:.8f} Å"
        )
        if diagnostics.maximum_displacement is None:
            print("- maximum displacement limit: disabled")
        else:
            print(
                f"- clipped atoms: {diagnostics.clipped_atoms}, "
                f"affected snapshots: {diagnostics.affected_snapshots}"
            )

    @staticmethod
    def _ordered(values, size: int, name: str) -> NDArray[np.float64]:
        if isinstance(values, Mapping):
            expected = set(range(size))
            if set(values) != expected:
                raise ValueError(f"{name} IDs must be exactly 0..{size - 1}")
            array = np.asarray([values[index] for index in range(size)], dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contain NaN or infinite values")
        return array


__all__ = ["SSCHA", "SSCHAIteration"]
