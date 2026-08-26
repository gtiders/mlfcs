"""Self-consistent harmonic sampling and FC2 fitting with native MLFCS components."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from numpy.typing import NDArray

from mlfcs.fitting import ForceConstantFitter
from mlfcs.force_constants.dense import expand_compact_fc2, lattice_fc2, replace_lattice_fc2
from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.force_constants.representation import (
    ForceConstants,
)
from mlfcs.physics.temperature import TemperatureSeriesResult, normalize_temperature_schedule
from mlfcs.sampling.harmonic import HarmonicSampler, SamplingState
from mlfcs.sampling.structures import _sample_perturbations

Progress = Callable[[int, int], None]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SSCHAIteration:
    """Immutable summary of one fitted effective harmonic Hamiltonian."""

    index: int
    sampling: Literal["cartesian", "canonical"]
    free_energy: float | None
    free_energy_error: float | None
    potential_energy: float | None
    harmonic_potential_energy: float
    qpoints: int | None
    total_modes: int | None
    sampled_modes: int | None
    excluded_modes: int | None
    imaginary_modes: int | None
    minimum_frequency_thz: float | None
    maximum_sampled_displacement: float
    clipped_atoms: int
    fitting_relative_force_error: float
    relative_force_constant_change: float | None
    raw_relative_force_constant_change: float | None


@dataclass(frozen=True, slots=True)
class SSCHAResult:
    """One temperature's self-consistent effective harmonic IFC result."""

    temperature: float
    force_constants: ForceConstants
    history: tuple[SSCHAIteration, ...]


class SSCHA:
    """Iterative effective-harmonic sampling driven by arbitrary ASE forces."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        reference: Atoms,
        cutoff: float | None,
        fitting_basis: Literal["taylor", "wick"] = "taylor",
        temperature: float | Sequence[float] = 300.0,
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
        initial_force_constants: ForceConstants | None = None,
        acoustic_sum_rule: bool = True,
        mixing: float = 1.0,
        continuation: bool = True,
    ) -> None:
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        temperatures = normalize_temperature_schedule(temperature)
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
        if not 0 < mixing <= 1:
            raise ValueError("mixing must be in (0, 1]")

        self.primitive = atoms.copy()
        self.primitive.wrap()
        from mlfcs.structure.relation import StructureRelation

        relation = StructureRelation.from_atoms(self.primitive, reference, tolerance=symprec)
        self._reference, self._index = relation.reference, relation.index
        self.supercell = self._index.supercell_matrix
        self.temperatures = temperatures
        self.temperature = temperatures[0]
        self.statistics = statistics
        self.snapshots = snapshots
        self.max_iterations = max_iterations
        self.initial_displacement = float(initial_displacement)
        self.random_seed = random_seed
        self.symprec = symprec
        self.cutoff = None if cutoff is None else float(cutoff)
        self.fitting_basis = fitting_basis
        self.cutoff_frequency = float(cutoff_frequency)
        self.imaginary_modes = imaginary_modes
        self.imaginary_tolerance = float(imaginary_tolerance)
        self.max_displacement = max_displacement
        self.acoustic_sum_rule = acoustic_sum_rule
        self.mixing = float(mixing)
        self.continuation = bool(continuation)
        self.history: list[SSCHAIteration] = []
        self._reference_energy: float | None = None
        self._fitter = ForceConstantFitter(
            self.primitive,
            self._reference,
            orders=(2,),
            cutoffs={2: self.cutoff},
            fitting_basis=fitting_basis,
            symprec=symprec,
        )
        self._active_compact: NDArray[np.float64] | None = None
        self._force_constants: ForceConstants | None = None
        if initial_force_constants is not None:
            if not isinstance(initial_force_constants, ForceConstants):
                raise TypeError("initial_force_constants must be a ForceConstants object")
            initial = realize_force_constants(
                initial_force_constants, self._reference, primitive=self.primitive
            )
            if 2 not in initial.orders:
                raise ValueError("initial_force_constants does not contain FC2")
            self._force_constants = initial
            self._active_compact = initial.materialize(2, max_bytes=None).copy()
        self._initial_force_constants = self._force_constants

    @property
    def force_constants(self) -> ForceConstants | None:
        """Return the complete effective FC2 after at least one fitted update."""
        return self._force_constants

    @property
    def supercell_atoms(self) -> Atoms:
        return self._reference.copy()

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    def sample(self) -> list[Atoms]:
        """Generate one displacement ensemble without evaluating forces."""
        structures, _, _, _ = self._sample_structures()
        return structures

    def _sample_structures(
        self,
    ) -> tuple[
        list[Atoms],
        NDArray[np.float64] | None,
        HarmonicSampler | None,
        Literal["cartesian", "canonical"],
    ]:
        """Draw one internal SSCHA ensemble in the reference atom order."""
        self._require_single_temperature()
        index = self.current_iteration
        if index > self.max_iterations:
            raise StopIteration("all requested SSCHA iterations are complete")

        count = self._snapshot_count()
        sampling_compact = None if self._active_compact is None else self._active_compact.copy()
        sampling_ensemble = None
        if sampling_compact is None:
            batch = _sample_perturbations(
                self._reference,
                snapshots=count,
                method="gaussian",
                displacement=self.initial_displacement,
                random_seed=self.random_seed,
            )
            displacement = batch.displacements
            sampling = "cartesian"
        else:
            assert self._force_constants is not None
            batch = _sample_perturbations(
                self._reference,
                snapshots=count,
                method="harmonic",
                force_constants=self._force_constants,
                temperature=self.temperature,
                statistics=self.statistics,
                cutoff_frequency=self.cutoff_frequency,
                imaginary_modes=self.imaginary_modes,
                imaginary_tolerance=self.imaginary_tolerance,
                max_displacement=self.max_displacement,
                random_seed=self._sampling_seed(index),
            )
            displacement = batch.displacements
            sampling_ensemble = batch.harmonic_sampler
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
        logger.info("SSCHA iteration %d/%d: %s sampling", index, self.max_iterations, sampling)
        if sampling_ensemble is not None:
            self._report_ensemble(sampling_ensemble.state)
        return structures, sampling_compact, sampling_ensemble, sampling

    def _fit_sampled_structures(
        self,
        snapshots: list[Atoms],
        forces: NDArray[np.float64],
        energies: NDArray[np.float64] | None,
        sampling_compact: NDArray[np.float64] | None,
        sampling_ensemble: HarmonicSampler | None,
        sampling: Literal["cartesian", "canonical"],
    ) -> SSCHAIteration:
        """Fit and mix FC2 from one internally sampled ensemble."""
        n_snapshots, n_atoms = len(snapshots), len(self._reference)
        if forces.shape != (n_snapshots, n_atoms, 3):
            raise ValueError(
                f"internal forces must have shape {(n_snapshots, n_atoms, 3)}, got {forces.shape}"
            )
        if energies is not None and energies.shape != (n_snapshots,):
            raise ValueError(f"energies must have shape {(n_snapshots,)}")

        for atoms, values in zip(snapshots, forces, strict=True):
            # One ASE calculator instance evaluates every snapshot.  Detach it
            # before fitting: otherwise FitDataset would see its final cached
            # force array for every structure instead of this snapshot's force.
            atoms.calc = None
            atoms.new_array("forces", np.asarray(values, dtype=float))
        gram = self._fitter.prepare_gram(
            snapshots,
            acoustic_sum_rule=self.acoustic_sum_rule,
        )
        fit = self._fitter.fit(gram, acoustic_sum_rule=self.acoustic_sum_rule)
        fitted_lattice = lattice_fc2(fit.force_constants)
        fitted_compact = fit.force_constants.materialize(2, max_bytes=None)
        raw_relative_change = None
        relative_change = None
        next_lattice = fitted_lattice
        if sampling_compact is not None:
            denominator = np.linalg.norm(sampling_compact)
            raw_relative_change = float(
                np.linalg.norm(fitted_compact - sampling_compact)
                / max(float(denominator), np.finfo(float).tiny)
            )
            if self._force_constants is None:
                raise RuntimeError("mixed SSCHA iteration is missing its previous exact FC2 state")
            previous_lattice = lattice_fc2(self._force_constants)
            if previous_lattice.keys() != fitted_lattice.keys():
                raise ValueError("SSCHA FC2 support changed between self-consistent iterations")
            next_lattice = {
                key: (1.0 - self.mixing) * previous_lattice[key] + self.mixing * fitted_lattice[key]
                for key in fitted_lattice
            }
            next_force_constants = replace_lattice_fc2(fit.force_constants, next_lattice)
            next_compact = next_force_constants.materialize(2, max_bytes=None)
            relative_change = float(
                np.linalg.norm(next_compact - sampling_compact)
                / max(float(denominator), np.finfo(float).tiny)
            )
        displacement = np.asarray(
            [atoms.positions - self._reference.positions for atoms in snapshots]
        )
        trial_compact = fitted_compact if sampling_compact is None else sampling_compact
        trial_full = expand_compact_fc2(trial_compact, self._reference)
        n_cells = self._index.n_cells
        harmonic_each = (
            np.einsum("ijab,mia,mjb->m", trial_full, displacement, displacement, optimize=True) / 2
        ) / n_cells
        free_energy = free_energy_error = potential_energy = None
        ensemble = sampling_ensemble or self._make_ensemble(trial_compact)
        if energies is not None and self._reference_energy is not None:
            potential_each = energies - self._reference_energy
            potential_energy = float(np.mean(potential_each) / n_cells)
            if sampling_compact is not None:
                correction = potential_each / n_cells - harmonic_each
                free_energy = float(ensemble.harmonic_free_energy() + np.mean(correction))
                free_energy_error = (
                    float(np.std(correction, ddof=1) / np.sqrt(len(correction)))
                    if len(correction) > 1
                    else float("nan")
                )
        sampling_state = None if sampling_ensemble is None else ensemble.state
        result = SSCHAIteration(
            index=self.current_iteration,
            sampling=sampling,
            free_energy=free_energy,
            free_energy_error=free_energy_error,
            potential_energy=potential_energy,
            harmonic_potential_energy=float(np.mean(harmonic_each)),
            qpoints=None if sampling_state is None else sampling_state.qpoints,
            total_modes=None if sampling_state is None else sampling_state.total_modes,
            sampled_modes=None if sampling_state is None else sampling_state.sampled_modes,
            excluded_modes=None if sampling_state is None else sampling_state.excluded_modes,
            imaginary_modes=None if sampling_state is None else sampling_state.imaginary_modes,
            minimum_frequency_thz=(
                None if sampling_state is None else sampling_state.minimum_frequency_thz
            ),
            maximum_sampled_displacement=(
                float(np.max(np.linalg.norm(displacement, axis=2)))
                if sampling_state is None
                else sampling_state.maximum_sampled_displacement
            ),
            clipped_atoms=0 if sampling_state is None else sampling_state.clipped_atoms,
            fitting_relative_force_error=fit.training_relative_force_error,
            relative_force_constant_change=relative_change,
            raw_relative_force_constant_change=raw_relative_change,
        )
        self._force_constants = replace_lattice_fc2(
            fit.force_constants,
            next_lattice,
            metadata={
                "method": "sscha",
                "temperature": self.temperature,
                "statistics": self.statistics,
                "mixing": self.mixing,
            },
        )
        self._active_compact = self._force_constants.materialize(2, max_bytes=None).copy()
        self.history.append(result)
        logger.info(
            f"- Fitting relative force error: {100 * result.fitting_relative_force_error:.6f} %"
        )
        if relative_change is not None:
            logger.info("Relative FC2 change: %.6e", relative_change)
            if self.mixing != 1.0:
                assert raw_relative_change is not None
                logger.info("Raw fitted FC2 change: %.6e", raw_relative_change)
        if result.free_energy is not None:
            logger.info(
                f"- Variational free-energy estimate: {result.free_energy:.10e} "
                f"+/- {result.free_energy_error:.3e} eV/primitive cell"
            )
        return result

    def step(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHAIteration:
        self._require_single_temperature()
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        if calculate_free_energy and self._reference_energy is None:
            equilibrium = self.supercell_atoms
            equilibrium.calc = calculator
            self._reference_energy = float(equilibrium.get_potential_energy())
        structures, sampling_compact, sampling_ensemble, sampling = self._sample_structures()
        forces = np.empty((len(structures), len(structures[0]), 3))
        energies = np.empty(len(structures)) if calculate_free_energy else None
        for index, atoms in enumerate(structures):
            atoms.calc = calculator
            forces[index] = atoms.get_forces()
            if energies is not None:
                energies[index] = atoms.get_potential_energy()
            if progress is not None:
                progress(index + 1, len(structures))
        return self._fit_sampled_structures(
            structures,
            forces,
            energies,
            sampling_compact,
            sampling_ensemble,
            sampling,
        )

    def run(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHAResult | TemperatureSeriesResult[SSCHAResult]:
        """Run one temperature or an automatically sorted temperature schedule."""
        if len(self.temperatures) > 1:
            return self._run_temperature_schedule(
                calculator,
                progress=progress,
                calculate_free_energy=calculate_free_energy,
            )
        while self.current_iteration <= self.max_iterations:
            self.step(
                calculator,
                progress=progress,
                calculate_free_energy=calculate_free_energy,
            )
        return self._result()

    def _run_temperature_schedule(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None,
        calculate_free_energy: bool,
    ) -> TemperatureSeriesResult[SSCHAResult]:
        previous = self._initial_force_constants
        results: list[SSCHAResult] = []
        for schedule_index, temperature in enumerate(self.temperatures):
            initial = (
                previous
                if self.continuation or schedule_index == 0
                else self._initial_force_constants
            )
            child = SSCHA(
                self.primitive,
                reference=self._reference,
                cutoff=self.cutoff,
                fitting_basis=self.fitting_basis,
                temperature=temperature,
                statistics=self.statistics,
                snapshots=self.snapshots,
                max_iterations=self.max_iterations,
                initial_displacement=self.initial_displacement,
                random_seed=self._temperature_seed(schedule_index),
                symprec=self.symprec,
                cutoff_frequency=self.cutoff_frequency,
                imaginary_modes=self.imaginary_modes,
                imaginary_tolerance=self.imaginary_tolerance,
                max_displacement=self.max_displacement,
                initial_force_constants=initial,
                acoustic_sum_rule=self.acoustic_sum_rule,
                mixing=self.mixing,
                continuation=False,
            )
            result = child.run(
                calculator,
                progress=progress,
                calculate_free_energy=calculate_free_energy,
            )
            assert isinstance(result, SSCHAResult)
            results.append(result)
            if self.continuation:
                previous = result.force_constants
        return TemperatureSeriesResult(self.temperatures, tuple(results), self.continuation)

    def _result(self) -> SSCHAResult:
        self._require_single_temperature()
        if self._force_constants is None:
            raise RuntimeError("no force constants are available")
        return SSCHAResult(self.temperature, self._force_constants, tuple(self.history))

    def _snapshot_count(self) -> int:
        if self.snapshots != "auto":
            return self.snapshots
        equations = max(3 * len(self._reference) - 3, 1)
        return max(1, int(np.ceil(4 * self._fitter.n_parameters / equations)))

    def _make_ensemble(self, compact: np.ndarray) -> HarmonicSampler:
        return HarmonicSampler(
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

    def _temperature_seed(self, schedule_index: int) -> int | None:
        if self.random_seed is None:
            return None
        sequence = np.random.SeedSequence([self.random_seed, schedule_index])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _require_single_temperature(self) -> None:
        if len(self.temperatures) != 1:
            raise RuntimeError(
                "step operations require one temperature; call run(calculator) for a temperature schedule"
            )

    def _report_ensemble(self, diagnostics: SamplingState) -> None:
        logger.info("q points: %d", diagnostics.qpoints)
        logger.info(
            f"- sampled modes: {diagnostics.sampled_modes}/{diagnostics.total_modes}, "
            f"imaginary={diagnostics.imaginary_modes}, excluded={diagnostics.excluded_modes}"
        )
        logger.info(
            f"- maximum sampled atomic displacement: "
            f"{diagnostics.maximum_sampled_displacement:.8f} Å"
        )
        if diagnostics.maximum_displacement is None:
            logger.info("maximum displacement limit: disabled")
        else:
            logger.info(
                f"- clipped atoms: {diagnostics.clipped_atoms}, "
                f"affected snapshots: {diagnostics.affected_snapshots}"
            )
