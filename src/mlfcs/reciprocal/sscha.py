"""Fixed-centroid stochastic self-consistent harmonic force matching."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log_error import get_logger
from mlfcs.fitting import FitSystem
from mlfcs.force_constants import ForceConstants
from mlfcs.reciprocal.ensemble import HarmonicEnsemble, UnstableTrialError
from mlfcs.reciprocal.grid import QGrid, QStars
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)

Status = Literal["converged", "max_iterations", "insufficient_samples", "unstable"]


@dataclass(frozen=True, slots=True)
class SSCHAStep:
    """One canonical trial update, with physical and sampling diagnostics."""

    index: int
    pairs: int
    frequency_change_thz: float
    uncertainty_thz: float
    minimum_frequency_thz: float
    mixing: float
    fit_rmse: float
    asr_residual: float
    mean_force_norm: float
    free_energy: float | None
    free_energy_error: float | None
    phase: Literal["canonical"] = "canonical"


@dataclass(frozen=True, slots=True)
class SSCHAResult:
    """Effective FC2 of one fixed-centroid temperature calculation."""

    temperature: float
    fc2: ForceConstants
    history: tuple[SSCHAStep, ...]
    status: Status
    n_qpoints: int
    n_irreducible: int
    mapping_fingerprint: str
    seed: int
    message: str | None = None
    bootstrap_displacement: float | None = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"


class SSCHAContinuationError(RuntimeError):
    """A temperature schedule cannot warm-start from a failed result."""

    def __init__(self, result: SSCHAResult, completed: list[SSCHAResult]) -> None:
        super().__init__(
            f"SSCHA stopped at {result.temperature:g} K with status {result.status}; "
            "lower temperatures were not evaluated"
        )
        self.result = result
        self.completed = tuple(completed)


def _temperature_seed(seed: int, temperature: float, iteration: int) -> np.random.Generator:
    bits = int.from_bytes(struct.pack(">d", temperature), "big")
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed & 0xFFFFFFFF, seed >> 32, bits & 0xFFFFFFFF, bits >> 32, iteration]
        )
    )


class SSCHA:
    """Sample a stable trial FC2, calculate ASE forces and refit FC2 only.

    The centroid and cell are fixed. An unstable input can be diagnosed, but
    canonical sampling never replaces its negative curvatures by absolute
    values or discards those modes. Only ``run`` and ``run_many`` evaluate data;
    no external-force or offline sample ingestion API is provided.
    """

    def __init__(
        self,
        mapping: ClusterMap,
        mesh: object,
        *,
        initial: ForceConstants | None = None,
        statistics: Literal["quantum", "classical"] = "quantum",
        seed: int | None = None,
        bootstrap_displacement: float | None = None,
    ) -> None:
        if not isinstance(mapping, ClusterMap):
            raise TypeError("mapping must be a ClusterMap")
        if mapping.space.orders != (2,):
            raise ValueError("SSCHA requires a ClusterMap with FC2 only")
        mapping.rank_info(2).require_full()
        if statistics not in ("quantum", "classical"):
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if bootstrap_displacement is not None and (
            not np.isfinite(bootstrap_displacement) or bootstrap_displacement <= 0.0
        ):
            raise ValueError("bootstrap_displacement must be positive and finite in angstrom")
        if seed is not None and (not isinstance(seed, int) or seed < 0 or seed >= 2**64):
            raise ValueError("seed must be an integer between 0 and 2**64 - 1")
        if initial is not None:
            if not isinstance(initial, ForceConstants) or initial.orders != (2,):
                raise ValueError("initial must be an FC2-only ForceConstants model")
            if initial.space.fingerprint != mapping.space.fingerprint or not np.array_equal(
                initial.space.primitive.masses, mapping.space.primitive.masses
            ):
                raise ValueError("initial FC2 uses a different cluster space or atomic masses")
        if isinstance(mesh, QGrid):
            grid = mesh
        else:
            values = np.asarray(mesh)
            grid = QGrid.from_matrix(np.diag(values) if values.shape == (3,) else mesh)
        if not np.array_equal(grid.matrix, np.asarray(mapping.supercell.matrix, dtype=np.int64)):
            raise ValueError("SSCHA reciprocal mesh must equal the declared supercell matrix")
        self.mapping = mapping
        self.stars = QStars.from_symmetry(grid, mapping.space.symmetry, time_reversal=True)
        self.initial = initial
        self.statistics = statistics
        self.seed = (
            int(seed)
            if seed is not None
            else int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        )
        self.bootstrap_displacement = bootstrap_displacement

    def _ensemble(self, model: ForceConstants, temperature: float) -> HarmonicEnsemble:
        return HarmonicEnsemble(
            model, self.mapping.supercell, self.stars, temperature, statistics=self.statistics
        )

    def _evaluate(
        self,
        displacements: np.ndarray,
        calculator: Calculator,
        *,
        energy: bool,
    ) -> tuple[list[Atoms], np.ndarray | None, float]:
        supercell = self.mapping.supercell
        positions = supercell.scaled_positions @ supercell.cell
        # FitSystem reads ASE geometry through a minimum image. A Gaussian has
        # unbounded tails, so a too-small cell can wrap one generated atomic
        # displacement into a different Taylor expansion point. Refuse that
        # sample instead of silently fitting its wrapped image.
        flat = np.asarray(displacements, dtype=np.float64).reshape(-1, 3)
        minimum, _ = PeriodicGeometry(supercell.cell).minimum_image(flat)
        shifts = np.rint((flat - minimum) @ np.linalg.inv(supercell.cell))
        if np.any(shifts != 0.0):
            raise ValueError(
                "a sampled displacement crosses the supercell minimum-image boundary; "
                "increase the supercell or choose a better stable trial FC2"
            )
        structures: list[Atoms] = []
        energies = [] if energy else None
        even_forces = []
        for pair, positive in enumerate(displacements):
            pair_forces = []
            for sign in (1, -1):
                atoms = Atoms(
                    numbers=supercell.numbers,
                    positions=positions + sign * positive,
                    cell=supercell.cell,
                    pbc=True,
                )
                atoms.set_masses(supercell.primitive.masses[supercell.sites])
                atoms.calc = calculator
                forces = np.asarray(atoms.get_forces(), dtype=np.float64).copy()
                if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
                    raise ValueError(f"ASE calculator returned invalid forces for pair {pair}")
                potential = None
                if energy:
                    potential = float(atoms.get_potential_energy())
                    if not np.isfinite(potential):
                        raise ValueError(f"ASE calculator returned invalid energy for pair {pair}")
                    assert energies is not None
                    energies.append(potential)
                atoms.calc = SinglePointCalculator(atoms, forces=forces, energy=potential)
                structures.append(atoms)
                pair_forces.append(forces)
            even_forces.append(0.5 * (pair_forces[0] + pair_forces[1]))
        mean_force_norm = float(np.linalg.norm(np.mean(even_forces, axis=0)))
        return structures, None if energies is None else np.asarray(energies), mean_force_norm

    def _fit(
        self, structures: list[Atoms], *, split: bool
    ) -> tuple[ForceConstants, float, float, tuple[ForceConstants, ForceConstants] | None]:
        pairs = len(structures) // 2
        half = pairs // 2
        if split:
            first = FitSystem.from_atoms(self.mapping, structures[: 2 * half])
            second = FitSystem.from_atoms(self.mapping, structures[2 * half :])
            system = first + second
        else:
            system = FitSystem.from_atoms(self.mapping, structures)
        model = system.force_constants(system.solve())
        projection = model.enforce_asr(orders=(2,))
        model = projection.force_constants
        halves = None
        if split:
            try:
                halves = tuple(
                    part.force_constants(part.solve()).enforce_asr(orders=(2,)).force_constants
                    for part in (first, second)
                )
            except (RuntimeError, ValueError):
                halves = None
        return model, system.rmse(model.parameters()), projection.report(2).relative_after, halves

    def _bootstrap(
        self,
        calculator: Calculator,
        temperature: float,
        pairs: int,
    ) -> ForceConstants:
        displacement = self.bootstrap_displacement or 0.01
        rng = _temperature_seed(self.seed, temperature, 0)
        supercell = self.mapping.supercell
        masses = supercell.primitive.masses[supercell.sites]
        values = rng.standard_normal((pairs, len(masses), 3)) * displacement
        centroid = np.einsum("a,pad->pd", masses, values) / np.sum(masses)
        values -= centroid[:, None, :]
        structures, _, _ = self._evaluate(values, calculator, energy=False)
        model, _, _, _ = self._fit(structures, split=False)
        try:
            self._ensemble(model, temperature)
        except UnstableTrialError as error:
            raise UnstableTrialError(
                f"Cartesian bootstrap at displacement {displacement:g} angstrom did not "
                f"produce a stable trial: {error}"
            ) from error
        logger.info("SSCHA bootstrap produced a stable trial from %d displacement pairs", pairs)
        return model

    def _free_energy(
        self,
        ensemble: HarmonicEnsemble,
        displacements: np.ndarray,
        energies: np.ndarray,
        equilibrium_energy: float,
    ) -> tuple[float, float]:
        from mlfcs.fitting.design import ForceDesign

        design = ForceDesign(self.mapping)
        parameters = ensemble.model.parameters()
        harmonic = []
        for value in displacements:
            force = design.matrix(value) @ parameters
            potential = -0.5 * float(value.reshape(-1) @ force)
            harmonic.extend((potential, potential))
        corrections = (energies - equilibrium_energy - np.asarray(harmonic)) / self.stars.grid.size
        pairs = corrections.reshape(-1, 2).mean(axis=1)
        error = float(np.std(pairs, ddof=1) / np.sqrt(len(pairs))) if len(pairs) > 1 else 0.0
        return ensemble.free_energy() + float(np.mean(pairs)), error

    def run(
        self,
        temperature: float,
        calculator: Calculator,
        *,
        pairs: int = 128,
        max_pairs: int | None = None,
        max_iterations: int = 20,
        tol_thz: float = 0.01,
        mixing: float = 0.5,
        max_backtracks: int = 8,
        energy: bool = False,
        start: ForceConstants | None = None,
    ) -> SSCHAResult:
        """Run one temperature using only freshly calculated ASE forces."""
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.statistics == "classical" and temperature == 0.0:
            raise ValueError("classical SSCHA at zero temperature is not identifiable")
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        if not isinstance(pairs, int) or pairs < 2:
            raise ValueError("pairs must be an integer of at least two")
        maximum = pairs if max_pairs is None else int(max_pairs)
        if maximum < pairs:
            raise ValueError("max_pairs must be at least pairs")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tol_thz) or tol_thz <= 0.0:
            raise ValueError("tol_thz must be positive and finite")
        if not np.isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("mixing must lie in (0, 1]")
        if not isinstance(max_backtracks, int) or max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        if start is not None and (
            not isinstance(start, ForceConstants)
            or start.orders != (2,)
            or start.space.fingerprint != self.mapping.space.fingerprint
            or not np.array_equal(start.space.primitive.masses, self.mapping.space.primitive.masses)
        ):
            raise ValueError("start must be FC2 on this cluster space and with these masses")

        model = start if start is not None else self.initial
        bootstrapped = False
        if model is None:
            model = self._bootstrap(calculator, temperature, pairs)
            bootstrapped = True
        else:
            model = model.enforce_asr(orders=(2,)).force_constants
            try:
                self._ensemble(model, temperature)
            except UnstableTrialError:
                if self.bootstrap_displacement is None:
                    raise
                model = self._bootstrap(calculator, temperature, pairs)
                bootstrapped = True

        logger.info(
            "SSCHA run started: temperature=%.6g K mesh_points=%d irreducible=%d "
            "statistics=%s seed=%d pairs=%d max_pairs=%d max_iterations=%d "
            "tol_thz=%.6g mixing=%.6g bootstrap=%s energy=%s",
            temperature,
            self.stars.grid.size,
            len(self.stars.representatives),
            self.statistics,
            self.seed,
            pairs,
            maximum,
            max_iterations,
            tol_thz,
            mixing,
            bootstrapped,
            energy,
        )

        equilibrium_energy = None
        if energy:
            supercell = self.mapping.supercell
            atoms = Atoms(
                numbers=supercell.numbers,
                positions=supercell.scaled_positions @ supercell.cell,
                cell=supercell.cell,
                pbc=True,
            )
            atoms.set_masses(supercell.primitive.masses[supercell.sites])
            atoms.calc = calculator
            equilibrium_energy = float(atoms.get_potential_energy())
            if not np.isfinite(equilibrium_energy):
                raise ValueError("ASE calculator returned invalid equilibrium energy")

        history: list[SSCHAStep] = []
        status: Status = "max_iterations"
        message = None
        count = pairs
        for iteration in range(max_iterations):
            started = perf_counter()
            ensemble = self._ensemble(model, temperature)
            prepared = perf_counter()
            displacements = ensemble.sample(count, seed=self.seed, iteration=iteration + 1)
            sampled = perf_counter()
            structures, energies, mean_force = self._evaluate(
                displacements, calculator, energy=energy
            )
            evaluated = perf_counter()
            fitted, rmse, asr_residual, halves = self._fit(structures, split=True)
            fitted_at = perf_counter()
            uncertainty = float("inf")
            if halves is not None:
                try:
                    first = self._ensemble(halves[0], temperature)
                    second = self._ensemble(halves[1], temperature)
                    uncertainty = first.frequency_change_thz(second)
                    half = count // 2
                    uncertainty *= np.sqrt(half * (count - half)) / count
                except UnstableTrialError:
                    pass

            previous = model.coefficients[2]
            difference = fitted.coefficients[2] - previous
            accepted = None
            last_error = None
            for backtracks in range(max_backtracks + 1):
                coefficient = mixing / 2**backtracks
                candidate = ForceConstants(
                    self.mapping.space, {2: previous + coefficient * difference}
                )
                try:
                    next_ensemble = self._ensemble(candidate, temperature)
                    accepted = coefficient
                    break
                except UnstableTrialError as error:
                    last_error = error
                    continue
            if accepted is None:
                status = "unstable"
                message = f"no stable FC2 update after {max_backtracks} backtracks: {last_error}"
                break
            change = ensemble.frequency_change_thz(next_ensemble)
            free_energy = free_energy_error = None
            if energies is not None:
                assert equilibrium_energy is not None
                free_energy, free_energy_error = self._free_energy(
                    ensemble, displacements, energies, equilibrium_energy
                )
            history.append(
                SSCHAStep(
                    iteration + 1,
                    count,
                    change,
                    uncertainty,
                    next_ensemble.minimum_frequency_thz,
                    accepted,
                    rmse,
                    asr_residual,
                    mean_force,
                    free_energy,
                    free_energy_error,
                )
            )
            model = candidate
            logger.info(
                "SSCHA %.1f K iteration %d: %d/%d q points, change %.6g THz, "
                "sampling error %.6g THz, pairs %d, fit RMSE %.6g eV/Å, "
                "ASR residual %.6g, mean-force norm %.6g eV/Å, mixing %.5g, "
                "free energy %s; trial %.3f s, draw %.3f s, ASE %.3f s, "
                "fit %.3f s, total %.3f s",
                temperature,
                iteration + 1,
                len(self.stars.representatives),
                self.stars.grid.size,
                change,
                uncertainty,
                count,
                rmse,
                asr_residual,
                mean_force,
                accepted,
                "not requested" if free_energy is None else f"{free_energy:.10g} eV",
                prepared - started,
                sampled - prepared,
                evaluated - sampled,
                fitted_at - evaluated,
                perf_counter() - started,
            )
            if change <= tol_thz and uncertainty <= tol_thz:
                status = "converged"
                break
            if uncertainty > tol_thz and count < maximum:
                count = min(2 * count, maximum)
            elif change <= tol_thz and uncertainty > tol_thz and count == maximum:
                status = "insufficient_samples"
                message = (
                    f"frequency uncertainty {uncertainty:.6g} THz exceeds "
                    f"tol_thz {tol_thz:.6g} with {count} pairs"
                )
                break

        result = SSCHAResult(
            temperature,
            model,
            tuple(history),
            status,
            self.stars.grid.size,
            len(self.stars.representatives),
            self.mapping.fingerprint,
            self.seed,
            message,
            (self.bootstrap_displacement or 0.01) if bootstrapped else None,
        )
        last = result.history[-1] if result.history else None
        logger.info(
            "SSCHA run complete: temperature=%.6g K status=%s iterations=%d pairs=%d "
            "fc2_fingerprint=%s%s",
            result.temperature,
            result.status,
            len(result.history),
            last.pairs if last is not None else 0,
            result.fc2.fingerprint,
            f" message={result.message}" if result.message else "",
        )
        return result

    def run_many(
        self,
        temperatures: Sequence[float],
        calculator: Calculator,
        **kwargs: object,
    ) -> list[SSCHAResult]:
        """Solve high to low temperature; return results in ascending order."""
        values = tuple(float(value) for value in temperatures)
        if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("temperatures must be nonempty, finite and nonnegative")
        if any(right <= left for left, right in pairwise(values)):
            raise ValueError("temperatures must be strictly increasing")
        logger.info("SSCHA temperature schedule: %s K (high to low execution)", values)
        if "start" in kwargs:
            raise TypeError("run_many manages the starting FC2; do not pass start")
        completed = []
        start = None
        for temperature in reversed(values):
            result = self.run(temperature, calculator, start=start, **kwargs)
            completed.append(result)
            if not result.converged and temperature != values[0]:
                raise SSCHAContinuationError(result, completed)
            start = result.fc2
        completed.reverse()
        return completed


__all__ = ["SSCHA", "SSCHAContinuationError", "SSCHAResult", "SSCHAStep"]
