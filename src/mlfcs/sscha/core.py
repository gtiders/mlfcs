"""Stochastic self-consistent harmonic approximation with ASE force providers.

This module deliberately forms an optional boundary around phonopy and symfc.
The finite-difference force-constant implementation in the rest of mlfcs does
not import either package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from numpy.typing import ArrayLike, NDArray

Progress = Callable[[int, int], None]
ForceInput = NDArray[np.floating] | Sequence[ArrayLike] | Mapping[int, ArrayLike]


def _optional_imports():
    try:
        from phonopy import Phonopy
        from phonopy.harmonic.force_constants import compact_fc_to_full_fc
        from phonopy.structure.atoms import PhonopyAtoms
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without extra
        raise ModuleNotFoundError(
            "SSCHA requires the optional dependencies; install with `uv sync --extra sscha`."
        ) from exc
    return Phonopy, PhonopyAtoms, compact_fc_to_full_fc


def _to_phonopy(atoms: Atoms):
    _, PhonopyAtoms, _ = _optional_imports()
    masses = atoms.get_masses() if atoms.has("masses") else None
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        positions=atoms.get_positions(),
        masses=masses,
    )


def _to_ase(atoms) -> Atoms:
    return Atoms(
        symbols=list(atoms.symbols),
        cell=np.asarray(atoms.cell),
        scaled_positions=np.asarray(atoms.scaled_positions),
        masses=None if atoms.masses is None else np.asarray(atoms.masses),
        pbc=True,
    )


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


class SSCHA:
    """Phonopy-style iterative SSCHA driven by arbitrary ASE forces.

    Iteration zero fits an initial FC2 from small random Cartesian
    displacements. Each of ``max_iterations`` subsequent iterations samples
    the canonical harmonic ensemble of the latest FC2 and refits it with
    symfc. Energies are optional unless free energies are requested.
    """

    def __init__(
        self,
        atoms: Atoms,
        *,
        supercell: Sequence[int] | Sequence[Sequence[int]] = (2, 2, 2),
        temperature: float = 300.0,
        snapshots: int | Literal["auto"] = 1000,
        max_iterations: int = 10,
        initial_displacement: float = 0.01,
        random_seed: int | None = None,
        symprec: float = 1e-5,
        cutoff_frequency: float | None = None,
        max_displacement: float | None = None,
        initial_force_constants: ArrayLike | None = None,
        log_level: int = 0,
    ) -> None:
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if snapshots != "auto" and snapshots < 1:
            raise ValueError("snapshots must be positive or 'auto'")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if initial_displacement <= 0:
            raise ValueError("initial_displacement must be positive")

        Phonopy, _, compact_fc_to_full_fc = _optional_imports()
        self._phonon = Phonopy(
            _to_phonopy(atoms),
            supercell_matrix=np.asarray(supercell, dtype=int),
            symprec=symprec,
            log_level=max(0, log_level - 1),
        )
        if initial_force_constants is not None:
            fc = np.asarray(initial_force_constants, dtype=float)
            if fc.ndim != 4 or fc.shape[-2:] != (3, 3):
                raise ValueError("initial_force_constants must have shape (n, N, 3, 3)")
            if fc.shape[0] != fc.shape[1]:
                fc = compact_fc_to_full_fc(self._phonon.primitive, fc)
            self._phonon.force_constants = fc

        self.temperature = float(temperature)
        self.snapshots = snapshots
        self.max_iterations = max_iterations
        self.initial_displacement = float(initial_displacement)
        self.random_seed = random_seed
        self.cutoff_frequency = cutoff_frequency
        self.max_displacement = max_displacement
        self.log_level = log_level
        self.history: list[SSCHAIteration] = []
        self._prepared_index: int | None = None
        self._sampling_fc: NDArray[np.float64] | None = None
        self._reference_energy: float | None = None

    @property
    def phonopy(self):
        """Underlying Phonopy object for advanced phonon analysis."""
        return self._phonon

    @property
    def force_constants(self) -> NDArray[np.float64] | None:
        fc = self._phonon.force_constants
        return None if fc is None else np.asarray(fc)

    @property
    def supercell_atoms(self) -> Atoms:
        return _to_ase(self._phonon.supercell)

    @property
    def current_iteration(self) -> int:
        """Index of the next fit (zero is Cartesian initialization)."""
        return len(self.history)

    def sow(self) -> list[Atoms]:
        """Create and return snapshots for the next iteration in reap order."""
        index = self.current_iteration
        if index > self.max_iterations:
            raise StopIteration("all requested SSCHA iterations are complete")
        if self._prepared_index == index:
            cells = self._phonon.supercells_with_displacements
            assert cells is not None
            return [self._tag_snapshot(cell, i, index) for i, cell in enumerate(cells)]

        fc = self.force_constants
        self._sampling_fc = None if fc is None else fc.copy()
        if fc is None:
            self._phonon.generate_displacements(
                distance=self.initial_displacement,
                number_of_snapshots=self.snapshots,
                random_seed=self.random_seed,
            )
            sampling = "cartesian"
        else:
            # generate_displacements replaces the dataset and invalidates FC2.
            self._phonon.generate_displacements(
                number_of_snapshots=self.snapshots,
                temperature=self.temperature,
                random_seed=self.random_seed,
                cutoff_frequency=self.cutoff_frequency,
                max_distance=self.max_displacement,
            )
            self._phonon.force_constants = self._sampling_fc
            sampling = "canonical"
        self._prepared_index = index
        if self.log_level:
            print(f"[SSCHA {index}/{self.max_iterations}] {sampling} sampling")
        cells = self._phonon.supercells_with_displacements
        assert cells is not None
        return [self._tag_snapshot(cell, i, index) for i, cell in enumerate(cells)]

    def reap(
        self,
        forces: ForceInput,
        *,
        energies: ArrayLike | Mapping[int, float] | None = None,
        reference_energy: float | None = None,
    ) -> SSCHAIteration:
        """Fit effective FC2 with symfc from forces matching :meth:`sow`."""
        snapshots = self.sow()
        n_snapshots, n_atoms = len(snapshots), len(self._phonon.supercell)
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

        displacements = np.asarray(self._phonon.displacements)
        sampling_fc = self._sampling_fc
        self._phonon.forces = force_array
        if energy_array is not None:
            self._phonon.supercell_energies = energy_array
        self._phonon.produce_force_constants(
            fc_calculator="symfc",
            calculate_full_force_constants=True,
            show_drift=False,
            fc_calculator_log_level=max(0, self.log_level - 1),
        )
        fc = np.asarray(self._phonon.force_constants).copy()
        harmonic_each = np.einsum("ijkl,mik,mjl->m", fc, displacements, displacements) / 2
        free_energy = free_energy_error = potential_energy = None
        if energy_array is not None and self._reference_energy is not None:
            potential_each = energy_array - self._reference_energy
            potential_energy = float(np.mean(potential_each))
            free_energy, free_energy_error = self._free_energy(potential_each, harmonic_each)
        result = SSCHAIteration(
            index=self.current_iteration,
            sampling="cartesian" if sampling_fc is None else "canonical",
            force_constants=fc,
            free_energy=free_energy,
            free_energy_error=free_energy_error,
            potential_energy=potential_energy,
            harmonic_potential_energy=float(np.mean(harmonic_each)),
        )
        self.history.append(result)
        self._prepared_index = None
        self._sampling_fc = None
        return result

    def step(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHAIteration:
        """Evaluate one iteration serially with a user-owned ASE calculator."""
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        if calculate_free_energy and self._reference_energy is None:
            equilibrium = self.supercell_atoms
            equilibrium.calc = calculator
            self._reference_energy = float(equilibrium.get_potential_energy())
        structures = self.sow()
        forces = np.empty((len(structures), len(structures[0]), 3))
        energies = np.empty(len(structures)) if calculate_free_energy else None
        for i, atoms in enumerate(structures):
            atoms.calc = calculator
            forces[i] = atoms.get_forces()
            if energies is not None:
                energies[i] = atoms.get_potential_energy()
            if progress is not None:
                progress(i + 1, len(structures))
        return self.reap(forces, energies=energies)

    def run(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        calculate_free_energy: bool = True,
    ) -> SSCHA:
        """Run initialization and all requested self-consistent iterations."""
        while self.current_iteration <= self.max_iterations:
            self.step(
                calculator,
                progress=progress,
                calculate_free_energy=calculate_free_energy,
            )
        return self

    def averaged_force_constants(self, last: int) -> NDArray[np.float64]:
        """Average FC2 over the last ``last`` completed iterations."""
        if last < 1 or not self.history:
            raise ValueError("last must be positive and at least one iteration must exist")
        return np.mean([item.force_constants for item in self.history[-last:]], axis=0)

    def use_average(self, last: int) -> NDArray[np.float64]:
        """Set and return the last-iteration average as the active FC2."""
        fc = self.averaged_force_constants(last)
        self._phonon.force_constants = fc
        return fc

    def write(self, target: str | Path, *, format: Literal["text", "hdf5"] = "hdf5") -> None:
        """Write the active full FC2 using phonopy's native writers."""
        if self.force_constants is None:
            raise RuntimeError("no force constants are available")
        if format == "text":
            from phonopy.file_IO import write_FORCE_CONSTANTS

            write_FORCE_CONSTANTS(self.force_constants, filename=str(target))
        elif format == "hdf5":
            from phonopy.file_IO import write_force_constants_to_hdf5

            write_force_constants_to_hdf5(self.force_constants, filename=str(target))
        else:
            raise ValueError("format must be 'text' or 'hdf5'")

    def _free_energy(
        self, potential_each: NDArray[np.float64], harmonic_each: NDArray[np.float64]
    ) -> tuple[float, float]:
        from phonopy.physical_units import get_physical_units

        self._phonon.run_mesh(mesh=100.0)
        self._phonon.run_thermal_properties(temperatures=[self.temperature])
        thermal = self._phonon.thermal_properties
        assert thermal is not None
        harmonic_free_energy = thermal.free_energy[0] / get_physical_units().EvTokJmol
        n_cells = len(self._phonon.supercell) / len(self._phonon.primitive)
        correction = (potential_each - harmonic_each) / n_cells
        error = (
            float(np.std(correction, ddof=1) / np.sqrt(len(correction)))
            if len(correction) > 1
            else float("nan")
        )
        return float(harmonic_free_energy + np.mean(correction)), error

    @staticmethod
    def _ordered(values, size: int, name: str) -> NDArray[np.float64]:
        if isinstance(values, Mapping):
            expected = set(range(size))
            if set(values) != expected:
                raise ValueError(f"{name} IDs must be exactly 0..{size - 1}")
            array = np.asarray([values[i] for i in range(size)], dtype=float)
        else:
            array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contain NaN or infinite values")
        return array

    @staticmethod
    def _tag_snapshot(cell, snapshot: int, iteration: int) -> Atoms:
        atoms = _to_ase(cell)
        atoms.info.update(
            mlfcs_sscha_iteration=iteration,
            mlfcs_configuration_id=snapshot,
        )
        return atoms
