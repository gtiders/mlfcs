"""Maintenance utility to regenerate the AlN second-order phonopy validation fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from phonopy import Phonopy, load
from phonopy.structure.atoms import PhonopyAtoms
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import ForceConstantCalculation
from mlfcs.core.geometry import make_supercell

SUPERCELL = (2, 2, 2)
DISPLACEMENT = 0.01
CUTOFF_MARGIN = 1e-6


def _ase_atoms(atoms: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=atoms.symbols,
        scaled_positions=atoms.scaled_positions,
        cell=atoms.cell,
        pbc=True,
    )


def generate_fixture(dataset: Path, potential: Path, target: Path) -> None:
    """Generate captured MLFCS forces and an independent phonopy FC2."""
    source = load(dataset, produce_fc=False)
    unitcell_ph = source.unitcell
    unitcell = _ase_atoms(unitcell_ph)
    calculator = PolymlpASECalculator(pot=str(potential))
    trial_supercell, _ = make_supercell(unitcell, SUPERCELL)
    maximum_mic_distance = float(trial_supercell.get_all_distances(mic=True).max())
    full_supercell_cutoff = maximum_mic_distance + CUTOFF_MARGIN

    calculation = ForceConstantCalculation(
        unitcell,
        order=2,
        supercell=SUPERCELL,
        cutoff=full_supercell_cutoff,
        displacement=DISPLACEMENT,
        jax_platform="cpu",
        report_cutoff=False,
    )
    mlfcs_forces = calculation.evaluate(calculator)

    phonon = Phonopy(unitcell_ph, supercell_matrix=np.diag(SUPERCELL))
    phonon.generate_displacements(distance=DISPLACEMENT, is_plusminus=True)
    phonopy_forces = []
    for displaced in phonon.supercells_with_displacements:
        atoms = _ase_atoms(displaced)
        atoms.calc = calculator
        phonopy_forces.append(atoms.get_forces())
    phonon.forces = np.asarray(phonopy_forces)
    phonon.produce_force_constants(
        calculate_full_force_constants=True,
        fc_calculator="traditional",
        show_drift=False,
    )
    phonopy_supercell = _ase_atoms(phonon.supercell)
    phonopy_fc2 = np.asarray(phonon.force_constants).copy()
    phonon.symmetrize_force_constants(level=3, show_drift=False, use_symfc_projector=False)
    phonopy_fc2_asr = np.asarray(phonon.force_constants).copy()

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        unitcell_numbers=unitcell.numbers,
        unitcell_cell=np.asarray(unitcell.cell),
        unitcell_scaled_positions=unitcell.get_scaled_positions(),
        mlfcs_forces=mlfcs_forces,
        mlfcs_plan_hash=np.asarray(calculation.plan.hash),
        cutoff_angstrom=np.asarray(calculation.cutoff),
        displacement_angstrom=np.asarray(DISPLACEMENT),
        maximum_mic_distance_angstrom=np.asarray(maximum_mic_distance),
        cutoff_mode=np.asarray("full_supercell"),
        phonopy_supercell_numbers=phonopy_supercell.numbers,
        phonopy_supercell_cell=np.asarray(phonopy_supercell.cell),
        phonopy_supercell_scaled_positions=phonopy_supercell.get_scaled_positions(),
        phonopy_fc2=phonopy_fc2,
        phonopy_fc2_asr=phonopy_fc2_asr,
        phonopy_configurations=np.asarray(len(phonopy_forces)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("potential", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    generate_fixture(args.dataset, args.potential, args.target)


if __name__ == "__main__":
    main()
