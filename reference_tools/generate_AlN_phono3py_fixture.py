"""Maintenance utility to regenerate the compact AlN third-order validation fixture.

This is a maintainer tool, not a public MLFCS command-line interface.  It uses
the phono3py ``example/AlN-rd`` training dataset and a pypolymlp model trained
from that dataset.  CI consumes only the derived NPZ fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from phono3py import Phono3py, load
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
    """Generate captured MLFCS forces and an independent phono3py FC3."""
    source = load(dataset, produce_fc=False)
    unitcell_ph = source.unitcell
    unitcell = _ase_atoms(unitcell_ph)
    calculator = PolymlpASECalculator(pot=str(potential))
    trial_supercell, _ = make_supercell(unitcell, SUPERCELL)
    maximum_mic_distance = float(trial_supercell.get_all_distances(mic=True).max())
    full_supercell_cutoff = maximum_mic_distance + CUTOFF_MARGIN

    calculation = ForceConstantCalculation(
        unitcell,
        order=3,
        supercell=SUPERCELL,
        cutoff=full_supercell_cutoff,
        displacement=DISPLACEMENT,
        report_cutoff=False,
    )
    mlfcs_forces = calculation.evaluate(calculator)

    ph3 = Phono3py(unitcell_ph, supercell_matrix=np.diag(SUPERCELL))
    ph3.generate_displacements(distance=DISPLACEMENT, is_plusminus=True)
    phono3py_forces = []
    for displaced in ph3.supercells_with_displacements:
        if displaced is None:
            # phono3py retains a positional placeholder for pairs excluded by
            # cutoff; the traditional solver ignores these force entries.
            phono3py_forces.append(np.zeros((len(ph3.supercell), 3)))
            continue
        atoms = _ase_atoms(displaced)
        atoms.calc = calculator
        phono3py_forces.append(atoms.get_forces())
    ph3.forces = np.asarray(phono3py_forces)
    ph3.produce_fc3(is_compact_fc=False, fc_calculator="traditional")
    phono3py_supercell = _ase_atoms(ph3.supercell)
    phono3py_fc3 = np.asarray(ph3.fc3).copy()
    ph3.symmetrize_fc3(use_symfc_projector=False, options="level=3")
    phono3py_fc3_asr = np.asarray(ph3.fc3).copy()

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        unitcell_numbers=unitcell.numbers,
        unitcell_cell=np.asarray(unitcell.cell),
        unitcell_scaled_positions=unitcell.get_scaled_positions(),
        mlfcs_forces=mlfcs_forces,
        cutoff_angstrom=np.asarray(calculation.cutoff),
        displacement_angstrom=np.asarray(DISPLACEMENT),
        maximum_mic_distance_angstrom=np.asarray(maximum_mic_distance),
        cutoff_mode=np.asarray("full_supercell"),
        phono3py_supercell_numbers=phono3py_supercell.numbers,
        phono3py_supercell_cell=np.asarray(phono3py_supercell.cell),
        phono3py_supercell_scaled_positions=phono3py_supercell.get_scaled_positions(),
        phono3py_fc3=phono3py_fc3,
        phono3py_fc3_asr=phono3py_fc3_asr,
        phono3py_configurations=np.asarray(len(phono3py_forces)),
    )


def refresh_mlfcs_forces(potential: Path, target: Path) -> None:
    """Refresh only the order-sensitive MLFCS force capture in an existing fixture.

    The independent phono3py tensors are unchanged because they were generated
    from the same immutable potential and geometry.  This narrow migration is
    useful when a symmetry-equivalent displacement-plan enumeration changes.
    """
    with np.load(target) as stored:
        values = {name: stored[name] for name in stored.files}
    unitcell = Atoms(
        numbers=values["unitcell_numbers"],
        cell=values["unitcell_cell"],
        scaled_positions=values["unitcell_scaled_positions"],
        pbc=True,
    )
    calculation = ForceConstantCalculation(
        unitcell,
        order=3,
        supercell=SUPERCELL,
        cutoff=float(values["cutoff_angstrom"]),
        displacement=DISPLACEMENT,
        report_cutoff=False,
    )
    values["mlfcs_forces"] = calculation.evaluate(PolymlpASECalculator(pot=str(potential)))
    values.pop("mlfcs_plan_hash", None)
    np.savez_compressed(target, **values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("potential", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--refresh-mlfcs-forces",
        action="store_true",
        help="update the captured MLFCS force sequence while retaining phono3py tensors",
    )
    args = parser.parse_args()
    if args.refresh_mlfcs_forces:
        refresh_mlfcs_forces(args.potential, args.target)
    else:
        generate_fixture(args.dataset, args.potential, args.target)


if __name__ == "__main__":
    main()
