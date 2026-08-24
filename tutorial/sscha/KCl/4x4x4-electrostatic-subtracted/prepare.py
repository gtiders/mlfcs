#!/usr/bin/env python3
"""Prepare KCl Gaussian data with phonopy Gonze long-range forces removed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, write_FORCE_CONSTANTS
from phonopy.structure.atoms import PhonopyAtoms
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import perturb_structures

ROOT = Path(__file__).resolve().parent
CASE = ROOT.parent
SUPERCELL_MATRIX = np.diag((4, 4, 4))


def _phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _ase_atoms(atoms: PhonopyAtoms) -> Atoms:
    return Atoms(
        numbers=atoms.numbers,
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        pbc=True,
    )


def _forces(calculator, structures: list[Atoms]) -> np.ndarray:
    values = []
    for structure in structures:
        structure.calc = calculator
        values.append(structure.get_forces())
        structure.calc = None
    return np.asarray(values)


def _assert_same_reference(reference: Atoms, generated: Atoms) -> None:
    if not np.array_equal(reference.numbers, generated.numbers):
        raise RuntimeError("phonopy and MLFCS supercell atom orders differ")
    delta = reference.get_scaled_positions() - generated.get_scaled_positions()
    delta -= np.rint(delta)
    if np.max(np.abs(delta)) > 1e-8:
        raise RuntimeError("phonopy and MLFCS supercell positions differ")


def main() -> None:
    unitcell = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    calculator = PolymlpASECalculator(pot=CASE / "input/polymlp.yaml")
    phonon = Phonopy(
        _phonopy_atoms(unitcell),
        SUPERCELL_MATRIX,
        primitive_matrix="auto",
    )
    _assert_same_reference(reference, _ase_atoms(phonon.supercell))

    phonon.generate_displacements(distance=0.01)
    displaced = [_ase_atoms(cell) for cell in phonon.supercells_with_displacements]
    phonon.forces = _forces(calculator, displaced)
    phonon.produce_force_constants()
    write_FORCE_CONSTANTS(
        phonon.force_constants,
        filename=ROOT / "FORCE_CONSTANTS_PHONOPY",
    )

    phonon.force_constants = np.zeros_like(phonon.force_constants)
    nac_params = parse_BORN(phonon.primitive, filename=ROOT / "born-nominal.txt")
    nac_params["factor"] = 14.399652
    phonon.nac_params = nac_params
    dynamical_matrix = phonon.dynamical_matrix
    dynamical_matrix.make_Gonze_nac_dataset()
    long_range_fc2 = -dynamical_matrix.Gonze_nac_dataset[0]
    np.save(ROOT / "LONG_RANGE_FC2.npy", long_range_fc2)

    total_structures = perturb_structures(
        reference,
        snapshots=100,
        method="gaussian",
        displacement=0.01,
        random_seed=42,
    )
    total_forces = _forces(calculator, total_structures)
    displacements = np.asarray(
        [structure.positions - reference.positions for structure in total_structures]
    )
    long_range_forces = -np.einsum("ijab,njb->nia", long_range_fc2, displacements)
    short_range_forces = total_forces - long_range_forces
    short_range_structures = []
    for total, total_force, short_force in zip(
        total_structures, total_forces, short_range_forces, strict=True
    ):
        total.new_array("forces", total_force)
        short = total.copy()
        short.arrays["forces"] = short_force.copy()
        short_range_structures.append(short)
    write(ROOT / "training-total.extxyz", total_structures, format="extxyz")
    write(ROOT / "training-short-range.extxyz", short_range_structures, format="extxyz")
    summary = {
        "atoms": len(reference),
        "gaussian_snapshots": len(total_structures),
        "gaussian_displacement_angstrom": 0.01,
        "random_seed": 42,
        "phonopy_displacements": len(displaced),
        "born_model": "nominal K=+1, Cl=-1",
        "electronic_dielectric": 2.365,
        "maximum_total_force_ev_per_angstrom": float(np.max(np.abs(total_forces))),
        "maximum_long_range_force_ev_per_angstrom": float(np.max(np.abs(long_range_forces))),
        "maximum_short_range_force_ev_per_angstrom": float(np.max(np.abs(short_range_forces))),
    }
    (ROOT / "preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote phonopy and Gaussian long-range-subtracted data to {ROOT}")


if __name__ == "__main__":
    main()
