#!/usr/bin/env python3
"""Convert the official hiPhive NaCl data and remove Gonze long-range forces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, write_FORCE_CONSTANTS
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
SUPERCELL_MATRIX = np.diag((4, 4, 4))
PRIMITIVE_MATRIX = np.array(
    ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_with_forces(structure: Atoms, forces: np.ndarray, displacements: np.ndarray) -> Atoms:
    copied = structure.copy()
    copied.new_array("forces", np.asarray(forces, dtype=float))
    copied.new_array("displacements", np.asarray(displacements, dtype=float))
    return copied


def main() -> None:
    unitcell_path = INPUT / "NaCl_unitcell.xyz"
    force_path = INPUT / "supercells_with_forces.xyz"
    born_path = INPUT / "BORN"
    unitcell = read(unitcell_path)
    source_frames = read(force_path, index=":")
    if len(unitcell) != 8 or len(source_frames) != 2:
        raise RuntimeError("unexpected upstream NaCl dataset dimensions")

    phonon = Phonopy(
        _phonopy_atoms(unitcell),
        SUPERCELL_MATRIX,
        primitive_matrix=PRIMITIVE_MATRIX,
    )
    reference = _ase_atoms(phonon.supercell)
    primitive = _ase_atoms(phonon.primitive)
    if len(reference) != 512 or len(primitive) != 2:
        raise RuntimeError("unexpected phonopy conventional/primitive mapping")
    phonon.generate_displacements(distance=0.01)
    if len(phonon.supercells_with_displacements) != len(source_frames):
        raise RuntimeError("upstream frames do not match phonopy displacement count")

    total_frames: list[Atoms] = []
    displacements = []
    total_forces = []
    for frame in source_frames:
        if not np.array_equal(frame.numbers, reference.numbers):
            raise RuntimeError("upstream frame atom order differs from phonopy reference")
        # The upstream extended XYZ stores the 22.7612059 Å cell with fewer
        # decimal places than the unit-cell-derived phonopy supercell.
        if not np.allclose(frame.cell.array, reference.cell.array, rtol=0.0, atol=1e-8):
            raise RuntimeError("upstream frame cell differs from phonopy reference")
        displacement = find_mic(frame.positions - reference.positions, reference.cell, pbc=True)[0]
        force = frame.get_forces()
        total_frames.append(_copy_with_forces(frame, force, displacement))
        displacements.append(displacement)
        total_forces.append(force)
    displacements_array = np.asarray(displacements)
    total_forces_array = np.asarray(total_forces)

    phonon.forces = total_forces_array
    phonon.produce_force_constants()
    phonopy_fc2 = np.asarray(phonon.force_constants)
    write_FORCE_CONSTANTS(phonopy_fc2, filename=ROOT / "FORCE_CONSTANTS_PHONOPY")

    phonon.force_constants = np.zeros_like(phonopy_fc2)
    phonon.nac_params = parse_BORN(phonon.primitive, filename=born_path)
    dynamical_matrix = phonon.dynamical_matrix
    dynamical_matrix.make_Gonze_nac_dataset()
    long_range_fc2 = -np.asarray(dynamical_matrix.Gonze_nac_dataset[0])
    long_range_forces = -np.einsum("ijab,njb->nia", long_range_fc2, displacements_array)
    short_range_forces = total_forces_array - long_range_forces
    reconstructed = short_range_forces + long_range_forces
    reconstruction_error = float(np.max(np.abs(reconstructed - total_forces_array)))
    if reconstruction_error > 1e-14:
        raise RuntimeError(f"force decomposition failed: {reconstruction_error:.3e} eV/angstrom")

    short_frames = [
        _copy_with_forces(frame, force, displacement)
        for frame, force, displacement in zip(
            source_frames, short_range_forces, displacements_array, strict=True
        )
    ]
    write(ROOT / "primitive.vasp", primitive, format="vasp", direct=True, sort=False)
    write(ROOT / "supercell.vasp", reference, format="vasp", direct=True, sort=False)
    write(ROOT / "training-total.extxyz", total_frames, format="extxyz")
    write(ROOT / "training-short-range.extxyz", short_frames, format="extxyz")
    np.save(ROOT / "LONG_RANGE_FC2.npy", long_range_fc2)

    reread_total = read(ROOT / "training-total.extxyz", index=":")
    reread_short = read(ROOT / "training-short-range.extxyz", index=":")
    for original, converted, short, expected_short in zip(
        source_frames, reread_total, reread_short, short_range_forces, strict=True
    ):
        if not np.array_equal(original.numbers, converted.numbers):
            raise RuntimeError("converted atom order changed")
        if not np.allclose(original.positions, converted.positions, rtol=0.0, atol=1e-12):
            raise RuntimeError("converted positions changed")
        if not np.allclose(original.get_forces(), converted.get_forces(), rtol=0.0, atol=1e-12):
            raise RuntimeError("converted total forces changed")
        if not np.allclose(short.get_forces(), expected_short, rtol=0.0, atol=1e-8):
            raise RuntimeError("converted short-range forces changed")

    nac = phonon.nac_params
    summary = {
        "source": "hiPhive official NaCl long-range-corrections example",
        "input_sha256": {
            path.name: _sha256(path) for path in (unitcell_path, force_path, born_path)
        },
        "conventional_atoms": len(unitcell),
        "primitive_atoms": len(primitive),
        "reference_atoms": len(reference),
        "frames": len(source_frames),
        "maximum_displacement_angstrom": float(np.max(np.abs(displacements_array))),
        "maximum_total_force_ev_per_angstrom": float(np.max(np.abs(total_forces_array))),
        "maximum_long_range_force_ev_per_angstrom": float(np.max(np.abs(long_range_forces))),
        "maximum_short_range_force_ev_per_angstrom": float(np.max(np.abs(short_range_forces))),
        "force_decomposition_maximum_error": reconstruction_error,
        "dielectric": np.asarray(nac["dielectric"]).tolist(),
        "born_effective_charges": np.asarray(nac["born"]).tolist(),
    }
    (ROOT / "preparation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"prepared official NaCl total and short-range datasets in {ROOT}")


if __name__ == "__main__":
    main()
