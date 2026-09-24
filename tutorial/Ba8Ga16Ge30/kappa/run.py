#!/usr/bin/env python3
"""Calculate 300 K Ba8Ga16Ge30 thermal conductivity with phono3py."""

from __future__ import annotations

import json
import logging
import traceback
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path

import h5py
import numpy as np
import phono3py
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.structure.atoms import PhonopyAtoms

from mlfcs import ClusterMap, ForceConstants, Supercell

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEMPERATURE_K = 300
MESH = (7, 7, 7)
SUPERCELL_MATRIX = np.diag([2, 2, 2])
OUTPUT_STEM = "ba8-m777-T300"


def _read_kappa(path: Path) -> tuple[list[float], np.ndarray]:
    with h5py.File(path, "r") as handle:
        temperatures = np.asarray(handle["temperature"], dtype=float).reshape(-1)
        values = np.asarray(handle["kappa"], dtype=float)
    if values.shape != (len(temperatures), 6):
        raise ValueError(f"unexpected phono3py kappa shape {values.shape}")
    return temperatures.tolist(), values


def run() -> None:
    model = ForceConstants.load(ROOT / "force_constants.mlfcs")
    primitive = model.space.primitive
    supercell = Supercell.from_atoms(
        primitive,
        read(ROOT / "supercell.vasp"),
        matrix=SUPERCELL_MATRIX,
    )
    mapping = ClusterMap.build(model.space, supercell)

    fc2_path = HERE / "fc2.hdf5"
    fc3_path = HERE / "fc3.hdf5"
    model.write(fc2_path, mapping, format="phonopy", order=2, storage="hdf5")
    model.write(fc3_path, mapping, format="phono3py", order=3)

    atoms = read(ROOT / "supercell.vasp")
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    ph3 = Phono3py(
        unitcell,
        np.eye(3, dtype=int),
        primitive_matrix="auto",
        log_level=1,
    )
    ph3.fc2 = read_fc2_from_hdf5(fc2_path, p2s_map=ph3.primitive.p2s_map)
    full_fc3 = read_fc3_from_hdf5(fc3_path)
    expected = (len(atoms), len(atoms), len(atoms), 3, 3, 3)
    if not isinstance(full_fc3, np.ndarray):
        raise TypeError("the full FC3 file unexpectedly contains a nonzero mask")
    if full_fc3.shape != expected:
        raise ValueError(f"full FC3 shape {full_fc3.shape} does not match {expected}")
    ph3.fc3 = full_fc3
    del full_fc3
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    with chdir(HERE):
        ph3.run_thermal_conductivity(
            temperatures=(TEMPERATURE_K,),
            is_isotope=True,
            write_kappa=True,
            output_filename=OUTPUT_STEM,
            log_level=1,
        )

    candidates = sorted(HERE.glob(f"kappa-*{OUTPUT_STEM}*.hdf5"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one kappa output, found {[x.name for x in candidates]}")
    temperatures, kappa = _read_kappa(candidates[0])
    report = {
        "material": "Ba8Ga16Ge30",
        "temperature_K": TEMPERATURE_K,
        "temperatures_K": temperatures,
        "mesh": list(MESH),
        "isotope_scattering": True,
        "primitive_matrix": "auto",
        "primitive_atoms": primitive.size,
        "supercell_atoms": len(supercell.numbers),
        "phono3py_version": phono3py.__version__,
        "force_constants_format": "full dense HDF5",
        "fc2_file": fc2_path.name,
        "fc3_file": fc3_path.name,
        "kappa_file": candidates[0].name,
        "kappa_W_mK": kappa.tolist(),
        "kappa_diagonal_W_mK": kappa[:, :3].tolist(),
    }
    (HERE / "thermal-conductivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {TEMPERATURE_K} K conductivity on {MESH}: {candidates[0].name}")


def main() -> None:
    with (HERE / "thermal-conductivity.log").open("w", encoding="utf-8") as log_file:
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                logging.basicConfig(
                    stream=log_file,
                    level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s",
                    force=True,
                )
                run()
        except BaseException:
            traceback.print_exc(file=log_file)
            raise


if __name__ == "__main__":
    main()
