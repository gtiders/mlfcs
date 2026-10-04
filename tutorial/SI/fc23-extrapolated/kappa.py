"""Calculate Si conductivity using extrapolated FC2 and FC3."""

from __future__ import annotations

import json
import traceback
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parent
MESH = (10, 10, 10)
TEMPERATURES = (300,)
FC2_FILE = ROOT / "fc2-extrapolated.hdf5"
OUTPUT_NAME = "Si101010-fc2-fc3-extrapolated"


def run() -> None:
    atoms = read(ROOT / "SPOSCAR")
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    ph3 = Phono3py(unitcell, np.eye(3, dtype=int), primitive_matrix="auto", log_level=1)
    p2s_map = np.asarray(ph3.phonon_primitive.p2s_map, dtype=int)
    ph3.fc2 = read_fc2_from_hdf5(FC2_FILE)[p2s_map]
    ph3.fc3 = read_fc3_from_hdf5(ROOT / "fc3-extrapolated.hdf5")[p2s_map]
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    with chdir(ROOT):
        ph3.run_thermal_conductivity(
            temperatures=TEMPERATURES,
            is_isotope=True,
            write_kappa=True,
            output_filename=OUTPUT_NAME,
            log_level=1,
        )

    mesh_name = "m" + "".join(str(value) for value in MESH)
    outputs = sorted(ROOT.glob(f"kappa-{mesh_name}.{OUTPUT_NAME}*.hdf5"))
    if not outputs:
        raise FileNotFoundError(f"phono3py did not create the kappa HDF5 output for {MESH}")
    result = {
        "material": "Si",
        "method": "phono3py RTA",
        "potential": "Erhart-Albe Si-II Tersoff; parameter file: Erhart-Albe-Si-II.tersoff",
        "mesh": list(MESH),
        "temperatures_K": list(TEMPERATURES),
        "isotope_scattering": True,
        "force_constants": {
            "fc2": "fc2-extrapolated.hdf5",
            "fc3": "fc3-extrapolated.hdf5",
            "supercell_atoms": len(atoms),
        },
        "kappa_file": outputs[-1].name,
    }
    (ROOT / "kappa.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    with (ROOT / "kappa.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                run()
        except BaseException:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
