"""Run phono3py thermal conductivity for Si at all requested temperatures."""
from __future__ import annotations
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import numpy as np
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.structure.atoms import PhonopyAtoms
ROOT = Path(__file__).resolve().parent
MESH = (15, 15, 15)
TEMPERATURES = list(range(300, 1101, 100))
def run() -> None:
    supercell = read(ROOT / "SPOSCAR")
    unitcell = PhonopyAtoms(symbols=supercell.get_chemical_symbols(), cell=supercell.cell.array, scaled_positions=supercell.get_scaled_positions(), masses=supercell.get_masses())
    ph3 = Phono3py(unitcell, np.eye(3, dtype=int), primitive_matrix="auto", log_level=1)
    ph3.fc2 = read_fc2_from_hdf5(ROOT / "force_constants.hdf5")
    ph3.fc3 = read_fc3_from_hdf5(ROOT / "fc3.hdf5")
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    output = ROOT / "kappa-m151515.m151515.hdf5"
    ph3.run_thermal_conductivity(temperatures=TEMPERATURES, is_isotope=True, write_kappa=True, output_filename="m151515", log_level=1)
    if not output.exists():
        raise FileNotFoundError(f"missing phono3py output: {output.name}")
    (ROOT / "thermal-conductivity.json").write_text(json.dumps({"temperatures_K": TEMPERATURES, "mesh": list(MESH), "isotope_scattering": True, "kappa_file": output.name}, indent=2) + "\n")
def main() -> None:
    with (ROOT / "thermal-conductivity.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log): run()
        except BaseException:
            traceback.print_exc(file=log)
            raise
if __name__ == "__main__": main()
