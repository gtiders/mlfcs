"""Run phono3py through its command-line supercell adapter."""

from __future__ import annotations

import json
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.structure.atoms import PhonopyAtoms

from mlfcs import read_hdf5, write_force_constants

MESH = tuple(int(value) for value in os.environ.get("MLFCS_PHONO3PY_MESH", "10,10,10").split(","))
if len(MESH) != 3 or any(value < 1 for value in MESH):
    raise ValueError("MLFCS_PHONO3PY_MESH must contain three positive integers")
TEMPERATURE = 300


def run(root: Path) -> None:
    force_constants = read_hdf5(root / "mlfcs.h5")
    supercell_path = root / "supercell.vasp"
    if not supercell_path.exists():
        supercell_path = root / "input" / "supercell.vasp"
    if not supercell_path.exists():
        raise FileNotFoundError("expected supercell.vasp in the case or input directory")
    supercell = read(supercell_path)
    primitive_path = root / "primitive.vasp"
    if not primitive_path.exists():
        primitive_path = root / "input" / "primitive.vasp"
    primitive = read(primitive_path) if primitive_path.exists() else None
    # The project writer emits dense external arrays in this explicit
    # reference-supercell ordering.  Do not pass the primitive as Phono3py's
    # unit cell; it discovers the primitive from the supplied supercell.
    write_force_constants(
        force_constants, root / "fc2.hdf5", format="phonopy_hdf5", order=2,
        primitive=primitive, supercell=supercell,
    )
    write_force_constants(
        force_constants, root / "fc3.hdf5", format="phono3py_hdf5", order=3,
        primitive=primitive, supercell=supercell,
    )
    atoms = supercell
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(), masses=atoms.get_masses(),
    )
    ph3 = Phono3py(unitcell, np.eye(3, dtype=int), primitive_matrix="auto", log_level=1)
    fc2 = read_fc2_from_hdf5(root / "fc2.hdf5")
    fc3 = read_fc3_from_hdf5(root / "fc3.hdf5")
    p2s_map = np.asarray(ph3.phonon_primitive.p2s_map, dtype=int)
    ph3.fc2 = fc2[p2s_map]
    ph3.fc3 = fc3[p2s_map]
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=[TEMPERATURE], is_isotope=True, write_kappa=True,
        output_filename="m101010", log_level=1,
    )
    candidates = sorted(root.glob("kappa-m101010*.hdf5"))
    if not candidates:
        raise FileNotFoundError("phono3py did not create a kappa-m101010 HDF5 file")
    (root / "thermal-conductivity.json").write_text(
        json.dumps({
            "temperature_K": TEMPERATURE, "mesh": list(MESH),
            "isotope_scattering": True, "method": "phono3py-python-api-RTA",
            "kappa_file": candidates[-1].name,
        }, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    root = Path.cwd()
    with (root / "thermal-conductivity.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                run(root)
        except BaseException:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
