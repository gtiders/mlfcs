"""Calculate K4As4Pt2 lattice thermal conductivity with phono3py RTA."""

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

from mlfcs import ClusterMap, ForceConstants, Supercell

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SUPERCELL_MATRIX = np.diag([2, 2, 3])
MESH = (10, 10, 10)
TEMPERATURES = tuple(range(300, 901, 100))


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
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    ph3 = Phono3py(unitcell, np.eye(3, dtype=int), primitive_matrix="auto", log_level=1)
    p2s_map = np.asarray(ph3.phonon_primitive.p2s_map, dtype=int)
    ph3.fc2 = read_fc2_from_hdf5(str(fc2_path))[p2s_map]
    ph3.fc3 = read_fc3_from_hdf5(str(fc3_path))[p2s_map]
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    with chdir(HERE):
        ph3.run_thermal_conductivity(
            temperatures=TEMPERATURES,
            is_isotope=True,
            write_kappa=True,
            output_filename="m101010",
            log_level=1,
        )

    candidates = sorted(HERE.glob("kappa-m101010*.hdf5"))
    if not candidates:
        raise FileNotFoundError("phono3py did not create a kappa-m101010 HDF5 file")
    result = {
        "material": "K4As4Pt2",
        "method": "phono3py RTA",
        "mesh": list(MESH),
        "temperatures_K": list(TEMPERATURES),
        "isotope_scattering": True,
        "force_constants": {
            "format": "full dense HDF5",
            "fc2": fc2_path.name,
            "fc3": fc3_path.name,
            "supercell_atoms": len(supercell.numbers),
        },
        "kappa_file": candidates[-1].name,
    }
    (HERE / "thermal-conductivity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    with (HERE / "thermal-conductivity.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                run()
        except BaseException:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
