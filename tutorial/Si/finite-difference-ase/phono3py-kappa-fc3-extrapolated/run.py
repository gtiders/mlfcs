"""Calculate Si conductivity with FC2 and zero-step-extrapolated FC3."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import h5py
import numpy as np
import phono3py
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc2_from_hdf5, read_fc3_from_hdf5
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parent
FC2_FILE = ROOT.parent / "fc2" / "force_constants.hdf5"
FC3_FILE = ROOT.parent / "fc3-extrapolated" / "fc3-extrapolated.hdf5"
SUPERCELL_FILE = ROOT.parent / "fc3-extrapolated" / "SPOSCAR"
MESH = (10, 10, 10)
TEMPERATURES = tuple(range(300, 901, 100))
OUTPUT_STEM = "si-m101010-fc3-extrapolated"


def _result_file() -> Path:
    candidates = sorted(ROOT.glob(f"kappa-*{OUTPUT_STEM}*.hdf5"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one phono3py kappa HDF5 containing {OUTPUT_STEM!r}; "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _read_result(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as hdf5:
        if "temperature" not in hdf5 or "kappa" not in hdf5:
            raise KeyError(
                f"{path.name} lacks temperature/kappa datasets; found {list(hdf5.keys())}"
            )
        temperatures = np.asarray(hdf5["temperature"], dtype=float).reshape(-1)
        kappa = np.asarray(hdf5["kappa"], dtype=float)
    if kappa.shape[0] != len(temperatures):
        raise ValueError(f"kappa shape {kappa.shape} does not match its temperature axis")
    return {
        "temperatures_K": temperatures.tolist(),
        "kappa_dataset_shape": list(kappa.shape),
        "kappa_dataset": kappa.tolist(),
        "kappa_diagonal_W_mK": kappa[:, :3].tolist(),
    }


def run() -> None:
    missing = [path for path in (FC2_FILE, FC3_FILE, SUPERCELL_FILE) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run the FC2 and FC3-extrapolation workflows first; missing: "
            + ", ".join(str(path) for path in missing)
        )
    atoms = read(SUPERCELL_FILE)
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.cell),
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    print(
        f"Input supercell: {len(atoms)} atoms from {SUPERCELL_FILE.name}; "
        "supercell_matrix=identity, primitive_matrix='auto'"
    )
    ph3 = Phono3py(
        unitcell,
        np.eye(3, dtype=int),
        primitive_matrix="auto",
        log_level=1,
    )
    print(f"Identified primitive cell: {len(ph3.primitive)} atoms")
    ph3.fc2 = read_fc2_from_hdf5(FC2_FILE)
    ph3.fc3 = read_fc3_from_hdf5(FC3_FILE)
    ph3.mesh_numbers = MESH
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        temperatures=TEMPERATURES,
        is_isotope=True,
        write_kappa=True,
        output_filename=OUTPUT_STEM,
        log_level=1,
    )
    result_file = _result_file()
    report = {
        "material": "Si",
        "mesh": list(MESH),
        "temperatures_K_requested": list(TEMPERATURES),
        "isotope_scattering": True,
        "fc3": "five-displacement zero-step extrapolation",
        "primitive_matrix": "auto",
        "input_supercell_matrix": np.eye(3, dtype=int).tolist(),
        "input_supercell_atoms": len(atoms),
        "primitive_atoms": len(ph3.primitive),
        "phono3py_version": phono3py.__version__,
        "fc2_file": str(FC2_FILE.relative_to(ROOT.parent)),
        "fc3_file": str(FC3_FILE.relative_to(ROOT.parent)),
        "supercell_file": str(SUPERCELL_FILE.relative_to(ROOT.parent)),
        "kappa_file": result_file.name,
        **_read_result(result_file),
    }
    (ROOT / "thermal-conductivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote thermal conductivity: {result_file.name}")


def main() -> None:
    with (ROOT / "run.log").open("w", encoding="utf-8") as log_file:
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
