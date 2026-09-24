"""Calculate Si thermal conductivity from the neighboring FC2 and FC3 data."""

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
FC3_FILE = ROOT.parent / "fc3" / "fc3.hdf5"
SUPERCELL_FILE = ROOT.parent / "fc3" / "SPOSCAR"
MESH = (10, 10, 10)
TEMPERATURES = tuple(range(300, 901, 100))
OUTPUT_STEM = "si-m101010"


def _result_file() -> Path:
    candidates = sorted(ROOT.glob(f"kappa-*{OUTPUT_STEM}*.hdf5"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one phono3py kappa HDF5 containing {OUTPUT_STEM!r}; "
            f"found {[path.name for path in candidates]}"
        )
    return candidates[0]


def _summarize_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as hdf5:
        if "temperature" not in hdf5 or "kappa" not in hdf5:
            raise KeyError(
                f"{path.name} does not contain expected temperature/kappa datasets; "
                f"datasets are {list(hdf5.keys())}"
            )
        temperatures = np.asarray(hdf5["temperature"], dtype=float).reshape(-1)
        kappa = np.asarray(hdf5["kappa"], dtype=float)

    if kappa.shape[0] != len(temperatures):
        raise ValueError(
            f"kappa temperature axis has shape {kappa.shape}, but there are "
            f"{len(temperatures)} temperatures"
        )
    if kappa.ndim == 2 and kappa.shape[1] >= 3:
        diagonal = kappa[:, :3]
    elif kappa.ndim == 3 and kappa.shape[1:] == (3, 3):
        diagonal = np.diagonal(kappa, axis1=1, axis2=2)
    else:
        diagonal = None

    result: dict[str, object] = {
        "temperatures_K": temperatures.tolist(),
        "kappa_dataset_shape": list(kappa.shape),
        "kappa_dataset": kappa.tolist(),
    }
    if diagonal is not None:
        result["kappa_diagonal_W_mK"] = diagonal.tolist()
    return result


def run() -> None:
    missing = [path for path in (FC2_FILE, FC3_FILE, SUPERCELL_FILE) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run the neighboring FC2 and FC3 tutorials first; missing: "
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

    # phono3py 4.x deprecates output_filename but still supports it. Keeping a
    # distinctive stem makes this tutorial's result deterministic and easy to
    # identify next to the source FC files.
    ph3.run_thermal_conductivity(
        temperatures=TEMPERATURES,
        is_isotope=True,
        write_kappa=True,
        output_filename=OUTPUT_STEM,
        log_level=1,
    )

    result_file = _result_file()
    summary = _summarize_hdf5(result_file)
    report = {
        "material": "Si",
        "mesh": list(MESH),
        "temperatures_K_requested": list(TEMPERATURES),
        "isotope_scattering": True,
        "primitive_matrix": "auto",
        "input_supercell_matrix": np.eye(3, dtype=int).tolist(),
        "input_supercell_atoms": len(atoms),
        "primitive_atoms": len(ph3.primitive),
        "phono3py_version": phono3py.__version__,
        "fc2_file": str(FC2_FILE.relative_to(ROOT.parent)),
        "fc3_file": str(FC3_FILE.relative_to(ROOT.parent)),
        "supercell_file": str(SUPERCELL_FILE.relative_to(ROOT.parent)),
        "kappa_file": result_file.name,
        **summary,
    }
    (ROOT / "thermal-conductivity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote thermal conductivity: {result_file.name}")


def main() -> None:
    with (ROOT / "run.log").open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
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
        finally:
            log_file.flush()
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
