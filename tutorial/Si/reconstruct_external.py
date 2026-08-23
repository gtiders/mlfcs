"""Reconstruct Si FC2 from forces produced by an external calculator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
WORK = ROOT / "work" / "external"
RESULTS = ROOT / "results" / "external"


def main() -> None:
    primitive = read(SOURCE / "POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    calculation = ForceConstantCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=0.01,
    )
    structures = calculation.sow()
    forces = {
        atoms.info["mlfcs_configuration_id"]: np.load(
            WORK / "forces" / f"forces-{atoms.info['mlfcs_configuration_id']:05d}.npy"
        )
        for atoms in structures
    }
    force_constants = calculation.reap(forces)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_force_constants(force_constants, RESULTS / "fc2-mlfcs.h5", format="hdf5")
    write_force_constants(
        force_constants,
        RESULTS / "FORCE_CONSTANTS_2ND",
        format="phonopy",
        order=2,
    )
    (RESULTS / "metadata.json").write_text(
        json.dumps(force_constants.metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote FC2 results to {RESULTS}")


if __name__ == "__main__":
    main()
