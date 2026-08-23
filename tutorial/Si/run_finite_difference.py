"""Run the complete Si FC2 finite-difference tutorial with a NEP calculator."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RESULTS = ROOT / "results" / "finite-difference"
MODEL = SOURCE / "Si_2022_NEP3_5body.txt"


def main() -> None:
    primitive = read(SOURCE / "POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    calculator = CPUNEP(str(MODEL))
    calculation = ForceConstantCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=0.01,
    )
    force_constants = calculation.run(calculator)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_force_constants(force_constants, RESULTS / "fc2-mlfcs.h5", format="hdf5")
    write_force_constants(
        force_constants,
        RESULTS / "FORCE_CONSTANTS_2ND",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        force_constants,
        RESULTS / "force_constants.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    (RESULTS / "metadata.json").write_text(
        json.dumps(force_constants.metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote FC2 results to {RESULTS}")


if __name__ == "__main__":
    main()
