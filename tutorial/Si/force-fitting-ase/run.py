"""Generate Si force snapshots with SSCHA sampling and fit FC2 from extxyz."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import build_supercell, perturb_structures, write_force_constants
from mlfcs.fitting import ForceConstantFitter

MODEL = "Si_2022_NEP3_5body.txt"
TRAINING = Path("train.extxyz")


def main() -> None:
    primitive = read("POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)

    snapshots = perturb_structures(
        primitive,
        reference=reference,
        snapshots=3,
        displacement=0.01,
        random_seed=42,
    )

    calculator = CPUNEP(str(MODEL))
    clean_snapshots = []
    for atoms in snapshots:
        atoms.calc = calculator
        forces = atoms.get_forces().copy()
        clean = atoms.copy()
        clean.info.clear()
        for name in tuple(clean.arrays):
            if name not in {"numbers", "positions"}:
                del clean.arrays[name]
        clean.new_array("forces", forces)
        clean.calc = None
        clean_snapshots.append(clean)
    write(TRAINING, clean_snapshots, format="extxyz")

    training = read(TRAINING, index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: None},
    )
    result = fitter.fit(
        training,
        validation_split=0.0,
        acoustic_sum_rule=True,
        cache_directory="fit-cache",
    )

    write_force_constants(result.force_constants, "fc2-fit-mlfcs.h5", format="hdf5")
    write_force_constants(
        result.force_constants,
        "FORCE_CONSTANTS_2ND_FIT",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        result.force_constants,
        "force_constants-fit.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    Path("fit-metrics.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )

    print("wrote SPOSCAR, train.extxyz, fitted FC2 files, and fit-metrics.json")


if __name__ == "__main__":
    main()
