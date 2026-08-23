"""Run the complete Si FC2 finite-difference tutorial with a NEP calculator."""

import json
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants


MODEL = "Si_2022_NEP3_5body.txt"


def main() -> None:
    primitive = read("POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)
    calculator = CPUNEP(str(MODEL))
    calculation = ForceConstantCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=0.01,
    )
    force_constants = calculation.run(calculator)

    write_force_constants(
        force_constants, "fc2-mlfcs.h5", format="hdf5"
    )  # 这种hdf5只能mlfcs本身读写，不是phonopy那种稠密数组
    write_force_constants(
        force_constants,
        "FORCE_CONSTANTS_2ND",
        format="phonopy",
        order=2,
    )  # phonopy力常数的text文本格式
    write_force_constants(
        force_constants,
        "force_constants.hdf5",
        format="phonopy_hdf5",
        order=2,
    )  # 这个才是phonopy格式的hdf5格式
    Path("metadata.json").write_text(
        json.dumps(force_constants.metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
