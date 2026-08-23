"""Run the Si FC2 finite-difference zero-step extrapolation tutorial."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants

MODEL = "Si_2022_NEP3_5body.txt"
LOG = Path("extrapolation.log")


class Tee:
    """Write calculator progress to both the terminal and the log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
    with (
        LOG.open("w", encoding="utf-8") as log,
        redirect_stdout(Tee(sys.stdout, log)),
        redirect_stderr(Tee(sys.stderr, log)),
    ):
        primitive = read("POSCAR.vasp")
        reference = build_supercell(primitive, (4, 4, 4))
        write(
            "SPOSCAR-extrapolation",
            reference,
            format="vasp",
            direct=True,
            sort=False,
            vasp5=True,
        )
        calculator = CPUNEP(str(MODEL))
        calculation = ForceConstantCalculation(
            primitive,
            order=2,
            reference=reference,
            cutoff=None,
            displacement=0.01,
        )
        force_constants = calculation.run(
            calculator,
            derivative_backend="extrapolate",
            extrapolation_spacing=0.002,
            extrapolation_side_steps=2,
            extrapolation_degree=1,
        )

        write_force_constants(
            force_constants,
            "fc2-mlfcs-extrapolation.h5",
            format="hdf5",
        )
        write_force_constants(
            force_constants,
            "FORCE_CONSTANTS_2ND_EXTRAPOLATION",
            format="phonopy",
            order=2,
        )
        write_force_constants(
            force_constants,
            "force_constants-extrapolation.hdf5",
            format="phonopy_hdf5",
            order=2,
        )
        Path("metadata-extrapolation.json").write_text(
            json.dumps(force_constants.metadata, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
