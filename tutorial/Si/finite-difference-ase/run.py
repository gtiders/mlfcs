"""Run the complete Si FC2 finite-difference tutorial with a NEP calculator."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import (
    FiniteDifferenceCalculation,
    build_supercell,
    write_force_constants,
)

MODEL = "Si_2022_NEP3_5body.txt"
ROOT = Path(__file__).resolve().parent


class _Tee:
    def __init__(self, terminal, log_file) -> None:
        self._terminal, self._log_file = terminal, log_file

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()


def _run() -> None:
    primitive = read("POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)
    calculator = CPUNEP(str(MODEL))
    calculation = FiniteDifferenceCalculation(
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


def main() -> None:
    with (ROOT / "run.log").open("w", encoding="utf-8") as log_file:
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger = logging.getLogger("mlfcs")
        package_logger.addHandler(handler)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        try:
            _run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            package_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
