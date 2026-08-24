"""Run the 4x4x4 KCl Taylor-basis SSCHA reference calculation."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import read
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import write_force_constants
from mlfcs.physics.sscha.solver import SSCHA

ROOT = Path(__file__).resolve().parent
CASE = ROOT.parent


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
    primitive = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    sscha = SSCHA(
        primitive,
        reference=reference,
        cutoff=12.0,
        temperature=600.0,
        snapshots=100,
        max_iterations=50,
        random_seed=42,
        imaginary_modes="absolute",
        mixing=1.0,
    )
    calculator = PolymlpASECalculator(pot=CASE / "input/polymlp.yaml")
    bootstrap = None
    for iteration in range(50):
        sscha.step(calculator, calculate_free_energy=True)
        if iteration == 0:
            bootstrap = sscha.force_constants
    result = sscha.force_constants
    if result is None or bootstrap is None:
        raise RuntimeError("SSCHA did not produce the required FC2 states")
    (ROOT / "history.json").write_text(
        json.dumps([asdict(item) for item in sscha.history], default=str, indent=2) + "\n",
        encoding="utf-8",
    )
    write_force_constants(result, ROOT / "sscha.h5", format="hdf5")
    write_force_constants(result, ROOT / "FORCE_CONSTANTS_SSCHA", format="phonopy", order=2)
    write_force_constants(bootstrap, ROOT / "FORCE_CONSTANTS_BOOTSTRAP", format="phonopy", order=2)


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log_file:
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
