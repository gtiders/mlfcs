"""Prepare the single Taylor FC2+FC3+FC4 model consumed by loop-SCPH."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from ase.io import read
from ase.units import Bohr

from mlfcs import write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
SOURCE = ROOT / "source"


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
    SOURCE.mkdir(exist_ok=True)
    fitter = ForceConstantFitter(
        read(INPUT / "primitive.vasp"),
        read(INPUT / "supercell.vasp"),
        orders=(2, 3, 4),
        cutoffs={2: 6.5, 3: 12 * Bohr, 4: 8 * Bohr},
        max_body_orders={2: 2, 3: 3, 4: 3},
    )
    gram = fitter.prepare_gram(
        read(INPUT / "train.extxyz", index=":"), batch_size=4, acoustic_sum_rule=True
    )
    result = fitter.fit(
        gram,
        tolerance=1e-5,
        max_iterations=10_000,
    )
    write_force_constants(result.force_constants, SOURCE / "mlfcs.h5", format="hdf5")


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
