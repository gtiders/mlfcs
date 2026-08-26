"""Fit K4As4Pt2 FC2+FC3+FC4 in the default Taylor basis."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import read
from ase.units import Bohr

from mlfcs import write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"


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


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _run() -> None:
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
    write_force_constants(result.force_constants, ROOT / "mlfcs.h5", format="hdf5")
    write_force_constants(result.force_constants, ROOT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    write_force_constants(result.force_constants, ROOT / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    write_force_constants(result.force_constants, ROOT / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
    (ROOT / "metrics.json").write_text(
        json.dumps(_json_ready(asdict(result)), default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
