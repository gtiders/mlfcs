#!/usr/bin/env python3
"""Fit the 500 K Ba8Ga16Ge30 effective FC2+FC3 model."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs import write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
TEMPERATURE = 500


class _Tee:
    def __init__(self, terminal, log_file) -> None:
        self._terminal = terminal
        self._log_file = log_file

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
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _fit() -> None:
    primitive = read(ROOT / "primitive.vasp")
    supercell = read(ROOT / "supercell.vasp")
    snapshots = read(ROOT / "nve.extxyz", index=":")
    fitter = ForceConstantFitter(
        primitive,
        supercell,
        orders=(2, 3),
        cutoffs={2: 5.4, 3: 4.35},
        max_body_orders={2: 2, 3: 2},
        symprec=1e-4,
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.0,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
    )
    write_force_constants(result.force_constants, ROOT / "mlfcs.h5", format="hdf5")
    write_force_constants(
        result.force_constants, ROOT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2
    )
    write_force_constants(
        result.force_constants, ROOT / "fc2.h5", format="phonopy_hdf5", order=2
    )
    write_force_constants(
        result.force_constants, ROOT / "FORCE_CONSTANTS_3RD", format="shengbte", order=3
    )
    (ROOT / "metrics.json").write_text(
        json.dumps(_json_ready(asdict(result)), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"T={TEMPERATURE} K: wrote effective FC2+FC3 to {ROOT}")


def main() -> None:
    log_path = ROOT / "fit.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger = logging.getLogger("mlfcs")
        package_logger.addHandler(handler)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(original_stdout, log_file), _Tee(original_stderr, log_file)
        try:
            _fit()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            package_logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
