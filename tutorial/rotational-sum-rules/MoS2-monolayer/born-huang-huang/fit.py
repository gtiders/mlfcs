"""Fit MoS2 FC2, then enforce Born-Huang and Huang conditions."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import read

from mlfcs import enforce_rotational_sum_rules, write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent


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
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _run() -> None:
    primitive = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    snapshots = read(ROOT / "training.extxyz", index=":")
    result = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 8.0},
        max_body_orders={2: 2},
    ).fit(snapshots, validation_split=0.2, seed=0, acoustic_sum_rule=True)
    correction = enforce_rotational_sum_rules(result.force_constants, born_huang=True, huang=True)
    write_force_constants(correction.force_constants, ROOT / "mlfcs.h5", format="hdf5")
    write_force_constants(correction.force_constants, ROOT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    payload = {"fit": asdict(result), "rotational_sum_rules": asdict(correction)}
    (ROOT / "metrics.json").write_text(
        json.dumps(_json_ready(payload), default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    log_path = ROOT / "fit.log"
    with log_path.open("w", encoding="utf-8") as log_file:
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
