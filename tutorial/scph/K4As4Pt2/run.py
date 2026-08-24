"""Run K4As4Pt2 loop-SCPH from one same-model Taylor FC2/FC4 source."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from mlfcs import LoopSCPH, read_hdf5, write_force_constants

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source/mlfcs.h5"


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
    if not SOURCE.is_file():
        raise FileNotFoundError("run prepare_ifcs.py before loop-SCPH")
    source = read_hdf5(SOURCE)
    series = LoopSCPH(
        fc2=source,
        fc4=source,
        temperature=(300.0, 600.0, 900.0),
        interpolation_multiplier=1,
        scph_multiplier=2,
        mixing=0.5,
        tolerance=1e-10,
        max_iterations=200,
        qpoint_workers=4,
    ).run()
    results = series.results if hasattr(series, "results") else (series,)
    for result in results:
        directory = ROOT / f"T{int(result.temperature)}K"
        directory.mkdir(exist_ok=True)
        write_force_constants(result.force_constants, directory / "mlfcs.h5", format="hdf5")
        write_force_constants(result.force_constants, directory / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
        (directory / "history.json").write_text(
            json.dumps(
                {
                    "temperature_K": result.temperature,
                    "converged": result.converged,
                    "iterations": [
                        {
                            "index": item.index,
                            "frequency_change_thz": item.frequency_change_thz,
                            "correction_norm": item.correction_norm,
                        }
                        for item in result.history
                    ],
                },
                indent=2,
            )
            + "\n",
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
