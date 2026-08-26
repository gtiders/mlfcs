#!/usr/bin/env python3
"""Fit direct and long-range-subtracted NaCl FC2 models with MLFCS."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.io import read
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS

from mlfcs import ForceConstantFitter, write_force_constants

ROOT = Path(__file__).resolve().parent
CUTOFF = 11.0


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


def _fit(name: str, dataset: str):
    primitive = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    structures = read(ROOT / dataset, index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: CUTOFF},
        max_body_orders={2: 2},
    )
    gram = fitter.prepare_gram(structures, acoustic_sum_rule=True)
    result = fitter.fit(
        gram,
        acoustic_sum_rule=True,
        tolerance=1e-10,
        max_iterations=10_000,
    )
    write_force_constants(result.force_constants, ROOT / f"mlfcs-{name}.h5", format="hdf5")
    output = ROOT / f"FORCE_CONSTANTS_MLFCS_{name.upper()}"
    write_force_constants(result.force_constants, output, format="phonopy", order=2)
    return fitter, result, parse_FORCE_CONSTANTS(output)


def _run() -> None:
    total_fitter, total_result, _ = _fit("total", "training-total.extxyz")
    short_fitter, short_result, short_fc2 = _fit("short", "training-short-range.extxyz")
    long_range_fc2 = np.load(ROOT / "LONG_RANGE_FC2.npy")
    restored_fc2 = short_fc2 + long_range_fc2
    write_FORCE_CONSTANTS(restored_fc2, filename=ROOT / "FORCE_CONSTANTS_MLFCS_RESTORED")

    metrics = {
        "cutoff_angstrom": CUTOFF,
        "frames": 2,
        "fitting_basis": total_result.fitting_basis,
        "direct_total": {
            "orbits": len(total_fitter.calculations[0].realized_orbit_space.orbits),
            "parameters": total_fitter.n_parameters,
            "training_force_rmse_ev_per_angstrom": total_result.training_force_rmse,
            "training_relative_force_error": total_result.training_relative_force_error,
            "maximum_constraint_residual": total_result.maximum_constraint_residual,
            "solver_iterations": total_result.iterations,
            "solver_stop_code": total_result.stop_code,
        },
        "short_range": {
            "orbits": len(short_fitter.calculations[0].realized_orbit_space.orbits),
            "parameters": short_fitter.n_parameters,
            "training_force_rmse_ev_per_angstrom": short_result.training_force_rmse,
            "training_relative_force_error": short_result.training_relative_force_error,
            "maximum_constraint_residual": short_result.maximum_constraint_residual,
            "solver_iterations": short_result.iterations,
            "solver_stop_code": short_result.stop_code,
        },
    }
    (ROOT / "mlfcs-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote direct, short-range, and restored MLFCS FC2 to {ROOT}")


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
            handler.close()


if __name__ == "__main__":
    main()
