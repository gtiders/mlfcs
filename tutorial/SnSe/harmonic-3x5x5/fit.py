#!/usr/bin/env python3
"""Fit harmonic SnSe FC2 with cutoff=None in a 3x5x5 supercell."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from ase.io import read

from mlfcs import ForceConstantFitter, write_force_constants

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


def _fit() -> None:
    primitive = read(ROOT / "primitive.vasp")
    supercell = read(ROOT / "supercell.vasp")
    snapshots = read(ROOT / "training.extxyz", index=":")
    fitter = ForceConstantFitter(
        primitive,
        supercell,
        orders=(2,),
        cutoffs={2: None},
        max_body_orders={2: 2},
        symprec=1e-4,
    )
    gram = fitter.prepare_gram(snapshots, acoustic_sum_rule=True)
    result = fitter.fit(
        gram,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
    )
    write_force_constants(result.force_constants, ROOT / "mlfcs.h5", format="hdf5")
    write_force_constants(
        result.force_constants, ROOT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2
    )
    metrics = {
        "supercell_matrix": [[3, 0, 0], [0, 5, 0], [0, 0, 5]],
        "atoms": len(supercell),
        "frames": len(snapshots),
        "requested_cutoff": None,
        "resolved_cutoff_angstrom": fitter.calculations[0].cutoff,
        "orbits": len(fitter.calculations[0].realized_orbit_space.orbits),
        "parameters": fitter.n_parameters,
        "training_force_rmse_ev_per_angstrom": result.training_force_rmse,
        "training_relative_force_error": result.training_relative_force_error,
        "maximum_constraint_residual": result.maximum_constraint_residual,
        "solver_iterations": result.iterations,
        "solver_stop_code": result.stop_code,
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote harmonic FC2 fit to {ROOT}")


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log_file:
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
