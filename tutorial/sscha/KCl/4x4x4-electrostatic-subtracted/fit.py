#!/usr/bin/env python3
"""Fit short-range KCl FC2 and restore the phonopy Gonze contribution."""

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
    reference = read(ROOT / "supercell.vasp")
    structures = read(ROOT / "training-short-range.extxyz", index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 12.0},
        max_body_orders={2: 2},
    )
    gram = fitter.prepare_gram(structures, acoustic_sum_rule=True)
    result = fitter.fit(
        gram,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
    )
    write_force_constants(result.force_constants, ROOT / "short-range.h5", format="hdf5")
    write_force_constants(
        result.force_constants,
        ROOT / "FORCE_CONSTANTS_SHORT_RANGE",
        format="phonopy",
        order=2,
    )
    short_range_fc2 = parse_FORCE_CONSTANTS(ROOT / "FORCE_CONSTANTS_SHORT_RANGE")
    long_range_fc2 = np.load(ROOT / "LONG_RANGE_FC2.npy")
    if short_range_fc2.shape != long_range_fc2.shape:
        raise RuntimeError(
            f"FC2 shape mismatch: short={short_range_fc2.shape}, long={long_range_fc2.shape}"
        )
    write_FORCE_CONSTANTS(
        short_range_fc2 + long_range_fc2,
        filename=ROOT / "FORCE_CONSTANTS_LRC",
    )
    phonopy_fc2 = parse_FORCE_CONSTANTS(ROOT / "FORCE_CONSTANTS_PHONOPY")
    restored_fc2 = short_range_fc2 + long_range_fc2
    metrics = {
        "atoms": len(reference),
        "frames": len(structures),
        "fitting_basis": result.fitting_basis,
        "cutoff_angstrom": fitter.calculations[0].cutoff,
        "orbits": len(fitter.calculations[0].realized_orbit_space.orbits),
        "parameters": fitter.n_parameters,
        "training_force_rmse_ev_per_angstrom": result.training_force_rmse,
        "training_relative_force_error": result.training_relative_force_error,
        "maximum_constraint_residual": result.maximum_constraint_residual,
        "solver_iterations": result.iterations,
        "solver_stop_code": result.stop_code,
        "restored_long_range_fc2": True,
        "nac_used_for_band_plot": False,
        "relative_fc2_difference_from_phonopy": float(
            np.linalg.norm(restored_fc2 - phonopy_fc2) / np.linalg.norm(phonopy_fc2)
        ),
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote short-range and restored KCl FC2 to {ROOT}")


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log_file:
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger = logging.getLogger("mlfcs")
        package_logger.addHandler(handler)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        try:
            _fit()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            package_logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
