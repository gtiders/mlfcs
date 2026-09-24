#!/usr/bin/env python3
"""Fit the 300 K Ba8Ga16Ge30 effective FC2+FC3 model."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.io import iread, read

from mlfcs import (
    ClusterMap,
    FitSystem,
    PrimitiveCell,
    Supercell,
    build_cluster_space,
)

ROOT = Path(__file__).resolve().parent
TEMPERATURE_K = 300
SUPERCELL_MATRIX = np.diag([2, 2, 2])
CUTOFFS_ANGSTROM = {2: 5.4, 3: 4.35}
MAX_BODY_ORDERS = {2: 2, 3: 2}
SYMPREC_ANGSTROM = 1e-4
SOLVER_RTOL = 1e-8
SOLVER_MAX_STEPS = 10_000
ASR_RTOL = 1e-10


class _Tee:
    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def _fit() -> None:
    primitive_atoms = read(ROOT / "primitive.vasp")
    supercell_atoms = read(ROOT / "supercell.vasp")
    primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=SYMPREC_ANGSTROM)
    supercell = Supercell.from_atoms(
        primitive,
        supercell_atoms,
        matrix=SUPERCELL_MATRIX,
    )
    space = build_cluster_space(
        primitive,
        cutoffs=CUTOFFS_ANGSTROM,
        max_body_orders=MAX_BODY_ORDERS,
    )
    mapping = ClusterMap.build(space, supercell)
    fit_system = FitSystem.from_atoms(mapping, iread(ROOT / "nve.extxyz", index=":"))
    parameters = fit_system.solve(rtol=SOLVER_RTOL, max_steps=SOLVER_MAX_STEPS)
    unconstrained = fit_system.force_constants(parameters)
    projection = unconstrained.enforce_asr(rtol=ASR_RTOL)
    force_constants = projection.force_constants
    reports = {report.order: report for report in projection.reports}
    projected_parameters = force_constants.parameters()

    force_constants.save(ROOT / "force_constants.mlfcs")
    force_constants.write(
        ROOT / "FORCE_CONSTANTS_2ND",
        mapping,
        format="phonopy",
        order=2,
        storage="text",
    )
    force_constants.write(
        ROOT / "FORCE_CONSTANTS_3RD",
        mapping,
        format="shengbte",
        order=3,
    )
    metrics = {
        "temperature_K": TEMPERATURE_K,
        "primitive_atoms": primitive.size,
        "supercell_atoms": len(supercell.numbers),
        "supercell_matrix": supercell.matrix.tolist(),
        "training_structures": fit_system.n_structures,
        "training_equations": fit_system.n_equations,
        "orders": list(space.orders),
        "cutoffs_angstrom": {str(key): value for key, value in CUTOFFS_ANGSTROM.items()},
        "max_body_orders": {str(key): value for key, value in MAX_BODY_ORDERS.items()},
        "orbits": {
            str(block.order): block.orbits.stop - block.orbits.start for block in space.blocks
        },
        "parameters": fit_system.n_parameters,
        "solver": "column-scaled MINRES",
        "column_scale": "1 / sqrt(diag(FitSystem.matrix))",
        "solver_rtol": SOLVER_RTOL,
        "solver_max_steps": SOLVER_MAX_STEPS,
        "fit_system_fingerprint": fit_system.fingerprint,
        "force_constants_fingerprint": force_constants.fingerprint,
        "training_force_rmse_before_asr_eV_per_angstrom": fit_system.rmse(parameters),
        "training_relative_force_error_before_asr": fit_system.relative_error(parameters),
        "training_force_rmse_after_asr_eV_per_angstrom": fit_system.rmse(projected_parameters),
        "training_relative_force_error_after_asr": fit_system.relative_error(projected_parameters),
        "asr": {str(order): asdict(report) for order, report in reports.items()},
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Ba8Ga16Ge30 at {TEMPERATURE_K} K: {fit_system.n_structures} frames, "
        f"{fit_system.n_parameters} parameters; training relative force error "
        f"{fit_system.relative_error(projected_parameters):.6%} after ASR"
    )


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        package_logger = logging.getLogger("mlfcs")
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)
        try:
            _fit()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            package_logger.removeHandler(handler)
            handler.close()
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
