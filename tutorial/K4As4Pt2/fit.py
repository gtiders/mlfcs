"""Fit the K4As4Pt2 FC2, FC3 and FC4 model from stored ASE forces."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import iread, read
from ase.units import Bohr

from mlfcs import (
    ClusterMap,
    FitSystem,
    PrimitiveCell,
    Supercell,
    build_cluster_space,
)

ROOT = Path(__file__).resolve().parent
SUPERCELL_MATRIX = ((2, 0, 0), (0, 2, 0), (0, 0, 3))
CUTOFFS_ANGSTROM = {2: 6.5, 3: 12 * Bohr, 4: 8 * Bohr}
MAX_BODY_ORDERS = {2: 2, 3: 3, 4: 3}
SYMPREC_ANGSTROM = 1e-5
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
    primitive = PrimitiveCell.from_atoms(read(ROOT / "primitive.vasp"), symprec=SYMPREC_ANGSTROM)
    supercell = Supercell.from_atoms(
        primitive,
        read(ROOT / "supercell.vasp"),
        matrix=SUPERCELL_MATRIX,
    )
    space = build_cluster_space(
        primitive,
        cutoffs=CUTOFFS_ANGSTROM,
        max_body_orders=MAX_BODY_ORDERS,
    )
    mapping = ClusterMap.build(space, supercell)

    system = FitSystem.from_atoms(mapping, iread(ROOT / "train.extxyz", index=":"))
    parameters = system.solve(rtol=SOLVER_RTOL, max_steps=SOLVER_MAX_STEPS)
    raw = system.force_constants(parameters)
    projection = raw.enforce_asr(rtol=ASR_RTOL)
    force_constants = projection.force_constants
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
    force_constants.write(
        ROOT / "FORCE_CONSTANTS_4TH",
        mapping,
        format="shengbte",
        order=4,
    )

    reports = {str(report.order): asdict(report) for report in projection.reports}
    metrics = {
        "primitive_atoms": primitive.size,
        "supercell_atoms": len(supercell.numbers),
        "supercell_matrix": [list(row) for row in SUPERCELL_MATRIX],
        "training_structures": system.n_structures,
        "training_equations": system.n_equations,
        "orders": list(space.orders),
        "cutoffs_angstrom": {str(key): value for key, value in CUTOFFS_ANGSTROM.items()},
        "max_body_orders": {str(key): value for key, value in MAX_BODY_ORDERS.items()},
        "orbits": {
            str(block.order): block.orbits.stop - block.orbits.start for block in space.blocks
        },
        "parameters": system.n_parameters,
        "solver": "column-scaled MINRES",
        "column_scale": "1 / sqrt(diag(FitSystem.matrix))",
        "solver_rtol": SOLVER_RTOL,
        "solver_max_steps": SOLVER_MAX_STEPS,
        "fit_system_fingerprint": system.fingerprint,
        "force_constants_fingerprint": force_constants.fingerprint,
        "training_force_rmse_before_asr_eV_per_angstrom": system.rmse(parameters),
        "training_relative_force_error_before_asr": system.relative_error(parameters),
        "training_force_rmse_after_asr_eV_per_angstrom": system.rmse(projected_parameters),
        "training_relative_force_error_after_asr": system.relative_error(projected_parameters),
        "asr": reports,
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"K4As4Pt2: {system.n_structures} structures, {system.n_parameters} parameters "
        f"for FC2/FC3/FC4; training relative force error "
        f"{system.relative_error(projected_parameters):.6%} after ASR"
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
