"""Fit the MoS2 FC2 model and project it onto translational invariance."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from ase.io import iread, read

from mlfcs import ClusterMap, ClusterSpace, FitSystem

ROOT = Path(__file__).resolve().parent
SUPERCELL_MATRIX = ((8, 0, 0), (0, 8, 0), (0, 0, 1))
SYMPREC_ANGSTROM = 1e-5
SOLVER_RTOL = 1e-8
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
    primitive = read(ROOT / "primitive.vasp")
    space = ClusterSpace(
        primitive, cutoffs={2: 8.0}, max_body_orders={2: 2}, symprec=SYMPREC_ANGSTROM
    )
    mapping = ClusterMap(space, read(ROOT / "supercell.vasp"), supercell_matrix=SUPERCELL_MATRIX)
    system = FitSystem(mapping, iread(ROOT / "training.extxyz", index=":"))
    model = system.solve(rtol=SOLVER_RTOL, maxiter=10000)
    raw = model
    parameters = raw.parameters()
    projection = raw.enforce_asr(rtol=ASR_RTOL)
    model = projection.force_constants
    model.save(ROOT / "force_constants.mlfcs")
    model.write(ROOT / "FORCE_CONSTANTS_2ND", mapping, format="phonopy", order=2, storage="text")
    projected_parameters = model.parameters()
    metrics = {
        "projection": "translational invariance (ASR)",
        "primitive_atoms": space.n_atoms,
        "supercell_atoms": len(mapping.atomic_numbers),
        "supercell_matrix": [list(row) for row in SUPERCELL_MATRIX],
        "training_structures": system.n_structures,
        "training_equations": system.n_equations,
        "orders": list(space.orders),
        "cutoff_angstrom": 8.0,
        "max_body_order": 2,
        "orbits": space.block(2).orbits.stop - space.block(2).orbits.start,
        "parameters": system.n_parameters,
        "solver": "column-scaled MINRES",
        "solver_rtol": SOLVER_RTOL,
        "training_force_rmse_before_projection_eV_per_angstrom": system.rmse(parameters),
        "training_relative_force_error_before_projection": system.relative_error(parameters),
        "training_force_rmse_after_projection_eV_per_angstrom": system.rmse(projected_parameters),
        "training_relative_force_error_after_projection": system.relative_error(
            projected_parameters
        ),
        "asr": [asdict(report) for report in projection.reports],
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"MoS2 ASR: {system.n_structures} frames, {system.n_parameters} parameters; "
        f"relative force error {system.relative_error(projected_parameters):.6%} after projection"
    )


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        try:
            _fit()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
