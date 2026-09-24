"""Calculate Si FC2 using the current ASE-first finite-difference API."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import (
    ClusterMap,
    FiniteDifference,
    ForceConstants,
    PrimitiveCell,
    Supercell,
    build_cluster_space,
)

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Si_2022_NEP3_5body.txt"
ORDER = 2
SUPERCELL_REPETITIONS = 4
CUTOFF_ANGSTROM = 7.7237404951
DISPLACEMENT_ANGSTROM = 0.01
SYMPREC_ANGSTROM = 1e-5


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


def _run() -> None:
    primitive_atoms = read(ROOT / "POSCAR.vasp")
    primitive = PrimitiveCell.from_atoms(primitive_atoms, symprec=SYMPREC_ANGSTROM)
    supercell_atoms = primitive_atoms.repeat((SUPERCELL_REPETITIONS,) * 3)
    supercell = Supercell.from_atoms(
        primitive,
        supercell_atoms,
        matrix=np.diag([SUPERCELL_REPETITIONS] * 3),
    )
    write(ROOT / "SPOSCAR", supercell_atoms, format="vasp", direct=True, sort=False, vasp5=True)

    space = build_cluster_space(
        primitive,
        cutoffs={ORDER: CUTOFF_ANGSTROM},
        max_body_orders={ORDER: ORDER},
    )
    mapping = ClusterMap.build(space, supercell)
    calculation = FiniteDifference(
        mapping,
        order=ORDER,
        disps=DISPLACEMENT_ANGSTROM,
    )
    evaluated = calculation.evaluate(CPUNEP(str(MODEL)))
    raw = calculation.reconstruct(evaluated)
    projection = raw.enforce_asr(rtol=1e-10)
    force_constants: ForceConstants = projection.force_constants
    report = projection.report(ORDER)

    force_constants.save(ROOT / "fc2.mlfcs")
    force_constants.write(
        ROOT / "FORCE_CONSTANTS",
        mapping,
        format="phonopy",
        order=ORDER,
        storage="text",
    )
    force_constants.write(
        ROOT / "force_constants.hdf5",
        mapping,
        format="phonopy",
        order=ORDER,
        storage="hdf5",
    )
    metrics = {
        "order": ORDER,
        "primitive_atoms": primitive.size,
        "supercell_matrix": supercell.matrix.tolist(),
        "supercell_atoms": len(supercell.numbers),
        "cutoff_angstrom": CUTOFF_ANGSTROM,
        "displacement_angstrom": DISPLACEMENT_ANGSTROM,
        "orbits": len(space.orbits),
        "parameters": space.n_parameters,
        "configurations": calculation.n_configurations,
        "asr": asdict(report),
        "force_constants_fingerprint": force_constants.fingerprint,
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"FC{ORDER}: {space.n_parameters} parameters, {calculation.n_configurations} "
        f"configurations; ASR relative residual {report.relative_before:.3e} -> "
        f"{report.relative_after:.3e}"
    )


def main() -> None:
    with (ROOT / "run.log").open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        package_logger = logging.getLogger("mlfcs")
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)
        try:
            _run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            package_logger.removeHandler(handler)
            handler.close()
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
