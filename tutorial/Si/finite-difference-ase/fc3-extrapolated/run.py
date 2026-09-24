"""Calculate Si FC3 with five displacement lengths and extrapolate to zero."""

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
INPUT = ROOT.parent / "fc3"
MODEL = INPUT / "Si_2022_NEP3_5body.txt"
ORDER = 3
SUPERCELL_REPETITIONS = 4
CUTOFF_ANGSTROM = 7.7237404951
DISPLACEMENTS_ANGSTROM = (0.01, 0.015, 0.02, 0.025, 0.03)
SYMPREC_ANGSTROM = 1e-5
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


def _run() -> None:
    primitive_atoms = read(INPUT / "POSCAR.vasp")
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
        disps=DISPLACEMENTS_ANGSTROM,
    )
    evaluated = calculation.evaluate(CPUNEP(str(MODEL)))
    extrapolated_raw = calculation.reconstruct(evaluated)
    projection = extrapolated_raw.enforce_asr(rtol=ASR_RTOL)
    force_constants: ForceConstants = projection.force_constants
    report = projection.report(ORDER)

    # Compare with the middle step after applying the same ASR projection.
    # FiniteDifference's canonical sequence is grouped by key, then step, then
    # sign, so this extracts exactly the h=0.02 Å structures for the baseline.
    sign_count = 2 ** (ORDER - 1)
    steps = calculation.disps
    middle_step = steps.index(0.02)
    keys_count = calculation.n_configurations // (len(steps) * sign_count)
    middle_structures = tuple(
        structure
        for key_index in range(keys_count)
        for structure in evaluated[
            key_index * len(steps) * sign_count
            + middle_step * sign_count : key_index * len(steps) * sign_count
            + (middle_step + 1) * sign_count
        ]
    )
    middle_calculation = FiniteDifference(mapping, order=ORDER, disps=0.02)
    middle_raw = middle_calculation.reconstruct(middle_structures)
    middle = middle_raw.enforce_asr(rtol=ASR_RTOL).force_constants
    delta = force_constants.parameters((ORDER,)) - middle.parameters((ORDER,))
    baseline_norm = float(np.linalg.norm(middle.parameters((ORDER,))))
    relative_correction = float(np.linalg.norm(delta) / baseline_norm)

    force_constants.save(ROOT / "fc3-extrapolated.mlfcs")
    force_constants.write(
        ROOT / "fc3-extrapolated.hdf5",
        mapping,
        format="phono3py",
        order=ORDER,
    )
    force_constants.write(
        ROOT / "FORCE_CONSTANTS_3RD",
        mapping,
        format="shengbte",
        order=ORDER,
    )
    metrics = {
        "order": ORDER,
        "primitive_atoms": primitive.size,
        "supercell_matrix": supercell.matrix.tolist(),
        "supercell_atoms": len(supercell.numbers),
        "cutoff_angstrom": CUTOFF_ANGSTROM,
        "displacements_angstrom": list(DISPLACEMENTS_ANGSTROM),
        "displacement_configurations": calculation.n_configurations,
        "configurations_per_displacement": calculation.n_configurations // len(steps),
        "orbits": len(space.orbits),
        "parameters": space.n_parameters,
        "asr": asdict(report),
        "relative_l2_correction_from_0.02_angstrom": relative_correction,
        "force_constants_fingerprint": force_constants.fingerprint,
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"FC{ORDER} zero-step extrapolation: {len(steps)} steps, "
        f"{calculation.n_configurations} configurations; "
        f"relative L2 correction from 0.02 Å {relative_correction:.6e}; "
        f"ASR residual {report.relative_before:.3e} -> {report.relative_after:.3e}"
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
