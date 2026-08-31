"""Generate Si force snapshots with SSCHA sampling and fit FC2 from extxyz."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantFitter, build_supercell, perturb_structures, write_force_constants

MODEL = "Si_2022_NEP3_5body.txt"
TRAINING = Path("train.extxyz")
ROOT = Path(__file__).resolve().parent


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
    primitive = read("POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)

    snapshots = perturb_structures(
        reference,
        snapshots=3,
        method="gaussian",
        displacement=0.01,
        random_seed=42,
    )

    calculator = CPUNEP(str(MODEL))
    clean_snapshots = []
    for atoms in snapshots:
        atoms.calc = calculator
        forces = atoms.get_forces().copy()
        clean = atoms.copy()
        clean.info.clear()
        for name in tuple(clean.arrays):
            if name not in {"numbers", "positions"}:
                del clean.arrays[name]
        clean.new_array("forces", forces)
        clean.calc = None
        clean_snapshots.append(clean)
    write(TRAINING, clean_snapshots, format="extxyz")

    training = read(TRAINING, index=":")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: None},
    )
    gram = fitter.prepare_gram(training, acoustic_sum_rule=True)
    result = fitter.fit(gram)

    write_force_constants(result.force_constants, "fc2-fit-mlfcs.h5", format="hdf5")
    write_force_constants(
        result.force_constants,
        "FORCE_CONSTANTS_2ND_FIT",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        result.force_constants,
        "force_constants-fit.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    Path("fit-metrics.json").write_text(
        json.dumps(
            {
                "iterations": result.iterations,
                "stop_code": result.stop_code,
                "training_force_rmse_eV_per_A": result.training_force_rmse,
                "training_relative_force_error": result.training_relative_force_error,
                "maximum_constraint_residual": result.maximum_constraint_residual,
                "maximum_snapshot_net_force_eV_per_A": result.maximum_snapshot_net_force,
                "maximum_center_of_mass_displacement_A": (
                    result.maximum_center_of_mass_displacement
                ),
                "order_force_rms_eV_per_A": {
                    str(order): value for order, value in result.order_force_rms.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("wrote SPOSCAR, train.extxyz, fitted FC2 files, and fit-metrics.json")


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


if __name__ == "__main__":
    main()
