"""Fit harmonic KCl FC2 from ten Gaussian perturbation structures."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

from ase.io import read, write
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import perturb_structures, write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
CASE = ROOT.parent
LOG = ROOT / "fit.log"
MODEL = CASE / "input" / "polymlp.yaml"


class _Tee:
    """Mirror stdout and stderr to the terminal and this task's fit.log."""

    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def _run() -> None:
    primitive = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    structures = perturb_structures(
        reference,
        snapshots=10,
        method="gaussian",
        displacement=0.01,
        random_seed=42,
    )

    calculator = PolymlpASECalculator(pot=MODEL)
    training = []
    for structure in structures:
        structure.calc = calculator
        forces = structure.get_forces().copy()
        clean = structure.copy()
        clean.info.clear()
        for name in tuple(clean.arrays):
            if name not in {"numbers", "positions"}:
                del clean.arrays[name]
        clean.new_array("forces", forces)
        clean.calc = None
        training.append(clean)
    write(ROOT / "train.extxyz", training, format="extxyz")

    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 6.0},
        max_body_orders={2: 2},
    )
    result = fitter.fit(
        read(ROOT / "train.extxyz", index=":"),
        validation_split=0.0,
        acoustic_sum_rule=True,
        cache_directory=ROOT / "fit-cache",
    )

    write_force_constants(result.force_constants, ROOT / "fit.h5", format="hdf5")
    write_force_constants(
        result.force_constants,
        ROOT / "FORCE_CONSTANTS_FIT",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        result.force_constants,
        ROOT / "force_constants-fit.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    metrics = {
        key: getattr(result, key)
        for key in (
            "iterations",
            "training_force_rmse",
            "validation_force_rmse",
            "training_relative_force_error",
            "validation_relative_force_error",
            "order_force_rms",
            "stop_code",
            "maximum_constraint_residual",
            "maximum_reference_force",
            "maximum_snapshot_net_force",
            "maximum_center_of_mass_displacement",
        )
    }
    (ROOT / "fit-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote train.extxyz, fit.h5, FORCE_CONSTANTS_FIT, and fit-metrics.json")


def main() -> None:
    with LOG.open("w", encoding="utf-8") as log_file:
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
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
