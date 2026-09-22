"""Generate Si force snapshots with SSCHA sampling and fit FC2 from extxyz."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantFitter, write_force_constants
from mlfcs.fitting.dataset import FitDataset
from mlfcs.tools.gaussian import perturb_structures
from mlfcs.tools.supercell import build_supercell

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
        cutoffs={2: 7.7237404951},
    )
    gram = fitter.prepare_gram(training)
    result = fitter.fit(gram)
    # Training-data diagnostics come from the dataset, not from the fitter: the
    # per-snapshot net force and center-of-mass displacement describe the inputs.
    dataset = FitDataset.from_atoms(fitter.geometry, training)
    maximum_snapshot_net_force = float(np.max(np.linalg.norm(dataset.net_forces, axis=1)))
    maximum_center_of_mass_displacement = float(
        np.max(np.linalg.norm(dataset.center_of_mass_displacements, axis=1))
    )

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
                "unprojected_training_force_rmse_eV_per_A": (
                    result.unprojected_training_force_rmse
                ),
                "unprojected_training_relative_force_error": (
                    result.unprojected_training_relative_force_error
                ),
                "maximum_asr_residual_before": result.maximum_asr_residual_before,
                "maximum_asr_residual_after": result.maximum_asr_residual_after,
                "asr_parameter_correction": result.asr_parameter_correction,
                "asr_projection_iterations": result.asr_projection_iterations,
                "maximum_snapshot_net_force_eV_per_A": maximum_snapshot_net_force,
                "maximum_center_of_mass_displacement_A": maximum_center_of_mass_displacement,
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
