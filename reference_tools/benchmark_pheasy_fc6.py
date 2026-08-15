"""Fit the public Pheasy SrTiO3 data with MLFCS through FC6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read
from scipy.optimize import linear_sum_assignment

from mlfcs.fitting import ForceConstantFitter


def _primitive(supercell: Atoms | None = None) -> Atoms:
    cell = np.eye(3) * 3.90514 if supercell is None else np.asarray(supercell.cell) / 2.0
    return Atoms(
        symbols=("Sr", "Ti", "O", "O", "O"),
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
        ),
        cell=cell,
        pbc=True,
    )


def _reference_in_qe_order(first_snapshot: Atoms) -> Atoms:
    """Return the ideal 2x2x2 cell in the atom order used by the QE files."""
    ideal = _primitive(first_snapshot).repeat((2, 2, 2))
    order = np.empty(len(ideal), dtype=int)
    for number in np.unique(ideal.numbers):
        source = np.flatnonzero(ideal.numbers == number)
        target = np.flatnonzero(first_snapshot.numbers == number)
        vectors = first_snapshot.positions[target, None] - ideal.positions[source][None, :]
        _, distances = find_mic(vectors.reshape(-1, 3), ideal.cell, pbc=True)
        rows, columns = linear_sum_assignment(distances.reshape(len(target), len(source)))
        order[target[rows]] = source[columns]
    reference = ideal[order]
    reference.calc = SinglePointCalculator(reference, forces=np.zeros((len(reference), 3)))
    return reference


def _read_snapshots(reference_directory: Path) -> list[Atoms]:
    paths = sorted(reference_directory.glob("DISP.out.*"))
    if len(paths) != 30:
        raise ValueError(f"expected 30 Pheasy snapshots, found {len(paths)}")
    snapshots = [read(path, format="espresso-out") for path in paths]
    if any(len(atoms) != 40 for atoms in snapshots):
        raise ValueError("the Pheasy SrTiO3 benchmark must contain 40-atom snapshots")
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheasy-root", type=Path, default=Path("pheasy"))
    parser.add_argument("--output", type=Path, default=Path("results/pheasy_srtio3_fc6"))
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument(
        "--regularization",
        choices=("none", "scaled_group_lasso"),
        default="none",
    )
    arguments = parser.parse_args()

    snapshots = _read_snapshots(arguments.pheasy_root / "examples/SrTiO3-QE/reference")
    primitive = _primitive(snapshots[0])
    reference = _reference_in_qe_order(snapshots[0])
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4, 5, 6),
        cutoffs={2: None, 3: None, 4: 6.0, 5: 6.0, 6: 6.0},
        max_body_orders={2: 2, 3: 3, 4: 3, 5: 2, 6: 2},
        jax_platform=arguments.platform,
    )
    result = fitter.fit(
        snapshots,
        validation_split=0.0,
        max_iterations=arguments.max_iterations,
        acoustic_sum_rule=True,
        allow_unconverged=arguments.allow_unconverged,
        regularization=(None if arguments.regularization == "none" else arguments.regularization),
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    result.force_constants.write(arguments.output / "force_constants.hdf5", format="hdf5")
    result.force_constants.write(arguments.output / "force_constants_2_4.xml", format="alamode")
    diagnostics = result.diagnostics
    summary = {
        "source": "https://gitlab.com/cplin/pheasy.git",
        "dataset": "examples/SrTiO3-QE/reference",
        "structures": len(snapshots),
        "orders": [2, 3, 4, 5, 6],
        "cutoff_angstrom": {"2": None, "3": None, "4": 6.0, "5": 6.0, "6": 6.0},
        "max_body_order": {"2": 2, "3": 3, "4": 3, "5": 2, "6": 2},
        "training_force_rmse_ev_per_angstrom": diagnostics.training_force_rmse,
        "training_relative_force_error": diagnostics.training_relative_force_error,
        "maximum_constraint_residual": diagnostics.maximum_constraint_residual,
        "iterations": diagnostics.iterations,
        "stop_code": diagnostics.stop_code,
        "regularization": diagnostics.regularization,
        "effective_noise_scale_ev_per_angstrom": diagnostics.effective_noise_scale,
        "active_orbits": diagnostics.active_orbits,
        "admm_primal_residual": diagnostics.admm_primal_residual,
        "admm_dual_residual": diagnostics.admm_dual_residual,
        "order_force_rms_ev_per_angstrom": diagnostics.order_force_rms,
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
