#!/usr/bin/env python3
"""Reproduce the official hiPhive direct and long-range-corrected FC2 fits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from hiphive.utilities import prepare_structures
from phonopy.file_IO import write_FORCE_CONSTANTS
from trainstation import Optimizer

ROOT = Path(__file__).resolve().parent
CUTOFF = 11.0


def _fit_metrics(matrix, target, parameters) -> dict[str, float]:
    target_array = np.asarray(target).reshape(-1)
    prediction = np.asarray(matrix @ parameters).reshape(-1)
    residual = prediction - target_array
    return {
        "force_rmse_ev_per_angstrom": float(np.sqrt(np.mean(residual**2))),
        "relative_force_error": float(np.linalg.norm(residual) / np.linalg.norm(target_array)),
    }


def main() -> None:
    unitcell = read(ROOT / "input/NaCl_unitcell.xyz")
    reference = read(ROOT / "supercell.vasp")
    source_frames = read(ROOT / "input/supercells_with_forces.xyz", index=":")
    cluster_space = ClusterSpace(unitcell, [CUTOFF])
    container = StructureContainer(cluster_space)
    for structure in prepare_structures(source_frames, reference):
        container.add_structure(structure)

    matrix, total_target = container.get_fit_data()
    direct_optimizer = Optimizer((matrix, total_target), train_size=1.0)
    direct_optimizer.train()
    direct_fcp = ForceConstantPotential(cluster_space, direct_optimizer.parameters)
    direct_fc2 = direct_fcp.get_force_constants(reference).get_fc_array(order=2)
    write_FORCE_CONSTANTS(direct_fc2, filename=ROOT / "FORCE_CONSTANTS_HIPHIVE_TOTAL")

    long_range_fc2 = np.load(ROOT / "LONG_RANGE_FC2.npy")
    short_frames = read(ROOT / "training-short-range.extxyz", index=":")
    short_target = np.asarray([frame.get_forces() for frame in short_frames]).reshape(-1)
    short_optimizer = Optimizer((matrix, short_target), train_size=1.0)
    short_optimizer.train()
    short_fcp = ForceConstantPotential(cluster_space, short_optimizer.parameters)
    short_fc2 = short_fcp.get_force_constants(reference).get_fc_array(order=2)
    restored_fc2 = short_fc2 + long_range_fc2
    write_FORCE_CONSTANTS(short_fc2, filename=ROOT / "FORCE_CONSTANTS_HIPHIVE_SHORT")
    write_FORCE_CONSTANTS(restored_fc2, filename=ROOT / "FORCE_CONSTANTS_HIPHIVE_RESTORED")

    metrics = {
        "numpy_requirement": "numpy<2.5 (hiPhive/numba compatibility)",
        "cutoff_angstrom": CUTOFF,
        "frames": len(source_frames),
        "parameters": len(direct_optimizer.parameters),
        "direct_total": _fit_metrics(matrix, total_target, direct_optimizer.parameters),
        "short_range": _fit_metrics(matrix, short_target, short_optimizer.parameters),
    }
    (ROOT / "hiphive-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote official-style hiPhive direct and corrected FC2 to {ROOT}")


if __name__ == "__main__":
    main()
