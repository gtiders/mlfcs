#!/usr/bin/env python3
"""Reproduce Taylor/Wick fitting comparisons without modifying example results."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs import ForceConstantFitter, write_force_constants

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    primitive: Path
    reference: Path
    snapshots: Path
    orders: tuple[int, ...]
    cutoffs: dict[int, float]
    body_orders: dict[int, int]
    validation_split: float
    batch_size: int
    regularization: str | None = None
    symprec: float = 1e-5


CASES = {
    "si": Case(
        ROOT / "examples/fitting/Si/anharmonic/input/primitive.vasp",
        ROOT / "examples/fitting/Si/anharmonic/input/supercell.vasp",
        ROOT / "examples/fitting/Si/anharmonic/input/train.extxyz",
        (2, 3, 4),
        {2: 5.4, 3: 5.4, 4: 4.6},
        {2: 2, 3: 3, 4: 3},
        0.0,
        4,
        "scaled_group_lasso",
    ),
    "snse": Case(
        ROOT / "examples/fitting/SnSe/input/primitive.vasp",
        ROOT / "examples/fitting/SnSe/input/reference.vasp",
        ROOT / "examples/fitting/SnSe/md/T300K/nve.extxyz",
        (2, 3, 4),
        {2: 8.0, 3: 6.5, 4: 4.5},
        {2: 2, 3: 3, 4: 3},
        0.1,
        4,
        symprec=1e-4,
    ),
    "ba": Case(
        ROOT / "examples/fitting/Ba8Ga16Ge30/input/primitive.vasp",
        ROOT / "examples/fitting/Ba8Ga16Ge30/input/reference.vasp",
        ROOT / "examples/fitting/Ba8Ga16Ge30/md/results/T300K/nve.extxyz",
        (2, 3),
        {2: 5.4, 3: 4.35},
        {2: 2, 3: 2},
        0.0,
        1,
        symprec=1e-4,
    ),
}


def fit_case(name: str, case: Case, basis: str, output: Path):
    destination = output / name / basis
    destination.mkdir(parents=True, exist_ok=False)
    fitter = ForceConstantFitter(
        read(case.primitive),
        read(case.reference),
        orders=case.orders,
        cutoffs=case.cutoffs,
        max_body_orders=case.body_orders,
        symprec=case.symprec,
        fitting_basis=basis,
        verbose=True,
    )
    result = fitter.fit(
        read(case.snapshots, index=":"),
        validation_split=case.validation_split,
        batch_size=case.batch_size,
        regularization=case.regularization,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
        cache_directory=destination / "cache",
    )
    write_force_constants(result.force_constants, destination / "mlfcs.h5", format="hdf5")
    (destination / "metrics.json").write_text(
        json.dumps(asdict(result.diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def compare(name: str, results: dict[str, object]) -> dict[str, object]:
    taylor = results["taylor"]
    wick = results["wick"]
    orders = {}
    for order in taylor.force_constants.orders:
        left = taylor.force_constants.sparse[order]
        right = wick.force_constants.sparse[order]
        if not np.array_equal(left.sites, right.sites):
            raise RuntimeError(f"{name} FC{order}: site keys differ")
        if not np.array_equal(left.translations, right.translations):
            raise RuntimeError(f"{name} FC{order}: translation keys differ")
        difference = np.asarray(left.tensors) - np.asarray(right.tensors)
        orders[str(order)] = {
            "rows": len(left.tensors),
            "relative_l2_difference": float(
                np.linalg.norm(difference) / max(np.linalg.norm(right.tensors), 1e-300)
            ),
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
        }
    return {
        "case": name,
        "taylor": {
            "training_force_rmse": taylor.diagnostics.training_force_rmse,
            "validation_force_rmse": taylor.diagnostics.validation_force_rmse,
            "maximum_constraint_residual": taylor.diagnostics.maximum_constraint_residual,
        },
        "wick": {
            "training_force_rmse": wick.diagnostics.training_force_rmse,
            "validation_force_rmse": wick.diagnostics.validation_force_rmse,
            "maximum_constraint_residual": wick.diagnostics.maximum_constraint_residual,
        },
        "orders": orders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="mlfcs-fitting-bases-"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.cases:
        case_results = {
            basis: fit_case(name, CASES[name], basis, output) for basis in ("taylor", "wick")
        }
        summary.append(compare(name, case_results))
        (output / "comparison.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"wrote comparison to {output}")


if __name__ == "__main__":
    main()
