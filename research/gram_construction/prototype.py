"""Benchmark exact alternatives for the complete streamed Gram construction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from ase.io import read
from scipy import sparse
from scipy.linalg.blas import dsyrk
from threadpoolctl import threadpool_limits

from mlfcs.fitting.backends.taylor.features import taylor_axis_derivatives
from mlfcs.fitting.constraints import build_joint_constraints
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.design_operator import ForceDesignOperator
from mlfcs.fitting.fitter import ForceConstantFitter
from mlfcs.fitting.linear_solvers import explicit_constraint_null_space


@dataclass(frozen=True)
class Timings:
    feature: float
    assembly: float
    blas: float
    total: float


def build_case(frames: int, batch_size: int):
    root = Path(__file__).resolve().parents[2] / "examples/fitting/Si/anharmonic/input"
    primitive = read(root / "primitive.vasp")
    reference = read(root / "supercell.vasp")
    snapshots = read(root / "train.extxyz", index=f":{frames}")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: 5.4, 3: 5.4, 4: 4.6},
        max_body_orders={2: 2, 3: 3, 4: 3},
    )
    dataset = FitDataset.from_atoms(fitter.geometry, snapshots)
    constraints = build_joint_constraints(fitter.calculations, acoustic=True).matrix
    reduction = explicit_constraint_null_space(constraints)
    operator = ForceDesignOperator(
        dataset.displacements,
        np.empty(0),
        fitter.order_tensors,
        fitter.n_parameters,
        batch_size,
        parameter_map=reduction,
        axis_derivatives=taylor_axis_derivatives,
    )
    return operator, sparse.csr_matrix(reduction), dataset.forces.reshape(-1)


def _statistics(operator, reduction, target, *, fused: bool, blas_threads: int):
    started = perf_counter()
    feature_seconds = assembly_seconds = blas_seconds = 0.0
    n_reduced = reduction.shape[1]
    gram = np.zeros((n_reduced, n_reduced), order="F")
    rhs = np.zeros(n_reduced)
    rows_per_structure = int(np.prod(operator.force_shape[1:]))
    target = target.reshape(operator.force_shape)
    peak_workspace = 0
    plans = []
    if fused:
        for group in operator.program.groups:
            group_plans = []
            for columns in group.columns:
                local = reduction[columns]
                active = np.unique(local.nonzero()[1])
                group_plans.append((active, local[:, active].toarray()))
            plans.append(tuple(group_plans))

    with threadpool_limits(limits=blas_threads, user_api="blas"):
        for begin in range(0, len(operator.displacements), operator.batch_size):
            end = min(begin + operator.batch_size, len(operator.displacements))
            force_rows = (end - begin) * rows_per_structure
            design_width = n_reduced if fused else operator.n_parameters
            design = np.zeros((force_rows, design_width))
            peak_workspace = max(peak_workspace, design.nbytes)
            displacement_batch = jax.device_put(
                operator.displacements[begin:end], operator.program.device
            )
            for group_index, group in enumerate(operator.program.groups):
                tick = perf_counter()
                contributions = group.kernel(
                    displacement_batch, operator.basis_state, *group.device_arguments
                )
                contributions.block_until_ready()
                values = np.asarray(contributions)
                feature_seconds += perf_counter() - tick
                tick = perf_counter()
                for tile_index, (tile, columns) in enumerate(
                    zip(values, group.columns, strict=True)
                ):
                    tile = tile.reshape(force_rows, -1)
                    if fused:
                        active, local = plans[group_index][tile_index]
                        design[:, active] += tile @ local
                        peak_workspace = max(
                            peak_workspace, design.nbytes + tile.nbytes + local.nbytes
                        )
                    else:
                        design[:, columns] += tile
                        peak_workspace = max(peak_workspace, design.nbytes + tile.nbytes)
                assembly_seconds += perf_counter() - tick
            if not fused:
                tick = perf_counter()
                reduced = np.asarray(reduction.T @ design.T).T
                assembly_seconds += perf_counter() - tick
                peak_workspace = max(peak_workspace, design.nbytes + reduced.nbytes)
                design = reduced
            force = target[begin:end].reshape(-1)
            tick = perf_counter()
            gram = dsyrk(
                1.0, a=design, c=gram, beta=1.0, trans=1, lower=0, overwrite_c=1
            )
            rhs += design.T @ force
            blas_seconds += perf_counter() - tick
    upper = np.triu(gram)
    gram = upper + np.triu(upper, 1).T
    return (
        gram,
        rhs,
        Timings(feature_seconds, assembly_seconds, blas_seconds, perf_counter() - started),
        peak_workspace,
    )


def _warm_up(operator):
    end = min(operator.batch_size, len(operator.displacements))
    displacement_batch = jax.device_put(operator.displacements[:end], operator.program.device)
    for group in operator.program.groups:
        values = group.kernel(
            displacement_batch, operator.basis_state, *group.device_arguments
        )
        values.block_until_ready()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    args = parser.parse_args()
    operator, reduction, target = build_case(args.frames, args.batch_size)
    _warm_up(operator)
    physical = _statistics(
        operator, reduction, target, fused=False, blas_threads=args.blas_threads
    )
    fused = _statistics(operator, reduction, target, fused=True, blas_threads=args.blas_threads)
    gram_scale = max(float(np.linalg.norm(physical[0])), np.finfo(float).tiny)
    rhs_scale = max(float(np.linalg.norm(physical[1])), np.finfo(float).tiny)
    result = {
        "frames": args.frames,
        "batch_size": args.batch_size,
        "blas_threads": args.blas_threads,
        "physical_parameters": operator.n_parameters,
        "reduced_parameters": operator.fit_n_parameters,
        "reduction_nnz": reduction.nnz,
        "physical": {**asdict(physical[2]), "peak_workspace_bytes": physical[3]},
        "fused": {**asdict(fused[2]), "peak_workspace_bytes": fused[3]},
        "fused_to_physical_time": fused[2].total / physical[2].total,
        "fused_to_physical_memory": fused[3] / physical[3],
        "gram_relative_error": float(np.linalg.norm(fused[0] - physical[0]) / gram_scale),
        "rhs_relative_error": float(np.linalg.norm(fused[1] - physical[1]) / rhs_scale),
    }
    if result["fused_to_physical_memory"] > 3.0:
        raise RuntimeError("candidate exceeds the 3x workspace limit")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
