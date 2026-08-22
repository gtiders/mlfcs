"""独立的 Direct-Gram 研究 prototype。

这个文件不属于 MLFCS 运行时路径。它把当前 bounded design tiles 收集后，
分别用显式设计矩阵和 tile 两两 contraction 计算统计量，用于验证：

    G = X.T @ X,    b = X.T @ y

tile 两两 contraction 是严格等价的 Direct-Gram 候选，但故意保留了它的
平方级代价，以便在小型 Si 案例上测量并在大案例上拒绝。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs.basis.wick import symmetrized_covariance
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.design import ForceDesignOperator
from mlfcs.fitting.fitter import ForceConstantFitter


@dataclass(frozen=True)
class Tile:
    columns: np.ndarray
    values: np.ndarray


def _case(name: str):
    root = Path(__file__).resolve().parents[2] / "examples" / "fitting"
    if name == "si":
        case = root / "Si" / "anharmonic"
        return case, case / "input/primitive.vasp", case / "input/supercell.vasp", case / "input/train.extxyz", {
            2: 5.4,
            3: 5.4,
            4: 4.6,
        }, {2: 2, 3: 3, 4: 3}
    if name == "snse":
        case = root / "SnSe"
        return case, case / "input/primitive.vasp", case / "input/reference.vasp", case / "md/T300K/nve.extxyz", {
            2: 8.0,
            3: 6.5,
            4: 4.5,
        }, {2: 2, 3: 3, 4: 3}
    raise ValueError(f"unknown case: {name}")


def _operator(name: str, orders: tuple[int, ...], frames: int):
    case, primitive_path, reference_path, snapshots_path, cutoffs, bodies = _case(name)
    primitive = read(primitive_path)
    reference = read(reference_path)
    snapshots = read(snapshots_path, index=f":{frames}")
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=orders,
        cutoffs={order: cutoffs[order] for order in orders},
        max_body_orders={order: bodies[order] for order in orders},
        symprec=1e-4 if name == "snse" else 1e-5,
        verbose=False,
    )
    dataset = FitDataset.from_atoms(fitter.geometry, snapshots)
    covariance = symmetrized_covariance(dataset.displacements, fitter.calculations[0])
    operator = ForceDesignOperator(
        dataset.displacements,
        covariance,
        fitter.order_tensors,
        fitter.n_parameters,
        batch_size=4,
    )
    return case, fitter, dataset, operator


def collect_tiles(operator: ForceDesignOperator, target: np.ndarray) -> tuple[list[Tile], np.ndarray]:
    """Collect bounded tiles only; this is intentionally a small-case tool."""
    tiles: list[Tile] = []
    rows_per_structure = int(np.prod(operator.force_shape[1:]))
    target = np.asarray(target).reshape(operator.force_shape).reshape(-1)
    for begin in range(0, len(operator.displacements), operator.batch_size):
        end = min(begin + operator.batch_size, len(operator.displacements))
        displacement_batch = operator.displacements[begin:end]
        for group in operator.program.groups:
            contributions = group.kernel(
                displacement_batch,
                operator.covariance,
                *group.device_arguments,
            )
            contributions = np.asarray(contributions)
            for contribution, columns in zip(contributions, group.columns, strict=True):
                values = contribution.reshape((end - begin) * rows_per_structure, -1)
                tiles.append(Tile(np.asarray(columns, dtype=np.int32), values))
    return tiles, target


def explicit_statistics(tiles: list[Tile], n_parameters: int, target: np.ndarray):
    rows = target.size
    design = np.zeros((rows, n_parameters), dtype=np.float64)
    for tile in tiles:
        design[:, tile.columns] += tile.values
    return design, design.T @ design, design.T @ target


def tile_pair_statistics(tiles: list[Tile], n_parameters: int, target: np.ndarray):
    """Exact tile-pair Direct-Gram candidate; unsuitable for large cases."""
    gram = np.zeros((n_parameters, n_parameters), dtype=np.float64)
    rhs = np.zeros(n_parameters, dtype=np.float64)
    for tile in tiles:
        rhs[tile.columns] += tile.values.T @ target
    for left, first in enumerate(tiles):
        for right in range(left, len(tiles)):
            second = tiles[right]
            block = first.values.T @ second.values
            gram[np.ix_(first.columns, second.columns)] += block
            if right != left:
                gram[np.ix_(second.columns, first.columns)] += block.T
    return gram, rhs


def cost_report(tiles: list[Tile], n_parameters: int, rows: int, dof: int, max_degree: int):
    tile_columns = [len(tile.columns) for tile in tiles]
    pair_flops = 2 * rows * sum(
        tile_columns[left] * tile_columns[right]
        for left in range(len(tile_columns))
        for right in range(left, len(tile_columns))
    )
    explicit_flops = 2 * rows * n_parameters * n_parameters
    return {
        "tiles": len(tiles),
        "parameters": n_parameters,
        "force_rows": rows,
        "tile_values_bytes": int(sum(tile.values.nbytes for tile in tiles)),
        "explicit_design_bytes": int(rows * n_parameters * 8),
        "tile_pair_flops": int(pair_flops),
        "explicit_gram_flops": int(explicit_flops),
        "tile_pair_to_explicit_flop_ratio": float(pair_flops / explicit_flops)
        if explicit_flops
        else float("inf"),
        "moment_degree": max_degree,
        "moment_symmetric_components": comb(dof + max_degree - 1, max_degree),
        "moment_float64_bytes": int(comb(dof + max_degree - 1, max_degree) * 8),
    }


def run(name: str, orders: tuple[int, ...], frames: int, *, collect: bool):
    _case_path, fitter, dataset, operator = _operator(name, orders, frames)
    metadata = {
        "case": name,
        "frames": frames,
        "orders": orders,
        "parameters": fitter.n_parameters,
        "force_rows": int(dataset.forces.size),
        "dof": int(np.prod(dataset.displacements.shape[1:])),
        "orbits": {str(p.order): len(p.parameter_indices) for p in fitter.order_tensors},
        "order_parameters": {
            str(p.order): int(np.count_nonzero(p.parameter_mask)) for p in fitter.order_tensors
        },
        "order_images": {
            str(p.order): int(np.sum(p.image_mask)) for p in fitter.order_tensors
        },
        "translation_cells": {
            str(p.order): int(p.coordinates.shape[2]) for p in fitter.order_tensors
        },
        "design_signatures": len(operator.program.groups),
        "design_tiles": operator.program.tile_count,
        "static_device_bytes": operator.program.static_device_bytes,
    }
    if not collect:
        return metadata
    target = dataset.forces.reshape(-1)
    tiles, target = collect_tiles(operator, target)
    design, gram_explicit, rhs_explicit = explicit_statistics(tiles, fitter.n_parameters, target)
    gram_direct, rhs_direct = tile_pair_statistics(tiles, fitter.n_parameters, target)
    metadata["tile_pair"] = cost_report(
        tiles,
        fitter.n_parameters,
        target.size,
        metadata["dof"],
        2 * max(orders) - 2,
    )
    metadata["explicit_design_bytes"] = int(design.nbytes)
    metadata["gram_max_abs_error"] = float(np.max(np.abs(gram_direct - gram_explicit)))
    metadata["rhs_max_abs_error"] = float(np.max(np.abs(rhs_direct - rhs_explicit)))
    metadata["gram_relative_error"] = float(
        np.linalg.norm(gram_direct - gram_explicit) / max(np.linalg.norm(gram_explicit), 1e-300)
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("si", "snse"), required=True)
    parser.add_argument("--orders", nargs="+", type=int, default=[2])
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = run(args.case, tuple(sorted(set(args.orders))), args.frames, collect=not args.metadata_only)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
