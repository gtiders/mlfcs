#!/usr/bin/env python3
"""Compare two fourth-order ShengBTE/FourPhonon files semantically.

Blocks are canonicalized over all permutations of their four atomic axes.  This
also changes the anchor atom and translates the other lattice offsets, so files
need not use the same block order or the same representative of a periodic
cluster.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import TextIO

import numpy as np

N_COMPONENTS = 3**4


@dataclass(frozen=True, slots=True)
class Comparison:
    left_raw_blocks: int
    right_raw_blocks: int
    left_blocks: int
    right_blocks: int
    common_blocks: int
    left_only: int
    right_only: int
    common_max_abs: float
    common_rmse: float
    common_relative_l2: float
    union_max_abs: float
    union_rmse: float
    union_relative_l2: float


def _next_nonempty(handle: TextIO) -> str:
    for line in handle:
        if stripped := line.strip():
            return stripped
    raise EOFError("unexpected end of file")


def _canonicalize(
    sites: tuple[int, int, int, int],
    offsets: np.ndarray,
    tensor: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, ...], np.ndarray]:
    quantized = np.rint(offsets / tolerance).astype(np.int64)
    best_key: tuple[int, ...] | None = None
    best_tensor: np.ndarray | None = None
    for permutation in permutations(range(4)):
        anchor = quantized[permutation[0]]
        relative = quantized[list(permutation)] - anchor
        key = tuple(
            value
            for atom, vector in zip(permutation, relative, strict=True)
            for value in (sites[atom], *vector.tolist())
        )
        if best_key is None or key < best_key:
            best_key = key
            best_tensor = tensor.transpose(permutation)
    assert best_key is not None and best_tensor is not None
    return best_key, best_tensor


def _read_fc4(
    path: str | Path, *, tolerance: float = 1e-7
) -> tuple[int, dict[tuple[int, ...], np.ndarray]]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    tensors: dict[tuple[int, ...], list[np.ndarray]] = {}
    with Path(path).open() as handle:
        try:
            block_count = int(_next_nonempty(handle))
        except (EOFError, ValueError) as error:
            raise ValueError(f"{path}: invalid block count") from error
        for expected_block in range(1, block_count + 1):
            try:
                block_number = int(_next_nonempty(handle))
                if block_number != expected_block:
                    raise ValueError(
                        f"expected block {expected_block}, found block {block_number}"
                    )
                translations = np.asarray(
                    [[float(value) for value in _next_nonempty(handle).split()] for _ in range(3)]
                )
                if translations.shape != (3, 3):
                    raise ValueError("each block needs three translation vectors")
                sites = tuple(int(value) for value in _next_nonempty(handle).split())
                if len(sites) != 4:
                    raise ValueError("each block needs four atom indices")
                tensor = np.empty((3, 3, 3, 3), dtype=np.float64)
                seen: set[tuple[int, int, int, int]] = set()
                for _ in range(N_COMPONENTS):
                    fields = _next_nonempty(handle).split()
                    if len(fields) != 5:
                        raise ValueError("each FC4 component needs four directions and one value")
                    directions = tuple(int(value) - 1 for value in fields[:4])
                    if directions in seen or any(direction not in range(3) for direction in directions):
                        raise ValueError(f"invalid or duplicate Cartesian component {directions}")
                    seen.add(directions)
                    tensor[directions] = float(fields[4])
            except (EOFError, ValueError) as error:
                raise ValueError(f"{path}: invalid block {expected_block}: {error}") from error

            # The first atom is the file's origin; the following vectors are
            # lattice translations of atoms 2--4 in Cartesian coordinates.
            offsets = np.vstack((np.zeros(3), translations))
            key, canonical_tensor = _canonicalize(sites, offsets, tensor, tolerance)
            tensors.setdefault(key, []).append(canonical_tensor)

    return block_count, {key: np.mean(values, axis=0) for key, values in tensors.items()}


def read_fc4(path: str | Path, *, tolerance: float = 1e-7) -> dict[tuple[int, ...], np.ndarray]:
    """Read and canonicalize a fourth-order ShengBTE/FourPhonon text file."""
    return _read_fc4(path, tolerance=tolerance)[1]


def compare_fc4(
    left: str | Path,
    right: str | Path,
    *,
    tolerance: float = 1e-7,
) -> Comparison:
    """Compare common support and zero-filled union support of two FC4 files."""
    left_raw_blocks, lhs = _read_fc4(left, tolerance=tolerance)
    right_raw_blocks, rhs = _read_fc4(right, tolerance=tolerance)
    common = lhs.keys() & rhs.keys()
    left_only = lhs.keys() - rhs.keys()
    right_only = rhs.keys() - lhs.keys()

    common_difference = np.concatenate([(lhs[key] - rhs[key]).ravel() for key in common])
    common_reference = np.concatenate([rhs[key].ravel() for key in common])
    union_difference = np.concatenate(
        [common_difference]
        + [lhs[key].ravel() for key in left_only]
        + [-rhs[key].ravel() for key in right_only]
    )
    union_reference = np.concatenate(
        [common_reference]
        + [np.zeros(N_COMPONENTS * len(left_only))]
        + [rhs[key].ravel() for key in right_only]
    )

    def metrics(difference: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
        if difference.size == 0:
            return float("nan"), float("nan"), float("nan")
        denominator = np.linalg.norm(reference)
        relative = np.linalg.norm(difference) / denominator if denominator else float("inf")
        return (
            float(np.max(np.abs(difference))),
            float(np.sqrt(np.mean(difference**2))),
            float(relative),
        )

    common_metrics = metrics(common_difference, common_reference)
    union_metrics = metrics(union_difference, union_reference)
    return Comparison(
        left_raw_blocks, right_raw_blocks, len(lhs), len(rhs), len(common),
        len(left_only), len(right_only),
        *common_metrics, *union_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--geometry-tolerance",
        type=float,
        default=1e-7,
        help="Cartesian translation matching tolerance in angstrom (default: 1e-7)",
    )
    args = parser.parse_args()
    result = compare_fc4(args.left, args.right, tolerance=args.geometry_tolerance)
    print(f"left raw blocks:         {result.left_raw_blocks}")
    print(f"right raw blocks:        {result.right_raw_blocks}")
    print(f"left canonical clusters: {result.left_blocks}")
    print(f"right canonical clusters:{result.right_blocks:>9}")
    print(f"common clusters:         {result.common_blocks}")
    print(f"left-only clusters:      {result.left_only}")
    print(f"right-only clusters:     {result.right_only}")
    print(f"common max |difference|: {result.common_max_abs:.10e}")
    print(f"common RMSE:             {result.common_rmse:.10e}")
    print(f"common relative L2:      {100 * result.common_relative_l2:.8f}%")
    print(f"union max |difference|:  {result.union_max_abs:.10e}")
    print(f"union RMSE:              {result.union_rmse:.10e}")
    print(f"union relative L2:       {100 * result.union_relative_l2:.8f}%")


if __name__ == "__main__":
    main()
