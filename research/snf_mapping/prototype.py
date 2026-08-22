"""Independent full-SNF translation-mapping research prototype.

This module is deliberately outside ``src/mlfcs``.  It compares the current
row-HNF quotient with finite-group coordinates obtained from a complete Smith
decomposition ``D = U @ S @ V``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product

import numpy as np
from sympy import ZZ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp

from mlfcs.structure.reciprocal import quotient_qpoints
from mlfcs.structure.integer_lattice import (
    IntegerLatticeQuotient,
    adjugate_3x3,
    determinant_3x3,
    normalize_supercell_matrix,
    residue_key,
)


def _int64_domain_matrix(matrix: np.ndarray) -> DomainMatrix:
    values = normalize_supercell_matrix(matrix)
    entries = [[ZZ(int(value)) for value in row] for row in values]
    return DomainMatrix(entries, values.shape, ZZ)


def _checked_int64(matrix: DomainMatrix) -> np.ndarray:
    values = [[int(value) for value in row] for row in matrix.to_Matrix().tolist()]
    limit = np.iinfo(np.int64)
    if any(value < limit.min or value > limit.max for row in values for value in row):
        raise OverflowError("SNF result does not fit in int64")
    return np.asarray(values, dtype=np.int64)


def _unimodular_inverse(matrix: np.ndarray) -> np.ndarray:
    determinant = determinant_3x3(matrix)
    if abs(determinant) != 1:
        raise ValueError("matrix is not unimodular")
    return adjugate_3x3(matrix) // determinant


@dataclass(frozen=True, slots=True)
class SmithCoordinates:
    """Finite-group coordinates for the row quotient ``Z^3 / Z^3 S``."""

    matrix: np.ndarray
    diagonal: np.ndarray
    left: np.ndarray
    right: np.ndarray
    right_inverse: np.ndarray
    strides: np.ndarray

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> SmithCoordinates:
        values = normalize_supercell_matrix(matrix)
        diagonal_dm, left_dm, right_dm = smith_normal_decomp(_int64_domain_matrix(values))
        diagonal_matrix = _checked_int64(diagonal_dm)
        left = _checked_int64(left_dm)
        right = _checked_int64(right_dm)
        diagonal = np.diag(diagonal_matrix).copy()
        if np.any(diagonal <= 0):
            raise RuntimeError("SNF diagonal must be positive for a nonsingular matrix")
        if not np.array_equal(left @ values @ right, diagonal_matrix):
            raise RuntimeError("SNF decomposition identity failed")
        if abs(determinant_3x3(left)) != 1 or abs(determinant_3x3(right)) != 1:
            raise RuntimeError("SNF transformations are not unimodular")
        if diagonal[1] % diagonal[0] or diagonal[2] % diagonal[1]:
            raise RuntimeError("SNF invariant factors do not form a divisibility chain")
        expected = abs(determinant_3x3(values))
        if int(np.prod(diagonal, dtype=object)) != expected:
            raise RuntimeError("SNF group size differs from the determinant")
        strides = np.asarray(
            (int(diagonal[1]) * int(diagonal[2]), int(diagonal[2]), 1),
            dtype=np.int64,
        )
        return cls(
            values,
            diagonal,
            left,
            right,
            _unimodular_inverse(right),
            strides,
        )

    @property
    def size(self) -> int:
        return int(np.prod(self.diagonal, dtype=object))

    @property
    def group_coordinates(self) -> np.ndarray:
        ranges = (range(int(value)) for value in self.diagonal)
        return np.asarray(tuple(product(*ranges)), dtype=np.int64).reshape((-1, 3))

    @property
    def representatives(self) -> np.ndarray:
        """Return one original-coordinate translation for every group element."""
        return self.group_coordinates @ self.right_inverse

    def coordinates_many(self, translations: np.ndarray) -> np.ndarray:
        values = np.asarray(translations, dtype=np.int64)
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("translations must end in shape (3,)")
        return np.mod(values @ self.right, self.diagonal)

    def cell_index_many(self, translations: np.ndarray) -> np.ndarray:
        coordinates = self.coordinates_many(translations)
        return np.sum(coordinates * self.strides, axis=-1, dtype=np.int64)

    def qpoints(self) -> np.ndarray:
        """Return reciprocal characters paired with the direct SNF group."""
        scaled = self.group_coordinates / self.diagonal
        return np.mod(scaled @ self.right.T, 1.0)


def _sorted_points(points: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(points), decimals=13)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


def validate_matrix(matrix: np.ndarray, *, random_count: int, seed: int) -> dict[str, object]:
    values = normalize_supercell_matrix(matrix)
    smith = SmithCoordinates.from_matrix(values)
    hnf = IntegerLatticeQuotient(values)
    rng = np.random.default_rng(seed)
    translations = rng.integers(-100_000, 100_001, size=(random_count, 3), dtype=np.int64)

    smith_coordinates = smith.coordinates_many(translations)
    smith_roundtrip = smith.coordinates_many(smith.representatives)
    if not np.array_equal(smith_roundtrip, smith.group_coordinates):
        raise RuntimeError("SNF representative round trip failed")

    sample = min(random_count, 10_000)
    offsets = rng.integers(-20, 21, size=(sample, 3), dtype=np.int64) @ values
    shifted = translations[:sample] + offsets
    if not np.array_equal(
        smith.coordinates_many(translations[:sample]),
        smith.coordinates_many(shifted),
    ):
        raise RuntimeError("SNF coordinates are not invariant under supercell translations")
    for location in range(min(sample, 2_000)):
        difference = translations[location] - shifted[location]
        if residue_key(difference, values) != (0, 0, 0):
            raise RuntimeError("SNF and residue equivalence disagree")

    hnf_classes = hnf.cell_index_many(translations)
    smith_classes = smith.cell_index_many(translations)
    # Index numbers need not agree, but equality partitions must agree.
    pairs = rng.integers(0, random_count, size=(min(random_count, 20_000), 2))
    if not np.array_equal(
        hnf_classes[pairs[:, 0]] == hnf_classes[pairs[:, 1]],
        smith_classes[pairs[:, 0]] == smith_classes[pairs[:, 1]],
    ):
        raise RuntimeError("HNF and SNF quotient partitions disagree")

    snf_qpoints = smith.qpoints()
    current_qpoints = quotient_qpoints(values)
    if not np.array_equal(_sorted_points(snf_qpoints), _sorted_points(current_qpoints)):
        raise RuntimeError("SNF and current reciprocal quotients disagree")
    if not np.allclose(
        snf_qpoints @ values.T,
        np.rint(snf_qpoints @ values.T),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("SNF q points violate the row-vector commensurability condition")

    return {
        "matrix": values.tolist(),
        "determinant": abs(determinant_3x3(values)),
        "diagonal": smith.diagonal.tolist(),
        "left": smith.left.tolist(),
        "right": smith.right.tolist(),
        "random_translations": random_count,
        "unique_snf_coordinates": len(np.unique(smith_coordinates, axis=0)),
        "qpoint_count": len(snf_qpoints),
    }


def validate_random_decompositions(*, count: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    while accepted < count:
        attempts += 1
        matrix = rng.integers(-8, 9, size=(3, 3), dtype=np.int64)
        determinant = determinant_3x3(matrix)
        if determinant == 0 or abs(determinant) > 4096:
            continue
        SmithCoordinates.from_matrix(matrix)
        accepted += 1
    return {"accepted": accepted, "attempts": attempts}


def benchmark_lookup(
    matrix: np.ndarray,
    *,
    lookups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    values = normalize_supercell_matrix(matrix)
    hnf = IntegerLatticeQuotient(values)
    smith = SmithCoordinates.from_matrix(values)
    rng = np.random.default_rng(seed)
    translations = rng.integers(-100_000, 100_001, size=(lookups, 3), dtype=np.int64)
    hnf.cell_index_many(translations[:100])
    smith.cell_index_many(translations[:100])
    hnf_times: list[float] = []
    smith_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        hnf.cell_index_many(translations)
        hnf_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        smith.cell_index_many(translations)
        smith_times.append(time.perf_counter() - started)
    hnf_best = min(hnf_times)
    smith_best = min(smith_times)
    return {
        "matrix": values.tolist(),
        "lookups": lookups,
        "hnf_seconds": hnf_best,
        "snf_seconds": smith_best,
        "snf_over_hnf": smith_best / hnf_best,
    }


def basis_dependence(matrix: np.ndarray) -> dict[str, object]:
    values = normalize_supercell_matrix(matrix)
    change = np.asarray(((1, 1, 0), (0, 1, 0), (0, 0, 1)), dtype=np.int64)
    original = SmithCoordinates.from_matrix(values)
    changed = SmithCoordinates.from_matrix(change @ values)
    probes = np.asarray(tuple(product(range(-2, 3), repeat=3)), dtype=np.int64)
    original_coordinates = original.coordinates_many(probes)
    changed_coordinates = changed.coordinates_many(probes)
    differing = np.flatnonzero(np.any(original_coordinates != changed_coordinates, axis=1))
    probe_index = int(differing[0]) if len(differing) else 0
    probe = probes[probe_index]
    return {
        "same_sublattice": True,
        "same_invariant_factors": bool(np.array_equal(original.diagonal, changed.diagonal)),
        "same_right_transform": bool(np.array_equal(original.right, changed.right)),
        "probe": probe.tolist(),
        "original_coordinate": original_coordinates[probe_index].tolist(),
        "changed_coordinate": changed_coordinates[probe_index].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-matrices", type=int, default=2_000)
    parser.add_argument("--random-translations", type=int, default=50_000)
    parser.add_argument("--lookups", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    matrices = (
        np.diag((2, 2, 2)),
        np.asarray(((2, 1, 0), (0, 2, 1), (0, 0, 2)), dtype=np.int64),
        np.asarray(((4, 7, -3), (0, 5, 9), (0, 0, 3)), dtype=np.int64),
    )
    result = {
        "sympy_path": "sympy.polys.matrices.normalforms.smith_normal_decomp",
        "validation": [
            validate_matrix(
                matrix,
                random_count=args.random_translations,
                seed=args.seed + location,
            )
            for location, matrix in enumerate(matrices)
        ],
        "random_decompositions": validate_random_decompositions(
            count=args.random_matrices,
            seed=args.seed,
        ),
        "lookup_benchmarks": [
            benchmark_lookup(
                matrix,
                lookups=args.lookups,
                repeats=args.repeats,
                seed=args.seed + location,
            )
            for location, matrix in enumerate(matrices)
        ],
        "basis_dependence": basis_dependence(matrices[0]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
