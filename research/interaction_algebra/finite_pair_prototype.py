"""Finite pair-label specialization of the shared indexed orbit traversal."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from indexed_orbit import traverse_indexed_orbit
from scipy import sparse
from scipy.linalg import qr
from sympy.combinatorics import Permutation, PermutationGroup

from mlfcs.force_constants.periodic_fc2 import (
    SupercellHessianSpace,
    _compact_asr,
    _compact_exact_basis,
    _dense_null_space,
    _exact_asr_constraints,
    _finite_pair_basis,
)
from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.algebra.invariants import invariant_basis_from_gram
from mlfcs.interactions.space import ReferenceFrame


@dataclass(frozen=True, slots=True)
class FinitePairGenerator:
    name: str
    mapping: np.ndarray
    action: TensorAction

    def transform(self, states: np.ndarray) -> np.ndarray:
        return self.mapping[np.asarray(states[:, 0], dtype=np.int64), None]


def _all_pair_actions(
    frame: ReferenceFrame,
) -> tuple[tuple[tuple[int, int], ...], tuple[FinitePairGenerator, ...]]:
    relation = frame.relation
    index = relation.index
    labels = tuple(
        (site, atom)
        for site in range(len(relation.primitive))
        for atom in range(len(relation.reference))
    )
    positions = {label: position for position, label in enumerate(labels)}
    generators = []
    for operation, (permutation, cartesian) in enumerate(
        zip(frame.symmetry.atom_permutations, frame.symmetry.cartesian_rotations, strict=True)
    ):
        mapped = []
        for site, atom in labels:
            first = int(permutation[index.representative(site)])
            second = int(permutation[atom])
            second = index.translate_atom(second, -index.translations[first])
            mapped.append((int(index.primitive[first]), second))
        generators.append(
            FinitePairGenerator(
                f"space[{operation}]",
                np.asarray([positions[label] for label in mapped], dtype=np.int32),
                TensorAction(cartesian.T, (0, 1), 2),
            )
        )
    mapped = [
        (int(index.primitive[atom]), index.atom(site, -index.translations[atom]))
        for site, atom in labels
    ]
    generators.append(
        FinitePairGenerator(
            "transpose",
            np.asarray([positions[label] for label in mapped], dtype=np.int32),
            TensorAction(np.eye(3), (1, 0), 2),
        )
    )
    return labels, tuple(generators)


def finite_pair_generators(
    frame: ReferenceFrame,
) -> tuple[tuple[tuple[int, int], ...], tuple[FinitePairGenerator, ...], PermutationGroup]:
    """Use SymPy group order tests to retain a deterministic small generator set."""
    labels, actions = _all_pair_actions(frame)
    permutations = tuple(Permutation([int(value) for value in item.mapping]) for item in actions)
    selected: list[int] = []
    group = PermutationGroup([Permutation(list(range(len(labels))))])
    full = PermutationGroup(list(permutations))
    full.schreier_sims()
    while group.order() < full.order():
        candidates = []
        for index, permutation in enumerate(permutations):
            if index in selected:
                continue
            candidate = PermutationGroup([*(permutations[value] for value in selected), permutation])
            candidates.append((int(candidate.order()), -index, index, candidate))
        _size, _negative, index, group = max(candidates, key=lambda value: value[:2])
        selected.append(index)
    group.schreier_sims()
    return labels, tuple(actions[index] for index in selected), group


def indexed_finite_pair_basis(
    frame: ReferenceFrame, *, tolerance: float = 1e-10
) -> tuple[sparse.csc_matrix, dict[str, object]]:
    labels, generators, sympy_group = finite_pair_generators(frame)
    n_reference = len(frame.relation.reference)
    visited = np.zeros(len(labels), dtype=bool)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    column_offset = 0
    orbit_sizes = []
    orbit_dimensions = []
    traversed_edges = 0
    schreier_constraints = 0
    stabilizer_actions = 0
    started = perf_counter()
    for representative in range(len(labels)):
        if visited[representative]:
            continue
        result = traverse_indexed_orbit(
            np.asarray([representative], dtype=np.int64),
            generators,
            order=2,
            tolerance=tolerance,
        )
        indices = result.states[:, 0].astype(np.int64)
        visited[indices] = True
        invariant = (
            invariant_basis_from_gram(result.constraint_gram, tolerance=tolerance)
            if np.any(result.constraint_gram)
            else np.eye(9)
        )
        invariant = result.seed_to_canonical.apply_columns(invariant)
        invariant /= np.sqrt(len(indices))
        orbit_sizes.append(len(indices))
        orbit_dimensions.append(invariant.shape[1])
        for index, action in zip(indices, result.actions, strict=True):
            site, atom = labels[int(index)]
            values = action.apply_columns(invariant)
            components, local_columns = np.nonzero(np.abs(values) > 1e-13)
            rows.extend((site * n_reference + atom) * 9 + int(value) for value in components)
            columns.extend(column_offset + int(value) for value in local_columns)
            data.extend(
                float(values[component, column])
                for component, column in zip(components, local_columns, strict=True)
            )
        column_offset += invariant.shape[1]
        traversed_edges += result.traversed_edges
        schreier_constraints += result.schreier_constraints
        stabilizer_actions += result.unique_stabilizer_actions
    basis = sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(labels) * 9, column_offset)
    ).tocsc()
    metrics = {
        "pair_count": len(labels),
        "pair_orbit_count": len(orbit_sizes),
        "orbit_sizes": orbit_sizes,
        "orbit_dimensions": orbit_dimensions,
        "symmetry_dimension": column_offset,
        "basis_nnz": int(basis.nnz),
        "generator_count": len(generators),
        "sympy_group_order": int(sympy_group.order()),
        "traversed_edges": traversed_edges,
        "schreier_constraints": schreier_constraints,
        "unique_stabilizer_actions": stabilizer_actions,
        "indexed_seconds": perf_counter() - started,
    }
    return basis, metrics


def _subspace_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] != right.shape[1]:
        return float("inf")
    q_left = qr(left, mode="economic", check_finite=False)[0]
    q_right = qr(right, mode="economic", check_finite=False)[0]
    return float(np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord=2))


def validate_finite_pair_case(frame: ReferenceFrame) -> dict[str, object]:
    started = perf_counter()
    expected = _finite_pair_basis(frame)
    exhaustive_seconds = perf_counter() - started
    actual, metrics = indexed_finite_pair_basis(frame)
    symmetry_error = _subspace_error(expected.toarray(), actual.toarray())
    expected_asr = np.asarray(
        expected @ _dense_null_space((_compact_asr(frame.relation) @ expected).toarray())
    )
    actual_asr = np.asarray(
        actual @ _dense_null_space((_compact_asr(frame.relation) @ actual).toarray())
    )
    asr_error = _subspace_error(expected_asr, actual_asr)
    if symmetry_error > 1e-9 or asr_error > 1e-9:
        raise AssertionError(
            f"finite pair subspace mismatch: symmetry={symmetry_error:.3e}, ASR={asr_error:.3e}"
        )
    return {
        **metrics,
        "asr_dimension": actual_asr.shape[1],
        "maximum_symmetry_subspace_error": symmetry_error,
        "maximum_asr_subspace_error": asr_error,
        "production_seconds": exhaustive_seconds,
    }


def validate_periodic_completion(calculation) -> dict[str, object]:
    """Compare the complete indexed finite/exact complement with production."""
    production = SupercellHessianSpace.build(calculation)
    finite, _metrics = indexed_finite_pair_basis(calculation.frame)
    relation = calculation.frame.relation
    compact = np.asarray(
        finite @ _dense_null_space((_compact_asr(relation) @ finite).toarray())
    )
    exact_constraints = _exact_asr_constraints(calculation.primitive_orbit_space)
    exact_parameter_map = _dense_null_space(exact_constraints.toarray())
    exact_raw = _compact_exact_basis(calculation, relation)
    exact_map = compact.T @ exact_raw @ exact_parameter_map
    left, singular, _right = np.linalg.svd(exact_map, full_matrices=True)
    tolerance = (
        np.finfo(float).eps
        * max(exact_map.shape)
        * (float(singular[0]) if len(singular) else 1.0)
    )
    exact_rank = int(np.count_nonzero(singular > tolerance))
    completion = compact @ left[:, exact_rank:]
    error = _subspace_error(production.completion_basis, completion)
    if error > 1e-9:
        raise AssertionError(f"periodic completion subspace error is {error:.3e}")
    return {
        "exact_dimension": exact_map.shape[1],
        "exact_rank": exact_rank,
        "completion_dimension": completion.shape[1],
        "completion_subspace_error": error,
    }


__all__ = [
    "FinitePairGenerator",
    "finite_pair_generators",
    "indexed_finite_pair_basis",
    "validate_finite_pair_case",
    "validate_periodic_completion",
]
