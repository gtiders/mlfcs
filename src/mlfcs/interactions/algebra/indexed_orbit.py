"""NumPy-indexed generator orbit traversal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mlfcs.interactions.algebra.actions import TensorAction, compose_actions, inverse_action


class IndexedGenerator(Protocol):
    """One exact state action and its tensor representation in both frames."""

    name: str
    action: TensorAction

    def transform(self, states: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class IndexedOrbitResult:
    """One canonical orbit with transports and its deduplicated stabilizer actions.

    The traversal publishes the stabilizer *actions* instead of an accumulated floating
    point constraint Gram: the invariant subspace is decided exactly in the lattice
    frame, where their representation matrices are integers, and that decision needs the
    actions themselves rather than their squared sum.
    """

    canonical: np.ndarray
    states: np.ndarray
    actions: tuple[TensorAction, ...]
    seed_to_canonical: TensorAction
    stabilizers: tuple[TensorAction, ...]


def _row_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).ravel()


def _lookup_rows(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Return row positions or -1 using NumPy sorting rather than Python hashes."""
    if not len(needles):
        return np.empty(0, dtype=np.int64)
    if not len(haystack):
        return np.full(len(needles), -1, dtype=np.int64)
    haystack_keys = _row_keys(haystack)
    needle_keys = _row_keys(needles)
    order = np.argsort(haystack_keys)
    sorted_keys = haystack_keys[order]
    positions = np.searchsorted(sorted_keys, needle_keys)
    found = positions < len(sorted_keys)
    safe = np.minimum(positions, len(sorted_keys) - 1)
    found &= sorted_keys[safe] == needle_keys
    result = np.full(len(needles), -1, dtype=np.int64)
    result[found] = order[safe[found]]
    return result


def _canonical_order(states: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    return np.lexsort(tuple(states[:, axis] for axis in reversed(columns)))


def _action_signature(action: TensorAction) -> tuple[bytes, tuple[int, ...]]:
    """Return the exact hashable key of one action.

    The lattice rotation is an exact integer matrix, so its bytes key the action
    directly and the traversal never has to round a floating point rotation to decide
    whether two group elements agree.
    """
    if action.scaled_rotation is None:
        raise ValueError("orbit traversal requires actions in the lattice frame")
    rotation = np.ascontiguousarray(
        np.asarray(action.scaled_rotation, dtype=np.int64).reshape(3, 3)
    )
    return rotation.tobytes(), action.permutation


def _same_action(left: TensorAction, right: TensorAction) -> bool:
    """Compare two actions by their exact integer key."""
    return _action_signature(left) == _action_signature(right)


def traverse_indexed_orbit(
    seed: np.ndarray,
    generators: tuple[IndexedGenerator, ...],
    *,
    order: int,
    canonical_columns: tuple[int, ...] | None = None,
) -> IndexedOrbitResult:
    """Traverse an orbit with integer rows and re-anchor transports canonically.

    Dynamic states are discovered in NumPy batches.  Membership and duplicate detection
    use sorted fixed-width rows, so the traversal carries no floating point tolerance.
    """
    seed = np.asarray(seed, dtype=np.int64).reshape(1, -1)
    identity = TensorAction(np.eye(3), tuple(range(order)), order, np.eye(3, dtype=np.int64))
    states = seed.copy()
    transports: list[TensorAction] = [identity]
    frontier = np.asarray([0], dtype=np.int64)
    stabilizers: dict[tuple[bytes, tuple[int, ...]], TensorAction] = {}

    while len(frontier):
        source_states = states[frontier]
        candidate_rows = []
        candidate_actions: list[TensorAction] = []
        for generator in generators:
            transformed = np.asarray(generator.transform(source_states), dtype=np.int64)
            if transformed.shape != source_states.shape:
                raise ValueError(f"generator {generator.name} returned shape {transformed.shape}")
            candidate_rows.append(transformed)
            candidate_actions.extend(
                compose_actions(generator.action, transports[int(source)]) for source in frontier
            )
        candidates = np.vstack(candidate_rows)
        known = _lookup_rows(states, candidates)

        unknown_locations = np.flatnonzero(known < 0)
        new_indices: list[int] = []
        if len(unknown_locations):
            unknown_rows = candidates[unknown_locations]
            unique_rows, first, inverse = np.unique(
                unknown_rows, axis=0, return_index=True, return_inverse=True
            )
            first_locations = unknown_locations[first]
            base = len(states)
            states = np.vstack((states, unique_rows))
            transports.extend(candidate_actions[int(location)] for location in first_locations)
            new_indices.extend(range(base, len(states)))
            known[unknown_locations] = base + inverse

        for location, target in enumerate(known):
            candidate_action = candidate_actions[location]
            previous_action = transports[int(target)]
            if _same_action(candidate_action, previous_action):
                continue
            stabilizer = compose_actions(inverse_action(previous_action), candidate_action)
            stabilizers.setdefault(_action_signature(stabilizer), stabilizer)
        frontier = np.asarray(new_indices, dtype=np.int64)

    columns = tuple(range(states.shape[1])) if canonical_columns is None else canonical_columns
    if sorted(columns) != list(range(states.shape[1])):
        raise ValueError("canonical_columns must be a permutation of state columns")
    canonical_order = _canonical_order(states, columns)
    canonical_index = int(canonical_order[0])
    canonical_inverse = inverse_action(transports[canonical_index])
    anchored = tuple(compose_actions(action, canonical_inverse) for action in transports)
    sorted_states = states[canonical_order]
    sorted_actions = tuple(anchored[int(index)] for index in canonical_order)
    return IndexedOrbitResult(
        canonical=sorted_states[0].copy(),
        states=sorted_states,
        actions=sorted_actions,
        seed_to_canonical=transports[canonical_index],
        stabilizers=tuple(stabilizers.values()),
    )


__all__ = [
    "IndexedGenerator",
    "IndexedOrbitResult",
    "compose_actions",
    "inverse_action",
    "traverse_indexed_orbit",
]
