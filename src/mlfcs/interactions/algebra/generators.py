"""Generator-action protocols and finite-group helpers."""

from __future__ import annotations

from typing import Protocol

from sympy.combinatorics import Permutation, PermutationGroup

from mlfcs.interactions.algebra.actions import TensorAction


class GeneratorAction(Protocol):
    name: str
    action: TensorAction

    def transform(self, states): ...


def select_group_generators(
    permutations: tuple[Permutation, ...],
) -> tuple[Permutation, ...]:
    """Select deterministic generators by greedily increasing group order."""
    if not permutations:
        raise ValueError("at least one permutation is required")
    identities = [value for value in permutations if value.is_Identity]
    if len(identities) != 1:
        raise ValueError("permutations must contain exactly one identity")
    selected: list[Permutation] = []
    group = PermutationGroup([identities[0]])
    expected_order = len(permutations)
    while group.order() < expected_order:
        candidates = []
        for index, permutation in enumerate(permutations):
            if permutation.is_Identity or permutation in selected:
                continue
            candidate = PermutationGroup([*selected, permutation])
            candidates.append((int(candidate.order()), -index, permutation, candidate))
        if not candidates:
            raise ValueError("permutations are not closed under the generated group")
        _order, _negative_index, permutation, group = max(
            candidates, key=lambda value: value[:2]
        )
        selected.append(permutation)
    group.schreier_sims()
    if group.order() != expected_order:
        raise ValueError(
            f"generator group has order {group.order()}, expected {expected_order}"
        )
    return tuple(selected)


def validate_group_order(generators: tuple[Permutation, ...], expected_order: int) -> None:
    actual = int(PermutationGroup(list(generators)).order())
    if actual != expected_order:
        raise ValueError(f"generator group has order {actual}, expected {expected_order}")


__all__ = ["GeneratorAction", "select_group_generators", "validate_group_order"]
