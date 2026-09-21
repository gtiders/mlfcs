"""Generator-built primitive interaction spaces in the reduced lattice frame.

Every orbit is decided in the lattice (scaled) frame of a Minkowski-reduced primitive
cell, where spglib rotations are integer for every cell: an fcc primitive 60 degree cell,
hexagonal and rhombohedral cells have irrational entries in their Cartesian rotations
only.  The reduced frame is canonical, so two unimodular representations of the same
crystal produce the *same* integer algebra instead of the same algebra at wildly
different integer sizes.

The orbit keys that leave this module are anchored in the user's source cell, because
that is the frame the reference supercell and its atom index live in.  The lattice frame
carries the exact integer map between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from sympy.combinatorics import Permutation, PermutationGroup

from mlfcs.interactions.algebra.actions import TensorAction, scaled_to_cartesian_rotation
from mlfcs.interactions.algebra.generators import select_group_generators
from mlfcs.interactions.algebra.indexed_orbit import traverse_indexed_orbit
from mlfcs.interactions.algebra.invariants import invariant_kernel, label_symmetric_basis
from mlfcs.interactions.algebra.rendering import (
    cartesian_orbit_basis,
    observation_condition,
    observation_matrix,
    orthonormal_orbit_basis,
    select_observation_rows,
)
from mlfcs.interactions.keys import InteractionKey
from mlfcs.interactions.models import (
    PrimitiveInteractionOrbit,
    PrimitiveInteractionSpace,
    PrimitiveOrbitImage,
)
from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
from mlfcs.structure.lattice_frame import LatticeFrame
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


def build_primitive_interaction_space(
    primitive: Atoms,
    *,
    order: int,
    cutoff: float | None,
    max_body_order: int | None,
    symprec: float,
    frame: LatticeFrame | None = None,
) -> PrimitiveInteractionSpace:
    """Build primitive orbits through indexed generator traversal."""
    from mlfcs.interactions.primitive.candidates import iter_primitive_candidates

    primitive = primitive.copy()
    primitive.wrap()
    radius = resolve_primitive_cutoff(primitive, cutoff)
    frame = LatticeFrame.from_atoms(primitive, symprec=symprec) if frame is None else frame
    algebra = frame.algebra_atoms()
    symmetry = PrimitiveSymmetryOperations.from_atoms(algebra, symprec=symprec)
    generators, _group = primitive_generators(symmetry, order, cell=frame.algebra_cell)
    generated: dict[InteractionKey, GeneratedPrimitiveOrbit] = {}
    covered: set[InteractionKey] = set()
    for seed in iter_primitive_candidates(
        algebra, radius=radius, order=order, max_body_order=max_body_order
    ):
        if seed in covered:
            continue
        orbit = generated_orbit(seed, generators, frame=frame)
        generated[orbit.representative] = orbit
        covered.update(decode_key(row) for row in orbit.states)
    result = []
    for representative in sorted(generated):
        orbit = generated[representative]
        if orbit.dimension == 0:
            continue
        images = tuple(
            PrimitiveOrbitImage(source_key(frame, row), action)
            for row, action in zip(orbit.states, orbit.actions, strict=True)
        )
        result.append(
            PrimitiveInteractionOrbit(
                source_key(frame, encode_key(representative)),
                orbit.exact_lattice_basis,
                orbit.cartesian_basis,
                orbit.coefficient_transform,
                orbit.observation_rows,
                orbit.observation_condition,
                images,
            )
        )
    return PrimitiveInteractionSpace(primitive, order, radius, max_body_order, frame, tuple(result))


def encode_key(key: InteractionKey) -> np.ndarray:
    return np.asarray(key.labels, dtype=np.int64).reshape(-1)


def decode_key(values: np.ndarray) -> InteractionKey:
    rows = np.asarray(values, dtype=np.int64).reshape(-1, 4)
    return InteractionKey.from_labels(rows)


def source_key(frame: LatticeFrame, values: np.ndarray) -> InteractionKey:
    """Return the source-cell key of one lattice-frame interaction row.

    Sites are renamed through the motif match of the frame and translations go through the
    exact integer map of the reduction, so the reference supercell index of the user's
    structure addresses the same physical interaction.
    """
    rows = np.asarray(values, dtype=np.int64).reshape(-1, 4)
    return InteractionKey.from_labels(frame.source_labels(rows))


def _operation_signature(symmetry: PrimitiveSymmetryOperations, operation: int) -> tuple:
    shifts = np.asarray(symmetry.site_shifts[operation], dtype=np.int64)
    shifts = shifts - shifts[0]
    return (
        tuple(np.asarray(symmetry.rotations[operation], dtype=np.int64).reshape(-1)),
        tuple(np.asarray(symmetry.site_permutations[operation], dtype=np.int64)),
        tuple(shifts.reshape(-1)),
    )


def operation_composition_table(symmetry: PrimitiveSymmetryOperations) -> np.ndarray:
    """Build the exact affine operation table used to create a SymPy group."""
    lookup = {
        _operation_signature(symmetry, operation): operation for operation in range(symmetry.size)
    }
    table = np.empty((symmetry.size, symmetry.size), dtype=np.int32)
    for after in range(symmetry.size):
        for before in range(symmetry.size):
            before_sites = symmetry.site_permutations[before]
            rotation = symmetry.rotations[after] @ symmetry.rotations[before]
            sites = symmetry.site_permutations[after, before_sites]
            shifts = (
                symmetry.site_shifts[before] @ symmetry.rotations[after].T
                + symmetry.site_shifts[after, before_sites]
            )
            shifts -= shifts[0]
            signature = (
                tuple(np.asarray(rotation, dtype=np.int64).reshape(-1)),
                tuple(np.asarray(sites, dtype=np.int64)),
                tuple(np.asarray(shifts, dtype=np.int64).reshape(-1)),
            )
            table[after, before] = lookup[signature]
    return table


def sympy_space_group_generators(
    symmetry: PrimitiveSymmetryOperations,
) -> tuple[tuple[int, ...], PermutationGroup]:
    """Select deterministic affine generators using SymPy group orders."""
    table = operation_composition_table(symmetry)
    regular = tuple(
        Permutation([int(table[operation, value]) for value in range(symmetry.size)])
        for operation in range(symmetry.size)
    )
    selected_permutations = select_group_generators(regular)
    selected = tuple(regular.index(permutation) for permutation in selected_permutations)
    group = PermutationGroup(list(selected_permutations))
    group.schreier_sims()
    if group.order() != symmetry.size:
        raise RuntimeError("SymPy regular permutation group has the wrong order")
    return selected, group


@dataclass(frozen=True, slots=True)
class PrimitiveGenerator:
    name: str
    symmetry: PrimitiveSymmetryOperations
    operation: int | None
    permutation: tuple[int, ...]
    action: TensorAction

    def transform(self, states: np.ndarray) -> np.ndarray:
        result = np.empty_like(states)
        for row, values in enumerate(states):
            labels = np.asarray(values, dtype=np.int64).reshape(-1, 4)
            if self.operation is not None:
                transformed = np.empty_like(labels)
                sites = labels[:, 0].astype(np.int32)
                transformed[:, 0] = self.symmetry.site_permutations[self.operation, sites]
                transformed[:, 1:] = (
                    labels[:, 1:] @ self.symmetry.rotations[self.operation].T
                    + self.symmetry.site_shifts[self.operation, sites]
                )
                labels = transformed
            labels = labels[np.asarray(self.permutation, dtype=np.int32)]
            labels[:, 1:] -= labels[0, 1:]
            result[row] = labels.reshape(-1)
        return result


def primitive_generators(
    symmetry: PrimitiveSymmetryOperations, order: int, *, cell: np.ndarray
) -> tuple[tuple[PrimitiveGenerator, ...], PermutationGroup]:
    """Return the orbit generators, each carrying both frames of its rotation.

    The lattice rotation is spglib's integer matrix and the Cartesian rotation is derived
    from it through the cell, so the two frames of one generator cannot disagree and the
    transpose convention lives in :func:`scaled_to_cartesian_rotation` alone.
    """
    operations, group = sympy_space_group_generators(symmetry)
    identity = tuple(range(order))
    generators = []
    for operation in operations:
        scaled = np.asarray(symmetry.rotations[operation], dtype=np.int64)
        generators.append(
            PrimitiveGenerator(
                f"space[{operation}]",
                symmetry,
                operation,
                identity,
                TensorAction(
                    scaled_to_cartesian_rotation(scaled, cell),
                    identity,
                    order,
                    scaled,
                ),
            )
        )
    for axis in range(order - 1):
        permutation = list(identity)
        permutation[axis], permutation[axis + 1] = permutation[axis + 1], permutation[axis]
        value = tuple(permutation)
        generators.append(
            PrimitiveGenerator(
                f"swap[{axis},{axis + 1}]",
                symmetry,
                None,
                value,
                TensorAction(np.eye(3), value, order, np.eye(3, dtype=np.int64)),
            )
        )
    return tuple(generators), group


@dataclass(frozen=True, slots=True)
class GeneratedPrimitiveOrbit:
    """One lattice-frame orbit before its keys are mapped back to the source cell."""

    representative: InteractionKey
    exact_lattice_basis: np.ndarray
    cartesian_basis: np.ndarray
    coefficient_transform: np.ndarray
    observation_rows: np.ndarray
    observation_condition: float
    states: np.ndarray
    actions: tuple[TensorAction, ...]

    @property
    def dimension(self) -> int:
        return int(self.cartesian_basis.shape[1])


def generated_orbit(
    seed: InteractionKey,
    generators: tuple[PrimitiveGenerator, ...],
    *,
    frame: LatticeFrame,
) -> GeneratedPrimitiveOrbit:
    """Build one orbit with its two bases and its observed components.

    The invariant subspace, its certified dimension and the exact basis come from integer
    arithmetic on the lattice rotations; the Cartesian basis is rendered once here, so no
    consumer repeats the frame multiplication or re-derives the tensor order.
    """
    order = seed.order
    label_basis = label_symmetric_basis(seed.labels)
    result = traverse_indexed_orbit(
        encode_key(seed),
        generators,
        order=order,
        canonical_columns=tuple(range(0, order * 4, 4))
        + tuple(column for axis in range(1, order) for column in range(axis * 4 + 1, axis * 4 + 4))
        + tuple(range(1, 4)),
    )
    kernel, dimension = invariant_kernel(label_basis, result.stabilizers, order=order)
    exact = result.seed_to_canonical.apply_scaled_columns(label_basis @ kernel)
    cartesian = cartesian_orbit_basis(exact, frame)
    cartesian_basis, transform = orthonormal_orbit_basis(cartesian)
    if cartesian_basis.shape[1] != dimension:
        raise RuntimeError(
            f"the rendered Cartesian basis has {cartesian_basis.shape[1]} columns but the "
            f"certified invariant dimension is {dimension}"
        )
    rows = select_observation_rows(cartesian_basis, dimension)
    observed = observation_matrix(cartesian_basis, rows)
    return GeneratedPrimitiveOrbit(
        representative=decode_key(result.canonical),
        exact_lattice_basis=exact,
        cartesian_basis=cartesian_basis,
        coefficient_transform=transform,
        observation_rows=rows,
        observation_condition=observation_condition(observed),
        states=result.states,
        actions=result.actions,
    )


__all__ = [
    "GeneratedPrimitiveOrbit",
    "PrimitiveGenerator",
    "build_primitive_interaction_space",
    "decode_key",
    "encode_key",
    "generated_orbit",
    "operation_composition_table",
    "primitive_generators",
    "source_key",
    "sympy_space_group_generators",
]
