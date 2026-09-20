"""Compiled force-design construction for the joint Taylor IFC fit.

A design row is one force component of one snapshot and a design column is one
independent fitting parameter.  Every orbital image contributes

``-1/order! * (leave-one-axis monomial) * (rotated parameter tensor)``

to the force row of each cluster slot.  ``ForceDesignPlan`` compiles one order's
parameterization into the ragged arrays that a compiled parallel loop consumes:
one snapshot at a time becomes its ``(rows, n_parameters)`` design matrix, and
the Gram matrix is then accumulated by OpenBLAS from those rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import factorial

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from mlfcs.fitting.parameterization import OrderParameterization, image_parameter_basis

logger = logging.getLogger(__name__)


@njit(cache=True, parallel=True, nogil=True)
def _accumulate_design(
    displacement,
    out,
    orbit_pair_offsets,
    orbit_parameter_bases,
    orbit_dimensions,
    pair_atoms,
    pair_basis,
    pair_basis_offsets,
    components,
    factor,
    block,
):
    """Add one IFC order's contribution to one snapshot's design matrix.

    ``displacement`` holds the flat force components of a single snapshot and
    ``out`` is its ``(rows, n_parameters)`` design matrix.  Tasks are interaction
    orbits: orbits own disjoint parameter columns, and each thread accumulates an
    orbit into its own block before flushing it once, so no task writes a cell
    another task writes and hot threads stay off each other's cache lines.
    """
    n_orbits = orbit_pair_offsets.shape[0] - 1
    n_translations = pair_atoms.shape[1]
    order = pair_atoms.shape[2]
    n_components = components.shape[0]
    n_rows = displacement.size
    for orbit in prange(n_orbits):
        thread = get_thread_id()
        orbit_block = block[thread]
        dimensions = orbit_dimensions[orbit]
        parameter_base = orbit_parameter_bases[orbit]
        for row in range(n_rows):
            for dimension in range(dimensions):
                orbit_block[row, dimension] = 0.0
        for pair in range(orbit_pair_offsets[orbit], orbit_pair_offsets[orbit + 1]):
            basis_begin = pair_basis_offsets[pair]
            for translation in range(n_translations):
                for component in range(n_components):
                    for axis in range(order):
                        row = 3 * pair_atoms[pair, translation, axis] + components[component, axis]
                        monomial = 1.0
                        for other in range(order):
                            if other != axis:
                                monomial *= displacement[
                                    3 * pair_atoms[pair, translation, other]
                                    + components[component, other]
                                ]
                        value = -factor * monomial
                        basis_row = basis_begin + component * dimensions
                        for dimension in range(dimensions):
                            orbit_block[row, dimension] += value * pair_basis[basis_row + dimension]
        for row in range(n_rows):
            for dimension in range(dimensions):
                out[row, parameter_base + dimension] += orbit_block[row, dimension]


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """Ragged host arrays describing one IFC order's design contribution.

    One *pair* is one orbital image.  Basis tensors are concatenated per pair in
    component-major order and addressed through ``pair_basis_offsets``, so no
    orbit pays for the widest orbit's image or parameter count.  Orbit-level data
    (parameter block start, block width) is stored once per orbit, pair-level data
    (cluster atoms, basis block) once per image.
    """

    factor: float
    components: np.ndarray
    orbit_pair_offsets: np.ndarray
    orbit_parameter_bases: np.ndarray
    orbit_dimensions: np.ndarray
    pair_atoms: np.ndarray
    pair_basis: np.ndarray
    pair_basis_offsets: np.ndarray

    @property
    def orbit_count(self) -> int:
        return int(self.orbit_dimensions.size)

    @property
    def pair_count(self) -> int:
        return int(self.pair_atoms.shape[0])

    @property
    def max_dimensions(self) -> int:
        """Largest parameter block over this order's orbits."""
        return int(self.orbit_dimensions.max()) if self.orbit_dimensions.size else 0

    @property
    def n_bytes(self) -> int:
        return int(
            self.components.nbytes
            + self.orbit_pair_offsets.nbytes
            + self.orbit_parameter_bases.nbytes
            + self.orbit_dimensions.nbytes
            + self.pair_atoms.nbytes
            + self.pair_basis.nbytes
            + self.pair_basis_offsets.nbytes
        )


@dataclass(frozen=True, slots=True)
class ForceDesignPlan:
    """Reusable compiled design source for one joint fitting parameterization."""

    n_parameters: int
    orders: tuple[OrderPlan, ...]

    @classmethod
    def compile(cls, parameterizations) -> ForceDesignPlan:
        orders = tuple(_compile_order(item) for item in parameterizations)
        return cls(sum(int(order.orbit_dimensions.sum()) for order in orders), orders)

    @property
    def orbit_count(self) -> int:
        return sum(order.orbit_count for order in self.orders)

    @property
    def static_bytes(self) -> int:
        return sum(order.n_bytes for order in self.orders)

    def scratch(self, rows: int) -> tuple[np.ndarray, ...]:
        """Return one per-thread accumulation block per order for ``rows`` rows."""
        threads = get_num_threads()
        return tuple(
            np.zeros((threads, rows, order.max_dimensions), dtype=float) for order in self.orders
        )

    def accumulate(self, displacement: np.ndarray, out: np.ndarray, scratch) -> None:
        """Add every order's contribution to one snapshot's design matrix."""
        for order, block in zip(self.orders, scratch, strict=True):
            _accumulate_design(
                displacement,
                out,
                order.orbit_pair_offsets,
                order.orbit_parameter_bases,
                order.orbit_dimensions,
                order.pair_atoms,
                order.pair_basis,
                order.pair_basis_offsets,
                order.components,
                order.factor,
                block,
            )


def _orbit_blocks(parameterization: OrderParameterization):
    """Return per-orbit image counts, parameter counts and parameter block starts.

    Both invariants come from ``pack_order``: a mask selects a leading block of
    every row, and an orbit's parameters are one contiguous range of the joint
    parameter vector.  They are checked here because the compiled kernel relies on
    them for its loop bounds and unit-stride column updates.
    """
    image_counts = _prefix_counts(parameterization.image_mask, "image")
    dimensions = _prefix_counts(parameterization.parameter_mask, "parameter")
    indices = parameterization.parameter_indices
    bases = np.zeros(dimensions.size, dtype=np.int32)
    for orbit, width in enumerate(dimensions):
        if width:
            block = indices[orbit, :width]
            bases[orbit] = block[0]
            if not np.array_equal(block, bases[orbit] + np.arange(width, dtype=indices.dtype)):
                raise ValueError(
                    "fitting parameters of one orbit must form one contiguous block; "
                    f"orbit {orbit} maps to {block.tolist()}"
                )
    return image_counts, dimensions.astype(np.int32), bases


def _prefix_counts(mask, label):
    """Return per-row counts and require each mask row to be a true prefix."""
    counts = np.count_nonzero(mask, axis=1).astype(np.int64)
    expected = np.arange(mask.shape[1], dtype=np.int64)[None, :] < counts[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError(f"{label} masks must select a leading block of every row")
    return counts


def _compile_order(parameterization: OrderParameterization) -> OrderPlan:
    """Compile one order's parameterization into immutable ragged design arrays."""
    order = parameterization.order
    image_counts, dimensions, bases = _orbit_blocks(parameterization)
    n_pairs = int(image_counts.sum())

    orbit_pair_offsets = np.zeros(dimensions.size + 1, dtype=np.int64)
    np.cumsum(image_counts, out=orbit_pair_offsets[1:])
    pair_basis, pair_basis_offsets = image_parameter_basis(parameterization, image_counts)

    atoms = np.empty((n_pairs, *parameterization.coordinates.shape[2:]), dtype=np.int32)
    pair = 0
    for orbit, image_count in enumerate(image_counts):
        for image in range(int(image_count)):
            atoms[pair] = parameterization.coordinates[orbit, image]
            pair += 1

    return OrderPlan(
        factor=1.0 / factorial(order),
        components=np.ascontiguousarray(
            np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int32)
        ),
        orbit_pair_offsets=orbit_pair_offsets,
        orbit_parameter_bases=bases,
        orbit_dimensions=dimensions,
        pair_atoms=np.ascontiguousarray(atoms),
        pair_basis=np.ascontiguousarray(pair_basis, dtype=float),
        pair_basis_offsets=pair_basis_offsets,
    )


class ForceDesignOperator:
    """Snapshot displacements plus the design plan and constraint coordinates.

    Snapshots are processed one at a time: one design matrix fits the working set
    of one structure, and the compiled kernel takes its parallelism from the orbit
    dimension instead of from a snapshot batch.
    """

    def __init__(
        self,
        displacements,
        parameterizations=(),
        *,
        parameter_map=None,
        plan: ForceDesignPlan | None = None,
    ):
        values = np.ascontiguousarray(displacements, dtype=float)
        self.force_shape = values.shape
        self.displacements = values.reshape(len(values), -1)
        self.rows_per_snapshot = self.displacements.shape[1]
        self.plan = ForceDesignPlan.compile(parameterizations) if plan is None else plan
        if parameter_map is not None and parameter_map.shape[0] != self.plan.n_parameters:
            raise ValueError("the constraint map does not match the compiled design columns")
        self.parameter_map = parameter_map
        self.fit_n_parameters = (
            parameter_map.shape[1] if parameter_map is not None else self.plan.n_parameters
        )
        self._scratch = None
        if plan is None:
            pairs = sum(order.pair_count for order in self.plan.orders)
            logger.info(
                f"- Compiled design plan: {len(self.plan.orders)} orders, "
                f"{self.plan.orbit_count} orbits, {pairs} image pairs, "
                f"{self.plan.static_bytes / 1024**2:.1f} MiB arrays"
            )

    @property
    def n_parameters(self) -> int:
        return self.plan.n_parameters

    def with_displacements(self, displacements) -> ForceDesignOperator:
        """Reuse one compiled plan for another snapshot subset."""
        return ForceDesignOperator(
            displacements,
            parameter_map=self.parameter_map,
            plan=self.plan,
        )

    def design(self, index: int) -> np.ndarray:
        """Return the physical design matrix of one snapshot."""
        if self._scratch is None:
            self._scratch = self.plan.scratch(self.rows_per_snapshot)
        design = np.zeros((self.rows_per_snapshot, self.plan.n_parameters), dtype=float)
        self.plan.accumulate(self.displacements[index].reshape(-1), design, self._scratch)
        return design

    def reduce(self, design: np.ndarray) -> np.ndarray:
        """Map physical columns onto the independent constrained coordinates."""
        if self.parameter_map is None:
            return design
        return np.asarray(self.parameter_map.T @ design.T).T


__all__ = ["ForceDesignOperator", "ForceDesignPlan", "OrderPlan"]
