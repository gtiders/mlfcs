"""Compiled Cartesian force designs and caller-owned workspace."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from mlfcs._arrays import integer_array, readonly, require_allocation
from mlfcs.core.tensors import rotate_basis
from mlfcs.mapping import ClusterMap
from mlfcs.mapping.cluster_map import prepare_cluster_map
from mlfcs.mapping.geometry import mapped_labels


@dataclass(frozen=True, slots=True)
class OrderDesign:
    """Ragged compiled arrays for one force-constant order."""

    factor: float
    components: np.ndarray
    orbit_image_offsets: np.ndarray
    orbit_parameter_starts: np.ndarray
    orbit_dimensions: np.ndarray
    image_atoms: np.ndarray
    image_basis: np.ndarray
    image_basis_offsets: np.ndarray

    def __post_init__(self):
        for name in (
            "components",
            "orbit_image_offsets",
            "orbit_parameter_starts",
            "orbit_dimensions",
            "image_atoms",
            "image_basis_offsets",
        ):
            array = getattr(self, name)
            require_allocation(name, array.shape)
            object.__setattr__(self, name, integer_array(array, name=name))
        require_allocation("image basis", self.image_basis.shape)
        object.__setattr__(self, "image_basis", readonly(self.image_basis, np.float64))

    @property
    def max_dimension(self) -> int:
        return int(self.orbit_dimensions.max()) if len(self.orbit_dimensions) else 0


def _compile_order(
    cluster_map: ClusterMap, order: int, translations: tuple[tuple[int, int, int], ...], prepared
) -> OrderDesign:
    space = cluster_map.cluster_space
    block = space.block(order)
    parameter_offsets = space.parameter_offsets
    orbit_indices = range(block.orbits.start, block.orbits.stop)
    image_count = sum(len(space.orbits[i].clusters) for i in orbit_indices)
    basis_count = sum(
        len(space.orbits[i].clusters) * 3**order * space.orbits[i].dimension for i in orbit_indices
    )
    require_allocation("image atoms", (image_count, len(translations), order))
    require_allocation("image basis", (basis_count,))
    require_allocation("image basis offsets", (image_count + 1,))
    require_allocation("orbit offsets", (len(orbit_indices) + 1,))
    require_allocation("tensor components", (3**order, order))
    translations = integer_array(translations, name="mapping translations")
    orbit_image_offsets = [0]
    parameter_starts = []
    dimensions = []
    image_atoms = []
    basis_values = []
    basis_offsets = [0]
    for orbit_index in orbit_indices:
        orbit = space.orbits[orbit_index]
        dimension = orbit.dimension
        parameter_starts.append(int(parameter_offsets[orbit_index]))
        dimensions.append(dimension)
        for image, cluster in enumerate(orbit.clusters):
            labels = np.asarray(cluster.labels, dtype=np.int64)
            atoms = mapped_labels(labels, translations, prepared.periodic).T.copy()
            image_atoms.append(atoms)
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            flattened = np.ascontiguousarray(basis).reshape(-1)
            basis_values.append(flattened)
            basis_offsets.append(basis_offsets[-1] + len(flattened))
        orbit_image_offsets.append(len(image_atoms))
    return OrderDesign(
        factor=1.0 / factorial(order),
        components=np.ascontiguousarray(
            np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int64)
        ),
        orbit_image_offsets=np.asarray(orbit_image_offsets, dtype=np.int64),
        orbit_parameter_starts=np.asarray(parameter_starts, dtype=np.int64),
        orbit_dimensions=np.asarray(dimensions, dtype=np.int64),
        image_atoms=(
            np.ascontiguousarray(image_atoms, dtype=np.int64)
            if image_atoms
            else np.empty((0, len(translations), order), dtype=np.int64)
        ),
        image_basis=np.concatenate(basis_values) if basis_values else np.zeros(0),
        image_basis_offsets=np.asarray(basis_offsets, dtype=np.int64),
    )


class ForceDesign:
    """Reusable compiled design for every order in one cluster map."""

    __slots__ = ("cluster_map", "orders", "prepared", "rows")

    def __init__(self, cluster_map: ClusterMap):
        cluster_map.rank_info().require_full()
        self.cluster_map = cluster_map
        self.prepared = prepare_cluster_map(cluster_map)
        self.rows = 3 * len(cluster_map.atomic_numbers)
        require_allocation("force design", (self.rows, self.n_parameters))
        translations = cluster_map.translation_representatives
        self.orders = tuple(
            _compile_order(cluster_map, order, translations, self.prepared)
            for order in cluster_map.cluster_space.orders
        )

    @property
    def n_parameters(self) -> int:
        return self.cluster_map.cluster_space.n_parameters

    def allocate_workspace(self):
        return allocate_design_workspace(
            self.rows, tuple(order.max_dimension for order in self.orders)
        )

    def __reduce__(self):
        return type(self), (self.cluster_map,)

    def matrix(self, displacement: np.ndarray, *, workspace=None) -> np.ndarray:
        values = np.ascontiguousarray(displacement, dtype=np.float64).reshape(-1)
        if values.shape != (self.rows,):
            raise ValueError(f"displacement must contain {self.rows} Cartesian components")
        if not np.all(np.isfinite(values)):
            raise ValueError("displacement must be finite")
        if workspace is None:
            workspace = self.allocate_workspace()
        workspace.validate(self.rows, tuple(order.max_dimension for order in self.orders))
        result = np.zeros((self.rows, self.n_parameters), dtype=np.float64)
        for order, scratch in zip(self.orders, workspace.scratch, strict=True):
            accumulate_design(
                values,
                result,
                order.orbit_image_offsets,
                order.orbit_parameter_starts,
                order.orbit_dimensions,
                order.image_atoms,
                order.image_basis,
                order.image_basis_offsets,
                order.components,
                order.factor,
                scratch,
            )
        return result


@njit(cache=True, parallel=True, nogil=True)
def accumulate_design(
    displacement,
    out,
    orbit_image_offsets,
    orbit_parameter_starts,
    orbit_dimensions,
    image_atoms,
    image_basis,
    image_basis_offsets,
    components,
    factor,
    scratch,
):
    """Accumulate one tensor order into a snapshot design matrix."""
    orbit_count = orbit_dimensions.size
    translation_count = image_atoms.shape[1]
    order = image_atoms.shape[2]
    component_count = components.shape[0]
    rows = displacement.size
    for orbit in prange(orbit_count):
        local = scratch[get_thread_id()]
        dimension_count = orbit_dimensions[orbit]
        parameter_start = orbit_parameter_starts[orbit]
        for row in range(rows):
            for dimension in range(dimension_count):
                local[row, dimension] = 0.0
        for image in range(orbit_image_offsets[orbit], orbit_image_offsets[orbit + 1]):
            basis_start = image_basis_offsets[image]
            for translation in range(translation_count):
                for component in range(component_count):
                    for axis in range(order):
                        row = (
                            3 * image_atoms[image, translation, axis] + components[component, axis]
                        )
                        monomial = 1.0
                        for other in range(order):
                            if other != axis:
                                monomial *= displacement[
                                    3 * image_atoms[image, translation, other]
                                    + components[component, other]
                                ]
                        value = -factor * monomial
                        basis_row = basis_start + component * dimension_count
                        for dimension in range(dimension_count):
                            local[row, dimension] += value * image_basis[basis_row + dimension]
        for row in range(rows):
            for dimension in range(dimension_count):
                out[row, parameter_start + dimension] += local[row, dimension]


@dataclass(slots=True)
class DesignWorkspace:
    rows: int
    dimensions: tuple[int, ...]
    threads: int
    scratch: tuple[np.ndarray, ...]

    def validate(self, rows, dimensions):
        if (rows, dimensions) != (self.rows, self.dimensions):
            raise ValueError("design workspace belongs to different dimensions")
        if len(self.scratch) != len(dimensions):
            raise ValueError("design workspace has inconsistent scratch buffers")
        for array, dimension in zip(self.scratch, dimensions, strict=True):
            if (
                array.shape != (self.threads, rows, dimension)
                or array.dtype != np.float64
                or not array.flags.c_contiguous
                or not array.flags.writeable
            ):
                raise ValueError("design workspace scratch has invalid shape, dtype or storage")
        if get_num_threads() > self.threads:
            raise ValueError("design workspace has fewer slots than active Numba threads")


def allocate_design_workspace(rows, dimensions):
    threads = get_num_threads()
    for dimension in dimensions:
        require_allocation("design scratch", (threads, rows, dimension))
    return DesignWorkspace(
        rows,
        dimensions,
        threads,
        tuple(np.zeros((threads, rows, dimension), dtype=np.float64) for dimension in dimensions),
    )


__all__ = ["ForceDesign"]
