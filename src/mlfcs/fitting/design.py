"""Compiled Cartesian force design for a primitive cluster space."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from mlfcs.core import LatticeSite
from mlfcs.force_constants.lattice import rotate_basis
from mlfcs.supercell import ClusterMap


@njit(cache=True, parallel=True, nogil=True)
def _accumulate(
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

    @property
    def max_dimension(self) -> int:
        return int(self.orbit_dimensions.max()) if len(self.orbit_dimensions) else 0


def _compile_order(
    mapping: ClusterMap, order: int, translations: tuple[tuple[int, int, int], ...]
) -> OrderDesign:
    space = mapping.space
    block = space.block(order)
    parameter_offsets = space.parameter_offsets
    orbit_indices = range(block.orbits.start, block.orbits.stop)
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
            atoms = np.empty((len(translations), order), dtype=np.int32)
            for translation_index, translation in enumerate(translations):
                for axis, site in enumerate(cluster.sites):
                    shifted = tuple(
                        value + offset
                        for value, offset in zip(site.translation, translation, strict=True)
                    )
                    atoms[translation_index, axis] = mapping.supercell.atom(
                        LatticeSite(site.site, shifted)
                    )
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
            np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int32)
        ),
        orbit_image_offsets=np.asarray(orbit_image_offsets, dtype=np.int64),
        orbit_parameter_starts=np.asarray(parameter_starts, dtype=np.int64),
        orbit_dimensions=np.asarray(dimensions, dtype=np.int32),
        image_atoms=(
            np.ascontiguousarray(image_atoms, dtype=np.int32)
            if image_atoms
            else np.empty((0, len(translations), order), dtype=np.int32)
        ),
        image_basis=np.concatenate(basis_values) if basis_values else np.zeros(0),
        image_basis_offsets=np.asarray(basis_offsets, dtype=np.int64),
    )


class ForceDesign:
    """Reusable compiled design for every order in one cluster map."""

    __slots__ = ("_scratch", "mapping", "orders", "rows")

    def __init__(self, mapping: ClusterMap):
        mapping.rank_info().require_full()
        self.mapping = mapping
        self.rows = 3 * len(mapping.supercell.numbers)
        translations = mapping.supercell.cell_translations
        self.orders = tuple(
            _compile_order(mapping, order, translations) for order in mapping.space.orders
        )
        self._scratch: tuple[np.ndarray, ...] | None = None

    @property
    def n_parameters(self) -> int:
        return self.mapping.space.n_parameters

    def matrix(self, displacement: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(displacement, dtype=np.float64).reshape(-1)
        if values.shape != (self.rows,):
            raise ValueError(f"displacement must contain {self.rows} Cartesian components")
        if self._scratch is None:
            threads = get_num_threads()
            self._scratch = tuple(
                np.zeros((threads, self.rows, order.max_dimension), dtype=np.float64)
                for order in self.orders
            )
        result = np.zeros((self.rows, self.n_parameters), dtype=np.float64)
        for order, scratch in zip(self.orders, self._scratch, strict=True):
            _accumulate(
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


__all__ = ["ForceDesign"]
