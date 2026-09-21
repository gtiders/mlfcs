"""Symmetry-reduced fitting parameterization and IFC reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OrderParameterization:
    """Array representation of one order's orbit-to-parameter mapping.

    ``cartesian_basis`` holds each orbit's orthonormal Cartesian basis rows, padded to the
    widest orbit; a fitted parameter is the coefficient of one of those basis columns,
    which is the only coordinate meaning a parameter has.
    """

    order: int
    parameter_indices: np.ndarray
    parameter_mask: np.ndarray
    cartesian_basis: np.ndarray
    rotations: np.ndarray
    component_permutations: np.ndarray
    coordinates: np.ndarray
    image_mask: np.ndarray

    @property
    def n_parameters(self) -> int:
        return int(np.count_nonzero(self.parameter_mask))


def pack_order(calculation, offset):
    """Pack an orbit space into fixed-shape arrays used by compiled design kernels."""
    orbit_space = calculation.realized_orbit_space
    order = orbit_space.order
    orbits = orbit_space.orbits
    n_orbits = len(orbits)
    max_images = max(len(orbit.images) for orbit in orbits)
    max_dimension = max(orbit.dimension for orbit in orbits)
    parameter_indices = np.zeros((n_orbits, max_dimension), dtype=np.int32)
    parameter_mask = np.zeros_like(parameter_indices, dtype=bool)
    orbit_basis = np.zeros((n_orbits, 3**order, max_dimension))
    rotations = np.zeros((n_orbits, max_images, 3, 3))
    permutations = np.zeros((n_orbits, max_images, 3**order), dtype=np.int32)
    coordinates = np.zeros(
        (
            n_orbits,
            max_images,
            len(calculation.index.translations) // calculation.index.n_primitive,
            order,
        ),
        dtype=np.int32,
    )
    image_mask = np.zeros((n_orbits, max_images), dtype=bool)
    translations = calculation.index.cell_representatives
    base = np.arange(3**order).reshape((3,) * order)
    for orbit_index, orbit in enumerate(orbits):
        dimension = orbit.dimension
        images = len(orbit.images)
        parameter_indices[orbit_index, :dimension] = np.arange(offset, offset + dimension)
        parameter_mask[orbit_index, :dimension] = True
        orbit_basis[orbit_index, :, :dimension] = orbit.cartesian_basis
        for image_index, image in enumerate(orbit.images):
            rotations[orbit_index, image_index] = image.action.rotation
            permutations[orbit_index, image_index] = base.transpose(
                image.action.permutation
            ).ravel()
            coordinates[orbit_index, image_index] = calculation.index.translate_atoms(
                np.asarray(image.cluster, dtype=np.int32), translations
            )
        image_mask[orbit_index, :images] = True
        offset += dimension
    return (
        OrderParameterization(
            order,
            parameter_indices,
            parameter_mask,
            orbit_basis,
            rotations,
            permutations,
            coordinates,
            image_mask,
        ),
        offset,
    )


def image_parameter_basis(parameterization, image_counts=None):
    """Map every symmetry image's tensor components to independent parameters.

    Returns the concatenated per-image basis blocks and the pair offsets, so a
    caller never materializes a padded ``(orbits, images, 3**order, dimensions)``
    array just to slice it per image again.  Block ``pair`` occupies
    ``blocks[offsets[pair] : offsets[pair + 1]]`` in component-major order with
    ``dimensions`` columns.
    """
    order = parameterization.order
    basis = parameterization.cartesian_basis
    rotations = parameterization.rotations
    permutations = parameterization.component_permutations
    if image_counts is None:
        image_counts = np.count_nonzero(parameterization.image_mask, axis=1)
    dimension_counts = np.count_nonzero(parameterization.parameter_mask, axis=1)
    n_pairs = int(np.sum(image_counts))
    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    blocks = []
    pair = 0
    for orbit, image_count in enumerate(image_counts):
        dimensions = int(dimension_counts[orbit])
        for image in range(int(image_count)):
            rotation = rotations[orbit, image]
            block = np.zeros((3**order, dimensions))
            for dimension in range(dimensions):
                value = basis[orbit, :, dimension].reshape((3,) * order)
                for axis in range(order):
                    value = np.tensordot(rotation, value, axes=((1,), (axis,)))
                    value = np.moveaxis(value, 0, axis)
                block[:, dimension] = value.reshape(-1)
            block = np.take_along_axis(block, permutations[orbit, image][:, None], axis=0)
            blocks.append(np.ascontiguousarray(block).reshape(-1))
            offsets[pair + 1] = offsets[pair] + 3**order * dimensions
            pair += 1
    concatenated = np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)
    return concatenated, offsets
