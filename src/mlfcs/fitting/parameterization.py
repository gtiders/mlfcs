"""Symmetry-reduced fitting parameterization and IFC reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs.model import SparseOrderForceConstants


@dataclass(frozen=True, slots=True)
class OrderParameterization:
    """Array representation of one order's orbit-to-parameter mapping."""

    order: int
    parameter_indices: np.ndarray
    parameter_mask: np.ndarray
    representative_from_pivots: np.ndarray
    rotations: np.ndarray
    component_permutations: np.ndarray
    coordinates: np.ndarray
    image_mask: np.ndarray

    @property
    def n_parameters(self) -> int:
        return int(np.count_nonzero(self.parameter_mask))


def pack_order(calculation, offset):
    """Pack an orbit space into fixed-shape arrays used by JAX kernels."""
    orbit_space = calculation.orbit_space
    order = orbit_space.order
    orbits = orbit_space.orbits
    n_orbits = len(orbits)
    max_images = max(len(orbit.images) for orbit in orbits)
    max_dimension = max(orbit.dimension for orbit in orbits)
    components = np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int32)
    parameter_indices = np.zeros((n_orbits, max_dimension), dtype=np.int32)
    parameter_mask = np.zeros_like(parameter_indices, dtype=bool)
    representatives = np.zeros((n_orbits, 3**order, max_dimension))
    rotations = np.zeros((n_orbits, max_images, 3, 3))
    permutations = np.zeros((n_orbits, max_images, 3**order), dtype=np.int32)
    coordinates = np.zeros(
        (
            n_orbits,
            max_images,
            len(calculation.index.translations) // calculation.index.n_primitive,
            3**order,
            order,
        ),
        dtype=np.int32,
    )
    image_mask = np.zeros((n_orbits, max_images), dtype=bool)
    translations = np.unique(calculation.index.translations, axis=0)
    base = np.arange(3**order).reshape((3,) * order)
    for orbit_index, orbit in enumerate(orbits):
        dimension = orbit.dimension
        images = len(orbit.images)
        parameter_indices[orbit_index, :dimension] = np.arange(offset, offset + dimension)
        parameter_mask[orbit_index, :dimension] = True
        representatives[orbit_index, :, :dimension] = np.linalg.solve(
            orbit.basis[orbit.pivots].T, orbit.basis.T
        ).T
        for image_index, image in enumerate(orbit.images):
            rotations[orbit_index, image_index] = image.action.rotation
            permutations[orbit_index, image_index] = base.transpose(
                image.action.permutation
            ).ravel()
            for translation_index, translation in enumerate(translations):
                atoms = [
                    calculation.index.translate_atom(atom, translation) for atom in image.cluster
                ]
                coordinates[orbit_index, image_index, translation_index] = (
                    np.asarray(atoms)[None, :] * 3 + components
                )
        image_mask[orbit_index, :images] = True
        offset += dimension
    return (
        OrderParameterization(
            order,
            parameter_indices,
            parameter_mask,
            representatives,
            rotations,
            permutations,
            coordinates,
            image_mask,
        ),
        offset,
    )


def image_parameter_basis(parameterization):
    """Map each symmetry-image tensor component to independent parameters."""
    order = parameterization.order
    representative = parameterization.representative_from_pivots
    result = np.zeros(
        (*parameterization.rotations.shape[:2], 3**order, representative.shape[-1]),
        dtype=float,
    )
    for orbit in range(len(representative)):
        for image in range(parameterization.rotations.shape[1]):
            rotation = parameterization.rotations[orbit, image]
            for dimension in range(representative.shape[-1]):
                value = representative[orbit, :, dimension].reshape((3,) * order)
                for axis in range(order):
                    value = np.tensordot(rotation, value, axes=((1,), (axis,)))
                    value = np.moveaxis(value, 0, axis)
                result[orbit, image, :, dimension] = value.reshape(-1)
    component_indices = parameterization.component_permutations[..., None]
    return np.take_along_axis(result, component_indices, axis=2)


def expand_sparse(parameters, calculations, n_primitive, n_supercell):
    """Expand irreducible parameters into symmetry-related sparse IFC tensors."""
    result = {}
    offset = 0
    for calculation in calculations:
        clusters = []
        tensors = []
        for orbit in calculation.orbit_space.orbits:
            values = parameters[offset : offset + orbit.dimension]
            offset += orbit.dimension
            representative = orbit.basis @ np.linalg.solve(orbit.basis[orbit.pivots], values)
            for image in orbit.images:
                clusters.append(image.cluster)
                tensor = representative.reshape((3,) * calculation.config.order)
                for axis in range(calculation.config.order):
                    tensor = np.tensordot(image.action.rotation, tensor, axes=((1,), (axis,)))
                    tensor = np.moveaxis(tensor, 0, axis)
                tensors.append(np.transpose(tensor, image.action.permutation))
        result[calculation.config.order] = SparseOrderForceConstants(
            calculation.config.order,
            n_primitive,
            n_supercell,
            np.asarray(clusters, dtype=np.int32),
            np.asarray(tensors),
        )
    return result
