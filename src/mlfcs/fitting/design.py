"""JAX force prediction and bounded design-matrix execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from mlfcs.fitting.basis import wick_axis_derivatives
from mlfcs.fitting.parameterization import OrderParameterization, image_parameter_basis


@dataclass(frozen=True, slots=True)
class DesignKernelGroup:
    """Same-shaped design tiles executed by one compiled device-side scan."""

    order: int
    kernel: object
    arguments: tuple[np.ndarray, ...]

    @property
    def tile_count(self) -> int:
        return len(self.arguments[0])


class ForceDesignOperator:
    """Batched force predictor and source data for streamed Gram construction."""

    def __init__(
        self,
        displacements,
        covariance,
        parameterizations,
        n_parameters,
        batch_size,
        reporter=None,
    ):
        self.displacements = np.asarray(displacements)
        self.covariance = jnp.asarray(covariance)
        self.parameterizations = parameterizations
        self.n_parameters = n_parameters
        self.batch_size = batch_size
        self.force_shape = self.displacements.shape
        self.reporter = reporter

        def forward(parameters, batch):
            return predict_force(parameters, batch, self.covariance, self.parameterizations)

        self._forward = jax.jit(forward)

    def matvec(self, parameters):
        started = perf_counter()
        parameters = jnp.asarray(np.asarray(parameters).reshape(-1))
        output = np.empty(self.force_shape, dtype=float)
        for begin in range(0, len(self.displacements), self.batch_size):
            end = min(begin + self.batch_size, len(self.displacements))
            output[begin:end] = np.asarray(
                self._forward(parameters, jnp.asarray(self.displacements[begin:end]))
            )
        if self.reporter is not None:
            self.reporter(
                f"- Force prediction: {len(self.displacements)} structures in "
                f"{perf_counter() - started:.2f} s"
            )
        return output.reshape(-1)


def prepare_design_kernel_groups(operator):
    """Compile bounded orbit/image/parameter tiles from the interaction space."""
    tiles_by_shape = {}
    for parameterization in operator.parameterizations:
        order = parameterization.order
        image_counts = np.sum(parameterization.image_mask, axis=1)
        dimension_counts = np.sum(parameterization.parameter_mask, axis=1)
        shapes = sorted(set(zip(image_counts.tolist(), dimension_counts.tolist())))
        for n_images, n_dimensions in shapes:
            selected = np.flatnonzero(
                (image_counts == n_images) & (dimension_counts == n_dimensions)
            )
            orbit_batch, image_batch, dimension_batch = physical_tile_shape(
                order,
                len(selected),
                n_images,
                n_dimensions,
                operator.batch_size,
                parameterization.coordinates.shape[2],
            )
            for begin in range(0, len(selected), orbit_batch):
                orbit_indices = selected[begin : begin + orbit_batch]
                for image_begin in range(0, n_images, image_batch):
                    image_end = min(image_begin + image_batch, n_images)
                    for dimension_begin in range(0, n_dimensions, dimension_batch):
                        dimension_end = min(dimension_begin + dimension_batch, n_dimensions)
                        tile = tile_parameterization(
                            parameterization,
                            orbit_indices,
                            slice(image_begin, image_end),
                            slice(dimension_begin, dimension_end),
                        )
                        global_columns = tile.parameter_indices.ravel().copy()
                        local_indices = np.arange(len(global_columns), dtype=np.int32).reshape(
                            tile.parameter_indices.shape
                        )
                        tile = OrderParameterization(
                            tile.order,
                            local_indices,
                            tile.parameter_mask,
                            tile.representative_from_pivots,
                            tile.rotations,
                            tile.component_permutations,
                            tile.coordinates,
                            tile.image_mask,
                        )
                        image_basis = image_parameter_basis(tile)
                        key = (
                            order,
                            *tile.parameter_indices.shape,
                            tile.rotations.shape[1],
                            len(global_columns),
                        )
                        tiles_by_shape.setdefault(key, []).append(
                            (
                                global_columns,
                                tile.parameter_indices,
                                tile.parameter_mask,
                                tile.representative_from_pivots,
                                tile.rotations,
                                tile.component_permutations,
                                tile.coordinates,
                                tile.image_mask,
                                image_basis,
                            )
                        )
    groups = []
    for key, tiles in tiles_by_shape.items():
        order = key[0]
        arguments = tuple(np.stack(values) for values in zip(*tiles, strict=True))
        groups.append(
            DesignKernelGroup(
                order=order,
                kernel=compile_design_tile_group(
                    order, key[-1], operator.n_parameters, operator.covariance
                ),
                arguments=arguments,
            )
        )
    return groups, operator.batch_size


def compile_design_tile_group(order, n_local, n_parameters, covariance):
    def design_tile_group(displacements, global_columns, *tile_arguments):
        initial = jnp.zeros(
            (len(displacements), displacements.shape[1] * 3, n_parameters),
            dtype=jnp.float64,
        )

        def add_tile(design, values):
            (
                columns,
                parameter_indices,
                parameter_mask,
                representative,
                rotations,
                permutations,
                coordinates,
                image_mask,
                image_basis,
            ) = values
            dynamic = OrderParameterization(
                order,
                parameter_indices,
                parameter_mask,
                representative,
                rotations,
                permutations,
                coordinates,
                image_mask,
            )
            contribution = force_design_batch(
                displacements, covariance, (dynamic,), (image_basis,), n_local
            )
            return design.at[..., columns].add(contribution), None

        design, _ = jax.lax.scan(add_tile, initial, (global_columns, *tile_arguments))
        return design

    return jax.jit(design_tile_group)


def physical_tile_shape(order, n_orbits, n_images, n_dimensions, structure_batch, translations):
    """Bound the full contraction volume, including periodic translations."""
    scalar_budget = 32_000_000
    fixed = max(structure_batch * translations * 3**order * order, 1)
    capacity = max(1, scalar_budget // fixed)
    dimension_batch = min(n_dimensions, capacity)
    image_batch = min(n_images, max(1, capacity // dimension_batch))
    orbit_batch = min(n_orbits, max(1, capacity // (dimension_batch * image_batch)))
    return orbit_batch, image_batch, dimension_batch


def tile_parameterization(parameterization, selected, images, dimensions):
    """Slice an exact-shape orbit group along images and parameter dimensions."""
    return OrderParameterization(
        parameterization.order,
        parameterization.parameter_indices[selected, dimensions],
        parameterization.parameter_mask[selected, dimensions],
        parameterization.representative_from_pivots[selected, :, dimensions],
        parameterization.rotations[selected, images],
        parameterization.component_permutations[selected, images],
        parameterization.coordinates[selected, images],
        parameterization.image_mask[selected, images],
    )


def force_design_batch(displacements, covariance, parameterizations, image_bases, n_parameters):
    """Construct exact force-design rows directly from the linear FC basis."""

    def one_structure(displacement):
        design = jnp.zeros((displacement.size, n_parameters), dtype=jnp.float64)
        for parameterization, image_basis in zip(parameterizations, image_bases, strict=True):
            order = parameterization.order
            coordinates = jnp.asarray(parameterization.coordinates)
            parameter_indices = jnp.asarray(parameterization.parameter_indices)
            coefficient_mask = (
                jnp.asarray(parameterization.image_mask)[:, :, None, None, None]
                * jnp.asarray(parameterization.parameter_mask)[:, None, None, None, :]
            )
            basis = jnp.asarray(image_basis)[:, :, None, :, :]
            lowers = wick_axis_derivatives(displacement, covariance, coordinates, order)
            for axis, lower in enumerate(lowers):
                contribution = -lower[..., None] * basis * coefficient_mask / factorial(order)
                force_coordinates = coordinates[..., axis, None]
                parameter_coordinates = parameter_indices[:, None, None, None, :]
                design = design.at[force_coordinates, parameter_coordinates].add(contribution)
        return design

    return jax.vmap(one_structure)(displacements)


def predict_force(parameters, displacements, covariance, parameterizations):
    """Evaluate model forces without materializing the design matrix."""

    def one_structure(displacement):
        force = jnp.zeros(displacement.size, dtype=jnp.float64)
        for parameterization in parameterizations:
            order = parameterization.order
            indices = jnp.asarray(parameterization.parameter_indices)
            local_parameters = parameters[indices] * jnp.asarray(parameterization.parameter_mask)
            representative = jnp.einsum(
                "ocd,od->oc",
                jnp.asarray(parameterization.representative_from_pivots),
                local_parameters,
            ).reshape((-1,) + (3,) * order)
            rotated = rotate_images(representative, jnp.asarray(parameterization.rotations), order)
            image_tensors = jnp.take_along_axis(
                rotated, jnp.asarray(parameterization.component_permutations), axis=2
            )
            coordinates = jnp.asarray(parameterization.coordinates)
            mask = jnp.asarray(parameterization.image_mask)
            lowers = wick_axis_derivatives(displacement, covariance, coordinates, order)
            for axis, lower in enumerate(lowers):
                contribution = (
                    -lower
                    * image_tensors[:, :, None, :]
                    * mask[:, :, None, None]
                    / factorial(order)
                )
                force = force.at[coordinates[..., axis].reshape(-1)].add(contribution.reshape(-1))
        return force.reshape(displacement.shape)

    return jax.vmap(one_structure)(displacements)


def rotate_tensor(tensor, rotation, order):
    result = tensor
    for axis in range(order):
        result = jnp.tensordot(rotation, result, axes=((1,), (axis,)))
        result = jnp.moveaxis(result, 0, axis)
    return result


def rotate_images(tensors, rotations, order):
    def one_tensor(tensor, operations):
        return jax.vmap(lambda operation: rotate_tensor(tensor, operation, order).reshape(-1))(
            operations
        )

    return jax.vmap(one_tensor)(tensors, rotations)
