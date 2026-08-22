"""JAX force prediction and bounded design-matrix execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import factorial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from mlfcs.basis.wick import wick_axis_derivatives
from mlfcs.fitting.parameterization import OrderParameterization, image_parameter_basis
from mlfcs.fitting.runtime import transfer_guard


@dataclass(frozen=True, slots=True)
class DesignKernelGroup:
    """Same-shaped physical-design tiles and their persistent device buffers."""

    order: int
    kernel: object
    columns: np.ndarray
    device_columns: jax.Array
    arguments: tuple[np.ndarray, ...]
    device_arguments: tuple[jax.Array, ...]

    @property
    def tile_count(self) -> int:
        return len(self.columns)


class PreparedDesignProgram:
    """One immutable, reusable compiled representation of an interaction space.

    Parameterization arrays and covariance are host data while this object is
    prepared, then transferred to the selected JAX device once.  Training,
    validation, and post-fit diagnostics can subsequently change only the
    displacement batch and the parameter vector.
    """

    def __init__(self, covariance, parameterizations, batch_size, device):
        self.device = jax.devices()[0] if device is None else device
        self.covariance = jax.device_put(np.asarray(covariance, dtype=float), self.device)
        self.batch_size = batch_size
        # Retain the compact host parameterization for reproducible Gram-cache
        # fingerprints.  It is also reused by validation operators, so this is
        # metadata rather than an additional physical-design allocation.
        self.parameterizations = tuple(parameterizations)
        self.groups = _build_design_kernel_groups(self.parameterizations, batch_size, self.device)
        self.gram_feature_passes = 0
        self.prediction_feature_passes = 0

    @property
    def tile_count(self) -> int:
        return sum(group.tile_count for group in self.groups)

    @property
    def static_device_bytes(self) -> int:
        return int(
            self.covariance.nbytes
            + sum(
                group.columns.nbytes + sum(argument.nbytes for argument in group.arguments)
                for group in self.groups
            )
        )


@dataclass(frozen=True, slots=True)
class DeviceReductionPlan:
    """Bounded device-resident sparse map from physical to fit coordinates."""

    n_reduced: int
    chunks: tuple[tuple[jax.Array, jax.Array, jax.Array], ...]

    def apply(self, design):
        reduced = jax.device_put(
            np.zeros((design.shape[0], self.n_reduced), dtype=float), design.device
        )
        for rows, columns, values in self.chunks:
            reduced = _accumulate_sparse_reduction(reduced, design, rows, columns, values)
        return reduced


def prepare_device_reduction(parameter_map, force_rows, device):
    """Upload a sparse null-space map in bounded COO chunks.

    The intermediate has ``force_rows * chunk_entries`` elements, so a fixed
    internal budget bounds memory without exposing an expert-only API knob.
    """
    mapping = sparse.coo_matrix(parameter_map)
    if mapping.nnz == 0:
        return DeviceReductionPlan(mapping.shape[1], ())
    chunk_entries = min(
        mapping.nnz,
        max(1, 8_000_000 // max(int(force_rows), 1)),
    )
    chunks = []
    for begin in range(0, mapping.nnz, chunk_entries):
        end = min(begin + chunk_entries, mapping.nnz)
        size = chunk_entries
        rows = np.zeros(size, dtype=np.int32)
        columns = np.zeros(size, dtype=np.int32)
        values = np.zeros(size, dtype=float)
        count = end - begin
        rows[:count] = mapping.row[begin:end]
        columns[:count] = mapping.col[begin:end]
        values[:count] = mapping.data[begin:end]
        chunks.append(
            (
                jax.device_put(rows, device),
                jax.device_put(columns, device),
                jax.device_put(values, device),
            )
        )
    return DeviceReductionPlan(mapping.shape[1], tuple(chunks))


@jax.jit
def _accumulate_sparse_reduction(reduced, design, rows, columns, values):
    return reduced.at[:, columns].add(design[:, rows] * values)


@jax.jit
def accumulate_physical_design(design, contributions, columns):
    """Scatter one same-shaped tile group into a physical design matrix."""
    values = jnp.transpose(contributions, (1, 2, 0, 3)).reshape((design.shape[0], -1))
    return design.at[:, columns.reshape(-1)].add(values)


@jax.jit
def predict_group(contributions, columns, parameters):
    """Contract all local tiles of one IFC order into force rows at once."""
    values = jnp.transpose(contributions, (1, 2, 0, 3)).reshape((-1, columns.size))
    return values @ parameters[columns.reshape(-1)]


class ForceDesignOperator:
    """Batched force predictor and source data for streamed Gram construction."""

    def __init__(
        self,
        displacements,
        covariance,
        parameterizations,
        n_parameters,
        batch_size,
        parameter_map=None,
        reporter=None,
        program: PreparedDesignProgram | None = None,
        device=None,
        device_gram: bool | None = None,
    ):
        self.displacements = np.asarray(displacements)
        self.n_parameters = n_parameters
        self.parameter_map = parameter_map
        self.fit_n_parameters = (
            parameter_map.shape[1] if parameter_map is not None else n_parameters
        )
        self.batch_size = batch_size
        self.force_shape = self.displacements.shape
        self.reporter = reporter
        self._device_reductions = {}
        self.program = (
            PreparedDesignProgram(covariance, parameterizations, batch_size, device)
            if program is None
            else program
        )
        self.covariance = self.program.covariance
        self.device_gram = (
            self.program.device.platform == "gpu" if device_gram is None else bool(device_gram)
        )
        if program is None and self.reporter is not None:
            self.reporter(
                f"- Prepared JAX feature program: {len(self.program.groups)} signatures, "
                f"{self.program.tile_count} tiles, "
                f"{self.program.static_device_bytes / 1024**2:.1f} MiB static buffers"
            )

    def with_displacements(self, displacements, *, reporter=None):
        """Reuse static kernels and device buffers for another snapshot subset."""
        return ForceDesignOperator(
            displacements,
            None,
            (),
            self.n_parameters,
            self.batch_size,
            reporter=self.reporter if reporter is None else reporter,
            program=self.program,
            device_gram=self.device_gram,
        )

    def device_reduction(self, force_rows):
        """Return a reusable bounded device map for one batch row shape."""
        if self.parameter_map is None:
            return None
        plan = self._device_reductions.get(force_rows)
        if plan is None:
            plan = prepare_device_reduction(self.parameter_map, force_rows, self.program.device)
            self._device_reductions[force_rows] = plan
        return plan

    def matvec(self, parameters):
        """Predict forces through bounded physical-design groups.

        Passing the parameterization arrays as dynamic group arguments avoids
        capturing a multi-gigabyte interaction space as constants in one JAX
        lowering, which is particularly important for validation after a
        high-order fit.
        """
        started = perf_counter()
        self.program.prediction_feature_passes += 1
        with transfer_guard():
            parameters = jax.device_put(
                np.asarray(parameters, dtype=float).reshape(-1), self.program.device
            )
            output = np.empty(self.force_shape, dtype=float)
            groups = self.program.groups
            effective_batch_size = self.batch_size
            rows_per_structure = int(np.prod(self.force_shape[1:]))
            for begin in range(0, len(self.displacements), effective_batch_size):
                end = min(begin + effective_batch_size, len(self.displacements))
                force_rows = (end - begin) * rows_per_structure
                displacement_batch = jax.device_put(
                    self.displacements[begin:end], self.program.device
                )
                predicted = jax.device_put(np.zeros(force_rows, dtype=float), self.program.device)
                for group in groups:
                    contributions = group.kernel(
                        displacement_batch,
                        self.covariance,
                        *group.device_arguments,
                    )
                    predicted += predict_group(contributions, group.device_columns, parameters)
                output[begin:end] = np.asarray(jax.device_get(predicted)).reshape(
                    output[begin:end].shape
                )
        if self.reporter is not None:
            self.reporter(
                f"- Bounded force prediction: {len(self.displacements)} structures in "
                f"{perf_counter() - started:.2f} s"
            )
        return output.reshape(-1)

    def matvec_by_order(self, parameters):
        """Predict every IFC-order contribution in one shared feature pass."""
        started = perf_counter()
        self.program.prediction_feature_passes += 1
        with transfer_guard():
            parameters = jax.device_put(
                np.asarray(parameters, dtype=float).reshape(-1), self.program.device
            )
            orders = tuple(sorted({group.order for group in self.program.groups}))
            output = {order: np.empty(self.force_shape, dtype=float) for order in orders}
            rows_per_structure = int(np.prod(self.force_shape[1:]))
            for begin in range(0, len(self.displacements), self.batch_size):
                end = min(begin + self.batch_size, len(self.displacements))
                force_rows = (end - begin) * rows_per_structure
                displacement_batch = jax.device_put(
                    self.displacements[begin:end], self.program.device
                )
                predicted = {
                    order: jax.device_put(np.zeros(force_rows, dtype=float), self.program.device)
                    for order in orders
                }
                for group in self.program.groups:
                    contributions = group.kernel(
                        displacement_batch, self.covariance, *group.device_arguments
                    )
                    predicted[group.order] += predict_group(
                        contributions, group.device_columns, parameters
                    )
                for order in orders:
                    output[order][begin:end] = np.asarray(jax.device_get(predicted[order])).reshape(
                        output[order][begin:end].shape
                    )
        if self.reporter is not None:
            self.reporter(
                f"- Bounded order-resolved force prediction: {len(self.displacements)} "
                f"structures in {perf_counter() - started:.2f} s"
            )
        return {order: values.reshape(-1) for order, values in output.items()}


def prepare_design_kernel_groups(operator):
    """Return an operator's already-prepared bounded design groups.

    Kept as a private compatibility helper for internal tests and Gram code;
    it performs no packing, upload, or compilation work.
    """
    return operator.program.groups, operator.program.batch_size


def _build_design_kernel_groups(parameterizations, batch_size, device):
    """Pack and upload bounded orbit/image/parameter tiles exactly once."""
    tiles_by_shape = {}
    for parameterization in parameterizations:
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
                batch_size,
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
        values = tuple(np.stack(items) for items in zip(*tiles, strict=True))
        columns, arguments = values[0], values[1:]
        groups.append(
            DesignKernelGroup(
                order=order,
                kernel=compile_design_tile_group(order, key[-1]),
                columns=columns,
                device_columns=jax.device_put(columns, device),
                arguments=arguments,
                device_arguments=tuple(jax.device_put(value, device) for value in arguments),
            )
        )
    return tuple(groups)


@cache
def compile_design_tile_group(order, n_local):
    """Compile one shape-polymorphic kernel returning only local tile columns.

    Large covariance and interaction arrays are dynamic arguments.  In
    particular, the kernel neither captures them as XLA constants nor emits a
    full ``n_parameters``-wide matrix for every physical tile.
    """

    def design_tile_group(displacements, covariance, *tile_arguments):
        def one_tile(values):
            (
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
            return force_design_batch(
                displacements, covariance, (dynamic,), (image_basis,), n_local
            )

        return jax.lax.map(one_tile, tile_arguments)

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
            atom_coordinates = jnp.asarray(parameterization.coordinates)
            components = jnp.asarray(tuple(np.ndindex((3,) * order)), dtype=jnp.int32)
            coordinates = atom_coordinates[..., None, :] * 3 + components
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
            components = jnp.asarray(tuple(np.ndindex((3,) * order)), dtype=jnp.int32)
            coordinates = coordinates[..., None, :] * 3 + components
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
