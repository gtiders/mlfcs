"""Streamed Gram accumulation, recovery cache, and fitting metrics."""

from __future__ import annotations

from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy import sparse
from scipy.linalg.blas import dsyrk

from mlfcs.fitting.design import accumulate_physical_design
from mlfcs.fitting.design import prepare_design_kernel_groups as _prepare_physical_design_builders
from mlfcs.fitting.linear_solvers import solve_gram_system


@partial(jax.jit, donate_argnums=(0, 1))
def _update_device_statistics(current_gram, current_rhs, design, force):
    return current_gram + design.T @ design, current_rhs + design.T @ force


def _orbit_parameter_groups(calculations):
    """Return contiguous slices, one per symmetry-irreducible cluster orbit."""
    groups = []
    offset = 0
    for calculation in calculations:
        for orbit in calculation.orbit_space.orbits:
            groups.append(slice(offset, offset + orbit.dimension))
            offset += orbit.dimension
    return tuple(groups)


class _StreamingGramSystem:
    """Normal-equation sufficient statistics accumulated without storing A."""

    def __init__(self, gram, rhs, target_norm, reporter=None, cache_directory=None):
        self.gram = gram
        self.rhs = rhs
        self.target_norm = target_norm
        self.reporter = reporter
        self.cache_directory = cache_directory

    @classmethod
    def from_operator(cls, operator, target, *, cache_directory: str | Path | None = None):
        started = perf_counter()
        gram_nbytes = operator.fit_n_parameters**2 * np.dtype(np.float64).itemsize
        use_recovery_cache = gram_nbytes >= 64 * 1024**2
        if cache_directory is not None:
            cache_directory = (
                Path(cache_directory).expanduser().resolve()
                / f"gram-{_gram_recovery_key(operator, target)}"
            )
        elif use_recovery_cache:
            cache_key = _gram_recovery_key(operator, target)
            cache_directory = Path.cwd() / ".mlfcs-cache" / f"gram-{cache_key}"
        if cache_directory is not None and (cache_directory / "complete").exists():
            gram = np.load(cache_directory / "gram.npy")
            rhs = np.load(cache_directory / "rhs.npy")
            target_norm = float(np.load(cache_directory / "target_norm.npy"))
            if operator.reporter is not None:
                operator.reporter(
                    f"Recovered completed streamed Gram system from cache "
                    f"({perf_counter() - started:.2f} s)"
                )
            return cls(gram, rhs, target_norm, operator.reporter, cache_directory)
        use_device_gram = operator.device_gram
        if use_device_gram:
            gram = jax.device_put(
                np.zeros((operator.fit_n_parameters, operator.fit_n_parameters), dtype=float),
                operator.program.device,
            )
            rhs = jax.device_put(
                np.zeros(operator.fit_n_parameters, dtype=float), operator.program.device
            )
        else:
            gram = np.zeros(
                (operator.fit_n_parameters, operator.fit_n_parameters), dtype=float, order="F"
            )
            rhs = np.zeros(operator.fit_n_parameters, dtype=float)
        target_shaped = np.asarray(target).reshape(operator.force_shape)
        builders, effective_batch_size = _prepare_physical_design_builders(operator)
        rows_per_structure = int(np.prod(operator.force_shape[1:]))
        operator.program.gram_feature_passes += 1
        if operator.reporter is not None:
            tile_counts = [group.tile_count for group in builders]
            operator.reporter(
                f"Accumulating streamed Gram system: {operator.fit_n_parameters} x "
                f"{operator.fit_n_parameters} ({gram.nbytes / 1024**2:.1f} MiB), "
                f"effective_batch_size={effective_batch_size}, "
                f"backend={'JAX device' if use_device_gram else 'SciPy/OpenBLAS CPU'}"
            )
            operator.reporter(
                f"- Physical design kernel groups: {len(builders)}, "
                f"{sum(tile_counts)} bounded tiles"
            )
        kernel_seconds = 0.0
        transfer_seconds = 0.0
        scatter_seconds = 0.0
        reduction_seconds = 0.0
        gram_seconds = 0.0

        for begin in range(0, len(operator.displacements), effective_batch_size):
            end = min(begin + effective_batch_size, len(operator.displacements))
            force_rows = (end - begin) * rows_per_structure
            if use_device_gram:
                design = jax.device_put(
                    np.zeros((force_rows, operator.n_parameters), dtype=float),
                    operator.program.device,
                )
            else:
                design = np.zeros((force_rows, operator.n_parameters), dtype=float)
            displacement_batch = jax.device_put(
                operator.displacements[begin:end], operator.program.device
            )
            for group in builders:
                order_started = perf_counter()
                columns = group.columns
                contributions = group.kernel(
                    displacement_batch,
                    operator.basis_state,
                    *group.device_arguments,
                )
                # A physical tile contains only its local parameter columns.
                # Scatter after device execution so XLA never lowers one full
                # parameter-wide output for each kernel group.
                if use_device_gram:
                    design = accumulate_physical_design(design, contributions, group.device_columns)
                else:
                    contributions.block_until_ready()
                    kernel_seconds += perf_counter() - order_started
                    transfer_started = perf_counter()
                    host_contributions = np.asarray(contributions)
                    transfer_seconds += perf_counter() - transfer_started
                    scatter_started = perf_counter()
                    for contribution, tile_columns in zip(host_contributions, columns, strict=True):
                        contribution = contribution.reshape(force_rows, -1)
                        design[:, tile_columns] += contribution
                    scatter_seconds += perf_counter() - scatter_started
                if operator.reporter is not None and begin == 0:
                    contributions.block_until_ready()
                    operator.reporter(
                        f"- Compiled FC{group.order} physical design kernel in "
                        f"{perf_counter() - order_started:.2f} s"
                    )
            if operator.parameter_map is not None:
                reduction_started = perf_counter()
                if use_device_gram:
                    # The sparse map is uploaded once in bounded COO chunks.
                    # This keeps physical design, reduction, and Gram updates
                    # on the device instead of round-tripping through SciPy.
                    design = operator.device_reduction(force_rows).apply(design)
                else:
                    design = np.asarray(operator.parameter_map.T @ design.T).T
                reduction_seconds += perf_counter() - reduction_started
            force = target_shaped[begin:end].reshape(-1)
            gram_started = perf_counter()
            if use_device_gram:
                gram, rhs = _update_device_statistics(
                    gram,
                    rhs,
                    design,
                    jax.device_put(force, operator.program.device),
                )
            else:
                gram = dsyrk(
                    1.0,
                    a=design,
                    c=gram,
                    beta=1.0,
                    trans=1,
                    lower=0,
                    overwrite_c=1,
                )
                rhs += design.T @ force
            gram_seconds += perf_counter() - gram_started
            if operator.reporter is not None and (
                begin == 0 or end == len(operator.displacements) or end % 20 == 0
            ):
                if use_device_gram:
                    gram.block_until_ready()
                operator.reporter(
                    f"- Gram structures: {end}/{len(operator.displacements)}, "
                    f"elapsed={perf_counter() - started:.2f} s"
                )
        if use_device_gram:
            gram = np.asarray(jax.device_get(gram))
            rhs = np.asarray(jax.device_get(rhs))
        else:
            upper = np.triu(np.asarray(gram))
            gram = upper + np.triu(upper, 1).T
        if operator.reporter is not None:
            operator.reporter(f"- Streamed Gram system ready in {perf_counter() - started:.2f} s")
            if use_device_gram:
                operator.reporter(
                    "- Gram phase timing: device kernel, scatter, reduction, and BLAS "
                    "execute asynchronously as one device pipeline"
                )
            else:
                operator.reporter(
                    "- Gram phase timing: "
                    f"kernel={kernel_seconds:.2f} s, transfer={transfer_seconds:.2f} s, "
                    f"scatter={scatter_seconds:.2f} s, "
                    f"constraint reduction={reduction_seconds:.2f} s, "
                    f"BLAS={gram_seconds:.2f} s"
                )
        target_norm = float(np.vdot(target, target))
        if cache_directory is not None:
            cache_directory.mkdir(parents=True, exist_ok=True)
            np.save(cache_directory / "gram.npy", gram)
            np.save(cache_directory / "rhs.npy", rhs)
            np.save(cache_directory / "target_norm.npy", np.asarray(target_norm))
            (cache_directory / "complete").write_text("mlfcs streamed Gram recovery cache\n")
        return cls(gram, rhs, target_norm, operator.reporter, cache_directory)

    def exact_column_scale(self):
        norm = np.sqrt(np.maximum(np.diag(self.gram), 0.0))
        threshold = max(float(np.max(norm)) * 1e-12, np.finfo(float).tiny)
        result = np.zeros_like(norm)
        active = norm > threshold
        result[active] = 1.0 / norm[active]
        return result

    def force_metrics(self, parameters, target):
        residual_squared = max(
            float(
                parameters @ self.gram @ parameters - 2 * parameters @ self.rhs + self.target_norm
            ),
            0.0,
        )
        relative = (
            float(np.sqrt(residual_squared / self.target_norm))
            if self.target_norm > 0
            else (0.0 if residual_squared == 0 else float("inf"))
        )
        return float(np.sqrt(residual_squared / len(target))), relative

    def order_force_rms(self, parameters, orders, counts, n_equations):
        result = {}
        offset = 0
        for order, count in zip(orders, counts, strict=True):
            values = parameters[offset : offset + count]
            block = self.gram[offset : offset + count, offset : offset + count]
            result[order] = float(np.sqrt(max(float(values @ block @ values), 0.0) / n_equations))
            offset += count
        return result

    def solve(self, scale, constraints, *, tolerance, max_iterations, verbose):
        return solve_gram_system(
            self.gram,
            self.rhs,
            self.target_norm,
            scale,
            constraints,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            reporter=self.reporter,
        )


def _gram_recovery_key(operator, target):
    """Fingerprint every numerical input needed to safely reuse a failed run."""
    digest = sha256(b"mlfcs-streaming-gram-v3-compact-coordinates-null-space")
    arrays = [operator.displacements, np.asarray(operator.basis_state), np.asarray(target)]
    if operator.parameter_map is not None:
        parameter_map = sparse.csc_matrix(operator.parameter_map)
        arrays.extend([parameter_map.data, parameter_map.indices, parameter_map.indptr])
    for tensor in operator.program.parameterizations:
        arrays.extend(
            [
                tensor.parameter_indices,
                tensor.parameter_mask,
                tensor.representative_from_pivots,
                tensor.rotations,
                tensor.component_permutations,
                tensor.coordinates,
                tensor.image_mask,
            ]
        )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()[:20]


def _force_metrics(predicted, reference):
    residual = np.asarray(predicted) - np.asarray(reference)
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.linalg.norm(reference))
    relative = float(np.linalg.norm(residual) / denominator) if denominator else float("inf")
    return rmse, relative


def _order_force_rms_from_reduced_gram(
    gram_system,
    reduced_parameters,
    parameter_map,
    operator,
    physical_parameters,
    orders,
    counts,
    n_equations,
):
    """Evaluate order contributions cheaply unless constraints mix IFC orders."""
    mapping = sparse.csc_matrix(parameter_map)
    boundaries = np.cumsum([0, *counts])
    column_orders = []
    mixed = False
    for column in range(mapping.shape[1]):
        rows = mapping.indices[mapping.indptr[column] : mapping.indptr[column + 1]]
        owners = np.searchsorted(boundaries[1:], rows, side="right")
        if len(np.unique(owners)) != 1:
            mixed = True
            break
        column_orders.append(int(owners[0]))
    if not mixed:
        result = {}
        column_orders = np.asarray(column_orders)
        for owner, order in enumerate(orders):
            selected = np.flatnonzero(column_orders == owner)
            values = reduced_parameters[selected]
            block = gram_system.gram[np.ix_(selected, selected)]
            result[order] = float(np.sqrt(max(float(values @ block @ values), 0.0) / n_equations))
        return result

    # Adjacent-order rotational identities genuinely couple orders.  In that
    # less common case, evaluate every physical order in one shared feature
    # pass instead of regenerating the same backend features once per order.
    predicted = operator.matvec_by_order(physical_parameters)
    return {
        order: float(np.linalg.norm(predicted[order]) / np.sqrt(n_equations)) for order in orders
    }


def _normalize_constraint_rows(constraints):
    """Equilibrate equality rows without changing their common null space."""
    if constraints.shape[0] == 0:
        return constraints
    norms = np.sqrt(np.asarray(constraints.multiply(constraints).sum(axis=1)).reshape(-1))
    scale = np.ones_like(norms)
    active = norms > np.finfo(float).tiny
    scale[active] = 1.0 / norms[active]
    return sparse.diags(scale) @ constraints
