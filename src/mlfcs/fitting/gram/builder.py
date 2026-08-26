"""Streamed construction of portable least-squares sufficient statistics."""

from __future__ import annotations

import logging
from functools import partial
from time import perf_counter

import jax
import numpy as np
from scipy.linalg.blas import dsyrk

from mlfcs.fitting.design_operator import accumulate_physical_design, prepare_design_kernel_groups
from mlfcs.fitting.gram.models import GramStatistics

logger = logging.getLogger(__name__)


@partial(jax.jit, donate_argnums=(0, 1))
def _update_device_statistics(current_gram, current_rhs, design, force):
    return current_gram + design.T @ design, current_rhs + design.T @ force


def _orbit_parameter_groups(calculations):
    """Return contiguous slices, one per symmetry-irreducible cluster orbit."""
    groups = []
    offset = 0
    for calculation in calculations:
        for orbit in calculation.realized_orbit_space.orbits:
            groups.append(slice(offset, offset + orbit.dimension))
            offset += orbit.dimension
    return tuple(groups)


class GramAccumulator:
    """Normal-equation sufficient statistics accumulated without storing A."""

    def __init__(self, gram, rhs, target_norm):
        self.gram = gram
        self.rhs = rhs
        self.target_norm = target_norm

    @classmethod
    def from_operator(cls, operator, target):
        del cls
        started = perf_counter()
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
        builders, effective_batch_size = prepare_design_kernel_groups(operator)
        rows_per_structure = int(np.prod(operator.force_shape[1:]))
        operator.program.gram_feature_passes += 1
        if logger.isEnabledFor(logging.INFO):
            tile_counts = [group.tile_count for group in builders]
            logger.info(
                f"Accumulating streamed Gram system: {operator.fit_n_parameters} x "
                f"{operator.fit_n_parameters} ({gram.nbytes / 1024**2:.1f} MiB), "
                f"effective_batch_size={effective_batch_size}, "
                f"backend={'JAX device' if use_device_gram else 'SciPy/OpenBLAS CPU'}"
            )
            logger.info(
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
                if logger.isEnabledFor(logging.INFO) and begin == 0:
                    contributions.block_until_ready()
                    logger.info(
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
            if logger.isEnabledFor(logging.INFO) and (
                begin == 0 or end == len(operator.displacements) or end % 20 == 0
            ):
                if use_device_gram:
                    gram.block_until_ready()
                logger.info(
                    f"- Gram structures: {end}/{len(operator.displacements)}, "
                    f"elapsed={perf_counter() - started:.2f} s"
                )
        if use_device_gram:
            gram = np.asarray(jax.device_get(gram))
            rhs = np.asarray(jax.device_get(rhs))
        else:
            upper = np.triu(np.asarray(gram))
            gram = upper + np.triu(upper, 1).T
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"- Streamed Gram system ready in {perf_counter() - started:.2f} s")
            if use_device_gram:
                logger.info(
                    "- Gram phase timing: device kernel, scatter, reduction, and BLAS "
                    "execute asynchronously as one device pipeline"
                )
            else:
                logger.info(
                    "- Gram phase timing: "
                    f"kernel={kernel_seconds:.2f} s, transfer={transfer_seconds:.2f} s, "
                    f"scatter={scatter_seconds:.2f} s, "
                    f"constraint reduction={reduction_seconds:.2f} s, "
                    f"BLAS={gram_seconds:.2f} s"
                )
        target_norm = float(np.vdot(target, target))
        return GramStatistics(
            np.asarray(gram), np.asarray(rhs), target_norm, len(target),
            {
                "fitting_basis": getattr(operator, "fitting_basis", "unknown"),
                "covariance": np.asarray(operator.basis_state),
                "parameter_map": operator.parameter_map,
            },
        )

class GramBuilder:
    """Explicit entry point for one-shot, device-independent Gram construction."""

    @classmethod
    def from_operator(cls, operator, target, *, batch_size=None) -> GramStatistics:
        if batch_size is not None and batch_size != operator.batch_size:
            raise ValueError(
                "batch_size must match the already prepared ForceDesignOperator; "
                "construct a new operator to change it"
            )
        del cls
        return GramAccumulator.from_operator(operator, target)
