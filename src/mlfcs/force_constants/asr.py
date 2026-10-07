"""Translational-invariance equations and force-constant projection.

For each tensor order, the acoustic sum rule sets the sum over the final
atomic position and its periodic images to zero, with the other positions
and Cartesian directions fixed. Projection minimizes the Euclidean change
of the selected physical tensor-component parameters.

High-order projection uses a matrix-free operator and LSMR. An explicit
representation of the same equations supports FC2 rotational projection.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.arrays import as_int64_array, readonly, require_allocation
from mlfcs.foundation.errors import ConstraintProjectionError
from mlfcs.foundation.log import get_logger
from mlfcs.foundation.tensors import apply_tensor_action, rotate_basis, tensor_dimension

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ASRReport:
    """Residuals and parameter change for one order's acoustic projection.

    Absolute residuals are the largest Cartesian sum-rule violations, in
    eV/angstrom**order. Relative residuals divide these by the largest
    equation row absolute sum times the largest parameter magnitude.
    ``correction_norm`` measures the Euclidean change of physical component
    parameters; ``relative_correction`` divides it by their initial norm.
    ``iterations`` counts LSMR steps across all refinement passes.
    """

    order: int
    equations: int
    parameters: int
    residual_before: float
    residual_after: float
    relative_before: float
    relative_after: float
    correction_norm: float
    relative_correction: float
    iterations: int


@dataclass(frozen=True, slots=True)
class ASRResult:
    """Force constants satisfying the selected acoustic sum rules and their reports.

    Unselected orders retain their original coefficients. ``report(order)``
    retrieves the diagnostics for an order included in the projection.
    """

    force_constants: ForceConstants
    reports: tuple[ASRReport, ...]

    def report(self, order: int) -> ASRReport:
        """Return one projected order's diagnostics; raise KeyError if it was not selected."""
        for value in self.reports:
            if value.order == order:
                return value
        raise KeyError(f"order {order} was not projected")


def acoustic_constraint_matrix(cluster_space: ClusterSpace, order: int) -> sparse.csr_matrix:
    """Construct Cartesian acoustic sum-rule equations for one tensor order.

    Each equation sums the force-constant tensor over the final atomic
    position and its periodic images, at fixed preceding positions and
    Cartesian directions. Columns follow the order-local physical component
    parameters. Multiplying by that parameter vector gives sum-rule
    residuals in eV/angstrom**order.

    The CSR matrix includes all Cartesian components, flattened in C order,
    for each distinct prefix in first-encounter order. Contributions from
    different images to the same coefficient are summed before use.
    A missing order raises KeyError. High-order ASR projection uses
    ``AcousticSumRuleOperator`` to avoid constructing this matrix.
    """
    space = cluster_space
    block = space.block(order)
    orbit_indices = range(block.orbits.start, block.orbits.stop)
    dimensions = [space.orbits[index].dimension for index in orbit_indices]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[object, ...], int] = {}
    rows = []
    columns = []
    data = []
    for local_orbit, orbit_index in enumerate(orbit_indices):
        orbit = space.orbits[orbit_index]
        for image, cluster in enumerate(orbit.clusters):
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            prefix = tuple((site.site, *site.translation) for site in cluster.sites[:-1])
            for component, directions in enumerate(np.ndindex((3,) * order)):
                key = (*prefix, *directions)
                row = equations.setdefault(key, len(equations))
                for column, value in enumerate(basis[component]):
                    if value != 0.0:
                        rows.append(row)
                        columns.append(int(offsets[local_orbit] + column))
                        data.append(float(value))
    matrix = sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(len(equations), int(offsets[-1])),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


@njit(cache=True, parallel=True, nogil=True)
def _matvec_kernel(
    values,
    out,
    prefix_group_offsets,
    prefix_group_images,
    image_table_entry,
    image_col_start,
    image_dimension,
    image_permutations,
    table_offsets,
    table,
    digit_powers,
    dim3,
):
    """Sum Cartesian tensors over the final position for every acoustic equation.

    Axis permutations act on C-order tensor components. Independent equation
    rows accumulate their image contributions in a fixed order.
    """
    n_rows = out.shape[0]
    order = digit_powers.shape[0]
    for row in prange(n_rows):
        prefix = row // dim3
        component = row - prefix * dim3
        acc = 0.0
        for idx in range(prefix_group_offsets[prefix], prefix_group_offsets[prefix + 1]):
            image = prefix_group_images[idx]
            source = 0
            for axis in range(order):
                direction = component // digit_powers[order - 1 - axis] % 3
                source += direction * digit_powers[order - 1 - image_permutations[image, axis]]
            width = image_dimension[image]
            table_row = table_offsets[image_table_entry[image]] + source * width
            column_base = image_col_start[image]
            for column in range(width):
                acc += table[table_row + column] * values[column_base + column]
        out[row] = acc


@njit(cache=True, parallel=True, nogil=True)
def _rmatvec_kernel(
    values,
    out,
    orbit_col_start,
    orbit_dimension,
    orbit_group_offsets,
    orbit_group_images,
    image_prefix_id,
    image_table_entry,
    image_permutations,
    table_offsets,
    table,
    digit_powers,
    dim3,
):
    """Accumulate acoustic residual weights into physical parameter coordinates.

    Orbit parameter blocks are disjoint, so each orbit can accumulate its
    transpose contribution independently without concurrent writes.
    """
    n_orbits = orbit_col_start.shape[0]
    order = digit_powers.shape[0]
    for orbit in prange(n_orbits):
        width = orbit_dimension[orbit]
        acc = np.zeros(width, dtype=np.float64)
        for idx in range(orbit_group_offsets[orbit], orbit_group_offsets[orbit + 1]):
            image = orbit_group_images[idx]
            table_base = table_offsets[image_table_entry[image]]
            row_base = image_prefix_id[image] * dim3
            for component in range(dim3):
                source = 0
                for axis in range(order):
                    direction = component // digit_powers[order - 1 - axis] % 3
                    source += direction * digit_powers[order - 1 - image_permutations[image, axis]]
                weight = values[row_base + component]
                table_row = table_base + source * width
                for column in range(width):
                    acc[column] += weight * table[table_row + column]
        column_base = orbit_col_start[orbit]
        for column in range(width):
            out[column_base + column] = acc[column]


@njit(cache=True, parallel=True, nogil=True)
def _row_abs_sum_kernel(
    out,
    prefix_group_offsets,
    prefix_group_images,
    image_col_start,
    image_dimension,
    image_table_entry,
    image_permutations,
    table_offsets,
    table,
    digit_powers,
    dim3,
    n_parameters,
):
    """Compute acoustic equation scales after combining duplicate coefficients.

    Signed contributions to each parameter are summed before taking their
    absolute values; otherwise cancellations would inflate the residual scale.
    """
    n_prefixes = prefix_group_offsets.shape[0] - 1
    order = digit_powers.shape[0]
    for prefix in prange(n_prefixes):
        scratch = np.zeros(n_parameters, dtype=np.float64)
        low = n_parameters
        high = 0
        base = prefix * dim3
        for component in range(dim3):
            for idx in range(prefix_group_offsets[prefix], prefix_group_offsets[prefix + 1]):
                image = prefix_group_images[idx]
                source = 0
                for axis in range(order):
                    direction = component // digit_powers[order - 1 - axis] % 3
                    source += direction * digit_powers[order - 1 - image_permutations[image, axis]]
                width = image_dimension[image]
                table_row = table_offsets[image_table_entry[image]] + source * width
                column_base = image_col_start[image]
                low = min(low, column_base)
                high = max(high, column_base + width)
                for column in range(width):
                    scratch[column_base + column] += table[table_row + column]
            total = 0.0
            for column in range(low, high):
                total += abs(scratch[column])
            out[base + component] = total
            for idx in range(prefix_group_offsets[prefix], prefix_group_offsets[prefix + 1]):
                image = prefix_group_images[idx]
                column_base = image_col_start[image]
                for column in range(image_dimension[image]):
                    scratch[column_base + column] = 0.0
            low = n_parameters
            high = 0


class AcousticSumRuleOperator(LinearOperator):
    """Acoustic sum-rule equations acting on one order's physical parameters.

    ``matvec`` sums Cartesian force constants over the final atomic position
    and its periodic images, at fixed other positions and tensor directions.
    ``rmatvec`` applies the transpose used in the minimum-norm correction.
    The equation system is the same as ``acoustic_constraint_matrix`` but
    does not store each image's contribution as a sparse matrix entry.

    Parameters
    ----------
    cluster_space
        Primitive cluster model supplying the tensor bases and orbit images.
    order
        Tensor order to project; it must exist in the cluster space.

    Notes
    -----
    Columns follow the order-local physical component parameters. Rows group
    all C-order Cartesian components of each distinct cluster prefix in
    first-encounter order. Rotated bases are shared between images using the
    same orbit and symmetry operation.
    """

    def __init__(self, cluster_space: ClusterSpace, order: int) -> None:
        """Prepare the acoustic equations and their residual scale for one order."""
        self._bridge = None
        space = cluster_space
        block = space.block(order)
        order_index = int(order)
        dim3 = tensor_dimension(order_index)
        identity = np.arange(order_index, dtype=np.int64)

        orbit_col_start_list: list[int] = []
        orbit_dimension_list: list[int] = []
        image_prefix_list: list[tuple[tuple[int, int, int, int], ...]] = []
        image_orbit_list: list[int] = []
        image_table_list: list[int] = []
        image_column_list: list[int] = []
        image_dimension_list: list[int] = []
        table_offsets_list: list[int] = [0]
        table_blocks: list[np.ndarray] = []

        column_cursor = 0
        for orbit_index in range(block.orbits.start, block.orbits.stop):
            local_orbit = len(orbit_dimension_list)
            orbit = space.orbits[orbit_index]
            width = orbit.dimension
            orbit_col_start_list.append(column_cursor)
            orbit_dimension_list.append(width)
            column_cursor += width
            entry_of_operation = {}
            for operation in np.unique(orbit.operations):
                rotation = readonly(space.symmetry.cartesian_rotations[operation].T, np.float64)
                rotated = apply_tensor_action(orbit.component_basis, rotation, identity)
                require_allocation("rotated basis table entry", rotated.shape)
                entry_of_operation[int(operation)] = len(table_offsets_list) - 1
                table_blocks.append(np.ascontiguousarray(rotated, dtype=np.float64).reshape(-1))
                table_offsets_list.append(table_offsets_list[-1] + rotated.size)
            for image, cluster in enumerate(orbit.clusters):
                image_prefix_list.append(cluster.labels[:-1])
                image_orbit_list.append(local_orbit)
                image_table_list.append(entry_of_operation[int(orbit.operations[image])])
                image_column_list.append(orbit_col_start_list[local_orbit])
                image_dimension_list.append(width)

        image_count = len(image_prefix_list)
        prefix_row_base: dict[tuple[tuple[int, int, int, int], ...], int] = {}
        image_prefix_id = np.empty(image_count, dtype=np.int64)
        for image, prefix in enumerate(image_prefix_list):
            base = prefix_row_base.get(prefix)
            if base is None:
                base = len(prefix_row_base) * dim3
                prefix_row_base[prefix] = base
            image_prefix_id[image] = base // dim3
        n_prefixes = len(prefix_row_base)
        n_parameters = int(block.parameters.stop - block.parameters.start)

        self._image_prefix_id = as_int64_array(image_prefix_id, name="image prefix ids")
        self._image_table_entry = as_int64_array(image_table_list, name="image table entries")
        self._image_col_start = as_int64_array(image_column_list, name="image column starts")
        self._image_dimension = as_int64_array(image_dimension_list, name="image dimensions")
        self._image_permutations = as_int64_array(
            np.concatenate(
                [
                    space.orbits[index].permutations
                    for index in range(block.orbits.start, block.orbits.stop)
                ]
            )
            if image_count
            else np.empty((0, order_index), dtype=np.int64),
            name="image permutations",
        )
        self._table = (
            readonly(np.concatenate(table_blocks), np.float64)
            if table_blocks
            else readonly(np.empty(0, dtype=np.float64), np.float64)
        )
        self._table_offsets = as_int64_array(table_offsets_list, name="table offsets")
        self._orbit_col_start = as_int64_array(orbit_col_start_list, name="orbit column starts")
        self._orbit_dimension = as_int64_array(orbit_dimension_list, name="orbit dimensions")
        self._digit_powers = as_int64_array(
            [3**axis for axis in range(order_index)], name="digit powers"
        )
        self._n_parameters = n_parameters

        self._prefix_group_offsets, self._prefix_group_images = self._group(
            n_prefixes, self._image_prefix_id
        )
        self._orbit_group_offsets, self._orbit_group_images = self._group(
            len(orbit_dimension_list), np.asarray(image_orbit_list, dtype=np.int64)
        )

        n_rows = n_prefixes * dim3
        row_abs_sum = np.empty(n_rows, dtype=np.float64)
        _row_abs_sum_kernel(
            row_abs_sum,
            self._prefix_group_offsets,
            self._prefix_group_images,
            self._image_col_start,
            self._image_dimension,
            self._image_table_entry,
            self._image_permutations,
            self._table_offsets,
            self._table,
            self._digit_powers,
            dim3,
            n_parameters,
        )
        self._max_row_abs_sum = float(np.max(row_abs_sum, initial=0.0))
        self._dim3 = dim3
        super().__init__(dtype=np.float64, shape=(n_rows, n_parameters))

    @staticmethod
    def _group(groups: int, member_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Group images by prefix or orbit, preserving their accumulation order."""
        order = np.argsort(member_ids, kind="stable")
        sorted_ids = member_ids[order]
        offsets = np.searchsorted(sorted_ids, np.arange(groups + 1), side="left")
        return (
            as_int64_array(offsets, name="group offsets"),
            as_int64_array(order, name="group members"),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> AcousticSumRuleOperator:
        """Represent an explicit equation matrix with the acoustic operator interface.

        This adapter supports independent dense or sparse reference equations.
        The matrix must be finite and two-dimensional; otherwise ValueError is
        raised. Its columns represent parameters and its rows represent equations.
        """
        wrapper = cls.__new__(cls)
        require_allocation("equation matrix", matrix.shape)
        dense = np.ascontiguousarray(
            matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float64
        )
        if dense.ndim != 2:
            raise ValueError("equation matrix must be two-dimensional")
        if not np.all(np.isfinite(dense)):
            raise ValueError("equation matrix contains NaN or infinite values")
        wrapper._bridge = dense
        wrapper._n_parameters = dense.shape[1]
        wrapper._max_row_abs_sum = float(np.max(np.abs(dense).sum(axis=1), initial=0.0))
        LinearOperator.__init__(wrapper, dtype=np.float64, shape=dense.shape)
        return wrapper

    @property
    def max_row_abs_sum(self) -> float:
        """Largest equation row absolute sum after signed image contributions are combined."""
        return self._max_row_abs_sum

    def _matvec(self, values: np.ndarray) -> np.ndarray:
        """Return acoustic residuals for a finite physical parameter vector."""
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self._n_parameters,):
            raise ValueError(
                f"ASR operator expects {self._n_parameters} parameters, got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("ASR operator input contains NaN or infinite values")
        if self._bridge is not None:
            return self._bridge @ vector
        out = np.empty(self.shape[0], dtype=np.float64)
        _matvec_kernel(
            vector,
            out,
            self._prefix_group_offsets,
            self._prefix_group_images,
            self._image_table_entry,
            self._image_col_start,
            self._image_dimension,
            self._image_permutations,
            self._table_offsets,
            self._table,
            self._digit_powers,
            self._dim3,
        )
        return out

    def _rmatvec(self, values: np.ndarray) -> np.ndarray:
        """Map finite acoustic residual weights back to physical parameter coordinates."""
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (self.shape[0],):
            raise ValueError(
                f"ASR operator transpose expects {self.shape[0]} rows, got {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("ASR operator input contains NaN or infinite values")
        if self._bridge is not None:
            return self._bridge.T @ vector
        out = np.empty(self._n_parameters, dtype=np.float64)
        _rmatvec_kernel(
            vector,
            out,
            self._orbit_col_start,
            self._orbit_dimension,
            self._orbit_group_offsets,
            self._orbit_group_images,
            self._image_prefix_id,
            self._image_table_entry,
            self._image_permutations,
            self._table_offsets,
            self._table,
            self._digit_powers,
            self._dim3,
        )
        return out


def _operator_relative_residual(
    operator: AcousticSumRuleOperator, values: np.ndarray
) -> tuple[float, float]:
    """Measure the largest acoustic violation and its dimensionless scaled value.

    Scale by the largest equation row absolute sum times the largest
    parameter magnitude. Empty systems and zero scales return zero.
    """
    if operator.shape[0] == 0 or values.size == 0:
        return 0.0, 0.0
    residual = np.asarray(operator.matvec(values))
    maximum = float(np.max(np.abs(residual), initial=0.0))
    parameter_scale = float(np.max(np.abs(values), initial=0.0))
    scale = operator.max_row_abs_sum * parameter_scale
    relative = maximum / scale if scale else 0.0
    return maximum, relative


def _project(
    order: int,
    operator: AcousticSumRuleOperator,
    parameters: np.ndarray,
    *,
    rtol: float,
    name: str = "ASR",
) -> tuple[np.ndarray, ASRReport]:
    """Remove minimum-norm parameter corrections until the acoustic tolerance is met.

    LSMR operates in physical component coordinates. Up to three refinement
    passes reduce roundoff; failure to reach ``rtol`` raises
    ConstraintProjectionError.
    """
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (operator.shape[1],):
        raise ValueError(
            f"order-{order} {name} expects {operator.shape[1]} parameters, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"order-{order} parameters contain NaN or infinite values")
    before, relative_before = _operator_relative_residual(operator, values)
    if operator.shape[0] == 0 or values.size == 0 or relative_before <= rtol:
        return values.copy(), ASRReport(
            order,
            operator.shape[0],
            operator.shape[1],
            before,
            before,
            relative_before,
            relative_before,
            0.0,
            0.0,
            0,
        )

    projected = values.copy()
    correction = np.zeros_like(values)
    iterations = 0
    solver_rtol = max(rtol * 0.1, np.finfo(float).eps)
    maximum_steps = max(1, 4 * min(operator.shape))
    for _ in range(3):
        residual = np.asarray(operator.matvec(projected))
        solution = lsmr(
            operator,
            residual,
            atol=solver_rtol,
            btol=solver_rtol,
            conlim=0.0,
            maxiter=maximum_steps,
        )
        step = np.asarray(solution[0])
        if not np.all(np.isfinite(step)):
            raise ConstraintProjectionError(
                f"order-{order} {name} projection produced a non-finite correction"
            )
        projected -= step
        correction += step
        iterations += int(solution[2])
        after, relative_after = _operator_relative_residual(operator, projected)
        if relative_after <= rtol:
            break
    else:
        raise ConstraintProjectionError(
            f"order-{order} {name} projection did not converge: relative residual "
            f"{relative_before:.6e} -> {relative_after:.6e}, requested {rtol:.6e}"
        )

    correction_norm = float(np.linalg.norm(correction))
    parameter_norm = float(np.linalg.norm(values))
    relative_correction = correction_norm / parameter_norm if parameter_norm else 0.0
    return projected, ASRReport(
        order,
        operator.shape[0],
        operator.shape[1],
        before,
        after,
        relative_before,
        relative_after,
        correction_norm,
        relative_correction,
        iterations,
    )


def enforce_asr(
    model: ForceConstants,
    *,
    orders: Iterable[int] | None = None,
    rtol: float = 1e-10,
) -> ASRResult:
    """Project selected force-constant orders onto translational invariance.

    The acoustic sum rule requires the sum over the final atomic position
    and all included periodic images to vanish for fixed other positions
    and Cartesian tensor directions. Each order is projected independently
    by minimizing the Euclidean change of its physical component parameters.
    The result contains a new model and per-order residual diagnostics.

    Parameters
    ----------
    model
        Primitive force constants to project.
    orders
        Ascending, unique orders to project; defaults to all stored orders.
        Coefficients of unselected orders are retained.
    rtol
        Positive relative equation tolerance. Residuals are normalized by
        the largest equation row absolute sum times the largest parameter
        magnitude, not by the initial violation.

    Raises
    ------
    KeyError
        If a requested order is absent.
    ValueError
        If orders or the tolerance are invalid.
    ConstraintProjectionError
        If a finite correction meeting the requested tolerance cannot be found.

    Notes
    -----
    This floating-point projection is separate from the integer kernel used
    to construct orbit-invariant tensor bases. The source model is unchanged.
    """
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive")
    selected = model.orders if orders is None else tuple(int(order) for order in orders)
    if tuple(sorted(set(selected))) != selected:
        raise ValueError("orders must be unique and ascending")
    missing = tuple(order for order in selected if order not in model.coefficients)
    if missing:
        raise KeyError(f"force constants do not contain orders {missing}")

    coefficients = dict(model.coefficients)
    started = perf_counter()
    logger.info("ASR projection started: orders=%s rtol=%.3g", selected, rtol)
    reports = []
    for order in selected:
        operator = AcousticSumRuleOperator(model.cluster_space, order)
        projected, report = _project(
            order,
            operator,
            model.coefficients[order],
            rtol=float(rtol),
        )
        coefficients[order] = projected
        reports.append(report)
        logger.info(
            "ASR order complete: order=%d equations=%d iterations=%d "
            "relative_before=%.6g relative_after=%.6g correction_norm=%.6g",
            order,
            report.equations,
            report.iterations,
            report.relative_before,
            report.relative_after,
            report.correction_norm,
        )
    result = ASRResult(ForceConstants(model.cluster_space, coefficients), tuple(reports))
    logger.info(
        "ASR projection complete: orders=%s elapsed_s=%.2f", selected, perf_counter() - started
    )
    return result


__all__ = [
    "ASRReport",
    "ASRResult",
    "AcousticSumRuleOperator",
    "acoustic_constraint_matrix",
    "enforce_asr",
]
