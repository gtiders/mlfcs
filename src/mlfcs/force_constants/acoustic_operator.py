"""Matrix-free Cartesian acoustic sum-rule equation operator for one order.

The operator replaces the explicit COO/CSR assembly in
``acoustic.acoustic_constraint_matrix`` for the streamed ASR projection path.
Rows and columns carry the same meaning as the sparse builder:

- A row is one translational-invariance equation. For tensor order ``p`` it is
  addressed by ``(prefix, component)``: ``prefix`` collects the first ``p-1``
  lattice labels ``(site, tx, ty, tz)`` shared by the contributing cluster
  images, and ``component`` is one of the ``3**p`` C-order Cartesian direction
  multi-indices. Summing the rotated force-constant tensor over the final
  slot's images must vanish for fixed prefix and directions.
- A column is one order-local physical parameter; orbit parameter blocks are
  contiguous and follow ``ClusterSpace.block(order).parameters``.

Memory strategy. The naive assembly materialises one scatter entry per
(image, component, nonzero column) triple, which reaches ~88 million Python
objects for the Si FC5 workload. Two structural facts remove that cost:

1. Every image contributes exactly ``3**p`` rows keyed by its prefix, and the
   distinct prefixes number only thousands, so rows are addressed
   arithmetically as ``prefix_id * 3**p + component`` after one
   first-encounter enumeration of image prefixes. The enumeration mirrors the
   sparse builder's ``dict.setdefault`` order, so row numbering agrees with
   the CSR oracle exactly.
2. ``rotate_basis(B, R, pi)`` factorises into a rotation that depends only on
   the symmetry operation followed by a pure axis relabel of the component
   multi-index. The rotated bases are therefore tabulated per
   ``(orbit, operation)`` pair -- tens of megabytes for real materials -- and
   the per-image permutation is applied inside the kernels as index
   arithmetic with no floating-point cost.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator

from mlfcs._arrays import as_int64_array, readonly, require_allocation
from mlfcs.cluster_space import ClusterSpace
from mlfcs.tensors import apply_tensor_action, tensor_dimension


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
    """Apply the equation operator: out[r] = sum_c A[r, c] * values[c].

    ``values`` is the physical parameter vector and ``out`` the residual vector
    of shape ``(n_prefixes * 3**p,)``. Rows are processed in parallel and each
    row accumulates its images in stored order, so duplicate contributions to
    one (row, column) pair sum exactly once and the result is deterministic.
    ``digit_powers[k]`` equals ``3**k`` and ``dim3`` equals ``3**p``.
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
    """Apply the transposed operator: out[c] = sum_r A[r, c] * values[r].

    Orbits own disjoint parameter columns and are processed in parallel; the
    images of one orbit accumulate into a thread-local scratch before a single
    write, keeping the result deterministic. ``digit_powers[k]`` equals
    ``3**k`` and ``dim3`` equals ``3**p``.
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
    """Compute per-row absolute sums of the deduplicated equation matrix.

    Mirrors the CSR semantics of ``sum_duplicates`` followed by
    ``sum(abs(row))``: contributions to one (row, column) pair accumulate with
    their signs first -- duplicates arise only inside one orbit's column block
    because orbit blocks are disjoint -- and absolute values are taken
    afterwards. ``out`` has shape ``(n_prefixes * 3**p,)`` and
    ``digit_powers[k]`` equals ``3**k``.
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
    """Matrix-free Cartesian ASR equation operator for one order block.

    The operator represents the same equation matrix as
    ``acoustic.acoustic_constraint_matrix`` without materialising it: rows are
    ``(prefix, component)`` translational-invariance equations as described in
    the module docstring, columns are the order-local physical parameters.

    Parameters
    ----------
    cluster_space : ClusterSpace
        Immutable primitive model; the operator reads orbit bases, cluster
        images, symmetry operations and permutations and never mutates them.
    order : int
        Tensor order ``p >= 2`` selecting the order block; the block must
        exist in the cluster space.

    Notes
    -----
    Shape is ``(n_prefixes * 3**p, n_order_parameters)`` where ``n_prefixes``
    counts distinct image prefixes over the block; this equals the CSR row
    count of the sparse builder because every image registers all ``3**p``
    components of its prefix, and rows are numbered in the same
    first-encounter order as the builder's ``dict.setdefault``, so the two
    matrices agree row for row. ``matvec``/``rmatvec`` accumulate duplicate
    contributions per (row, column) pair, matching the CSR ``sum_duplicates``
    semantics. The rotated-basis table is bounded by ``sum over orbits of
    n_distinct_operations * 3**p * orbit_dimension`` float64 values and is
    admission-checked before allocation.
    """

    def __init__(self, cluster_space: ClusterSpace, order: int) -> None:
        """Build the metadata tables and operator state for one order."""
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
        self._row_abs_sums = readonly(row_abs_sum, np.float64)
        self._dim3 = dim3
        super().__init__(dtype=np.float64, shape=(n_rows, n_parameters))

    @staticmethod
    def _group(groups: int, member_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Group member indices by group id into CSR-style offsets and indices.

        ``member_ids`` holds one group id per member. Returns offsets of shape
        ``(groups + 1,)`` and member indices sorted by group, stable within a
        group. Both arrays are readonly int64.
        """
        order = np.argsort(member_ids, kind="stable")
        sorted_ids = member_ids[order]
        offsets = np.searchsorted(sorted_ids, np.arange(groups + 1), side="left")
        return (
            as_int64_array(offsets, name="group offsets"),
            as_int64_array(order, name="group members"),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> AcousticSumRuleOperator:
        """Wrap an explicit equation matrix with the same operator protocol.

        The bridge serves dense or sparse oracles in tests: ``matvec`` and
        ``rmatvec`` delegate to the matrix and ``max_row_abs_sum`` matches the
        CSR row-norm definition. ``matrix`` must be two-dimensional and finite.
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
        """Return the largest per-row absolute sum of the deduplicated equations."""
        return self._max_row_abs_sum

    def row_abs_sums(self) -> np.ndarray:
        """Return per-row absolute sums, one entry per equation row.

        Matches ``np.abs(matrix).sum(axis=1)`` of the sparse builder: each row
        sums the absolute values of its deduplicated (row, column) entries.
        """
        return self._row_abs_sums

    def _matvec(self, values: np.ndarray) -> np.ndarray:
        """Apply the equation operator to one parameter vector."""
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
        """Apply the transposed equation operator to one residual vector."""
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


def operator_relative_residual(
    operator: AcousticSumRuleOperator, values: np.ndarray
) -> tuple[float, float]:
    """Return maximum absolute equation residual and its dimensionless scaled value.

    Matches ``acoustic.relative_residual``: the scale is the operator's maximum
    row absolute-sum norm times the maximum absolute parameter. Empty systems
    return ``(0, 0)``; a zero scale gives relative residual zero.
    """
    if operator.shape[0] == 0 or values.size == 0:
        return 0.0, 0.0
    residual = np.asarray(operator.matvec(values))
    maximum = float(np.max(np.abs(residual), initial=0.0))
    parameter_scale = float(np.max(np.abs(values), initial=0.0))
    scale = operator.max_row_abs_sum * parameter_scale
    relative = maximum / scale if scale else 0.0
    return maximum, relative


__all__ = ["AcousticSumRuleOperator", "operator_relative_residual"]
