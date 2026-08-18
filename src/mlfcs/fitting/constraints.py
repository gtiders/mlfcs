from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import sparse

from mlfcs.core.constraints import (
    build_translational_constraints,
)
from mlfcs.core.orbits import cluster_invariant_dimension


@dataclass(frozen=True, slots=True)
class JointConstraints:
    matrix: sparse.csr_matrix
    translational_rows: int


def append_zero_taylor_order_constraints(
    constraints: JointConstraints, calculations, covariance, orders
) -> JointConstraints:
    """Require selected residual Taylor orders to vanish exactly."""
    orders = set(orders)
    if not orders:
        return constraints
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    offsets = np.cumsum([0, *dimensions])
    transform = build_wick_to_taylor_transform(calculations, covariance)
    rows = []
    for index, calculation in enumerate(calculations):
        if calculation.config.order in orders:
            rows.append(transform[offsets[index] : offsets[index + 1]])
    matrix = sparse.vstack([constraints.matrix, *rows], format="csr")
    return JointConstraints(_compress_rows(matrix), constraints.translational_rows)


def build_joint_constraints(
    calculations,
    *,
    acoustic: bool,
) -> JointConstraints:
    """Build only translational constraints in the fitting parameter basis.

    Born--Huang and Huang conditions deliberately live in the explicit FC2
    postprocessor.  Applying them here would couple Wick FC2 to FC4.
    """
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    total = sum(dimensions)
    translational = []
    if acoustic:
        for index, calculation in enumerate(calculations):
            local = build_translational_constraints(calculation.orbit_space)
            left = sum(dimensions[:index])
            right = total - left - dimensions[index]
            translational.append(
                sparse.hstack(
                    [
                        sparse.csr_matrix((local.shape[0], left)),
                        local,
                        sparse.csr_matrix((local.shape[0], right)),
                    ],
                    format="csr",
                )
            )
    matrices = translational
    matrix = sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, total))
    matrix = _compress_rows(matrix)
    return JointConstraints(
        matrix,
        sum(item.shape[0] for item in translational),
    )


def build_wick_to_taylor_transform(calculations, covariance) -> sparse.csr_matrix:
    """Map symmetry-reduced Wick parameters to ordinary Taylor IFC parameters.

    The map is the identity plus same-parity contractions with the displacement
    covariance.  Building it in the irreducible orbit basis lets Taylor ASR and
    rotational constraints be imposed exactly during a Wick-basis fit.
    """
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    offsets = np.cumsum([0, *dimensions])
    total = int(offsets[-1])
    transform = sparse.eye(total, format="lil", dtype=float)
    covariance = np.asarray(covariance).reshape(
        len(calculations[0].supercell), 3, len(calculations[0].supercell), 3
    )
    by_order = {
        calculation.config.order: (index, calculation)
        for index, calculation in enumerate(calculations)
    }
    image_maps = {}
    orbit_images = {}
    for order, (index, calculation) in by_order.items():
        images = {}
        grouped = {}
        for cluster, columns, local_offset in _image_columns(calculation):
            global_offset = int(offsets[index] + local_offset)
            images[cluster] = (global_offset, columns)
            grouped.setdefault(global_offset, []).append((cluster, columns))
        image_maps[order] = images
        orbit_images[order] = grouped

    for target_order in by_order:
        for source_order in range(target_order + 2, max(by_order, default=0) + 1, 2):
            if source_order not in by_order:
                continue
            source_index, source = by_order[source_order]
            pairs = (source_order - target_order) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            contracted_by_target: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
            contraction_scales: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
            for cluster, columns, local_offset in _image_columns(source):
                contracted = columns.reshape((3,) * source_order + (-1,))
                for pair in reversed(range(pairs)):
                    left = target_order + 2 * pair
                    contracted = np.einsum(
                        "...abp,ab->...p",
                        contracted,
                        covariance[cluster[left], :, cluster[left + 1], :],
                        optimize=True,
                    )
                target_cluster = cluster[:target_order]
                source_offset = int(offsets[source_index] + local_offset)
                contributions = contracted_by_target.setdefault(target_cluster, {})
                contribution = contracted.reshape(3**target_order, -1)
                contributions[source_offset] = (
                    contributions.get(source_offset, np.zeros_like(contribution)) + contribution
                )
                scales = contraction_scales.setdefault(target_cluster, {})
                scales[source_offset] = scales.get(
                    source_offset, np.zeros_like(contribution)
                ) + np.abs(contribution)
            _validate_missing_contractions(
                contracted_by_target,
                contraction_scales,
                image_maps[target_order],
                by_order[target_order][1],
                source_order=source_order,
                target_order=target_order,
            )
            # All symmetry images of one target orbit describe the same target
            # parameter vector.  Solve them together.  Solving each image and
            # adding the answers would multiply the contraction by the number
            # of target images.
            for target_offset, target_images in orbit_images[target_order].items():
                source_offsets = sorted(
                    {
                        source_offset
                        for cluster, _columns in target_images
                        for source_offset in contracted_by_target.get(cluster, {})
                    }
                )
                if not source_offsets:
                    continue
                widths = {
                    source_offset: next(
                        contracted_by_target[cluster][source_offset].shape[1]
                        for cluster, _columns in target_images
                        if source_offset in contracted_by_target.get(cluster, {})
                    )
                    for source_offset in source_offsets
                }
                target_columns = np.vstack([columns for _cluster, columns in target_images])
                contracted_rows = []
                for cluster, columns in target_images:
                    contributions = contracted_by_target.get(cluster, {})
                    contracted_rows.append(
                        np.concatenate(
                            [
                                contributions.get(
                                    source_offset,
                                    np.zeros((len(columns), widths[source_offset])),
                                )
                                for source_offset in source_offsets
                            ],
                            axis=1,
                        )
                    )
                contracted = np.vstack(contracted_rows)
                mapping = np.linalg.lstsq(target_columns, contracted, rcond=None)[0]
                begin = 0
                for source_offset in source_offsets:
                    width = widths[source_offset]
                    transform[
                        target_offset : target_offset + mapping.shape[0],
                        source_offset : source_offset + width,
                    ] += coefficient * mapping[:, begin : begin + width]
                    begin += width
    return transform.tocsr()


def _validate_missing_contractions(
    contracted_by_target,
    contraction_scales,
    target_images,
    target_calculation,
    *,
    source_order,
    target_order,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-9,
):
    """Classify missing Wick contractions after all symmetry images are aggregated."""
    missing = set(contracted_by_target).difference(target_images)
    for target_cluster in sorted(missing):
        contributions = contracted_by_target[target_cluster]
        magnitude = max(
            (float(np.max(np.abs(values))) for values in contributions.values()),
            default=0.0,
        )
        scale = max(
            (
                float(np.max(values))
                for values in contraction_scales.get(target_cluster, {}).values()
            ),
            default=0.0,
        )
        threshold = absolute_tolerance + relative_tolerance * scale
        dimension = cluster_invariant_dimension(
            target_cluster,
            target_calculation.index,
            target_calculation.symmetry,
        )
        if dimension == 0 and magnitude <= threshold:
            continue
        if dimension == 0:
            raise RuntimeError(
                f"Wick-to-Taylor FC{source_order}->FC{target_order} contraction "
                f"creates symmetry-forbidden cluster {target_cluster} with maximum "
                f"coefficient {magnitude:.6e} above tolerance {threshold:.6e}; "
                "check covariance symmetrization, periodic representatives, and image "
                "aggregation"
            )
        raise ValueError(
            f"Wick-to-Taylor contraction creates symmetry-allowed FC{target_order} "
            f"cluster {target_cluster} outside its configured support "
            f"(allowed dimension={dimension}, maximum coefficient={magnitude:.6e})"
        )


def omitted_taylor_fc1(calculations, parameters, covariance) -> np.ndarray:
    """Return the constant-force Taylor term generated by odd Wick orders."""
    transform = build_wick_to_taylor_fc1_transform(calculations, covariance)
    return np.asarray(transform @ np.asarray(parameters)).reshape(-1, 3)


def build_wick_to_taylor_fc1_transform(calculations, covariance) -> sparse.csr_matrix:
    """Map Wick parameters to Taylor FC1 at the reference structure."""
    covariance = np.asarray(covariance).reshape(
        len(calculations[0].supercell), 3, len(calculations[0].supercell), 3
    )
    n_primitive = calculations[0].index.n_primitive
    total = sum(_parameter_count(calculation) for calculation in calculations)
    matrix_rows = []
    matrix_columns = []
    matrix_data = []
    offset = 0
    for calculation in calculations:
        dimension = _parameter_count(calculation)
        order = calculation.config.order
        if order % 2:
            pairs = (order - 1) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            for cluster, columns, local_offset in _image_columns(calculation):
                contracted = columns.reshape((3,) * order + (-1,))
                for pair in reversed(range(pairs)):
                    left = 1 + 2 * pair
                    contracted = np.einsum(
                        "...abp,ab->...p",
                        contracted,
                        covariance[cluster[left], :, cluster[left + 1], :],
                        optimize=True,
                    )
                for direction in range(3):
                    coefficients = coefficient * contracted[direction]
                    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-12)
                    primitive_site = int(calculation.index.primitive[int(cluster[0])])
                    matrix_rows.extend([primitive_site * 3 + direction] * len(nonzero))
                    matrix_columns.extend(offset + local_offset + int(value) for value in nonzero)
                    matrix_data.extend(float(coefficients[value]) for value in nonzero)
        offset += dimension
    return sparse.coo_matrix(
        (matrix_data, (matrix_rows, matrix_columns)),
        shape=(n_primitive * 3, total),
    ).tocsr()


def _parameter_count(calculation):
    return sum(orbit.dimension for orbit in calculation.orbit_space.orbits)


def _image_columns(calculation):
    """Yield cluster and its dense Cartesian-component-to-pivot map."""
    offset = 0
    for orbit in calculation.orbit_space.orbits:
        representative = np.linalg.solve(orbit.basis[orbit.pivots].T, orbit.basis.T).T
        for image in orbit.images:
            yield image.cluster, image.action.apply_columns(representative), offset
        offset += orbit.dimension


def _compress_rows(matrix, tolerance=1e-12):
    """Drop empty rows, normalize, and remove numerically identical constraints."""
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    matrix = matrix[norms > tolerance]
    norms = norms[norms > tolerance]
    matrix = sparse.diags(1.0 / norms) @ matrix
    rounded = matrix.copy()
    rounded.data = np.round(rounded.data, 12)
    keys = []
    for row in range(rounded.shape[0]):
        begin, end = rounded.indptr[row : row + 2]
        indices = rounded.indices[begin:end]
        values = rounded.data[begin:end]
        if len(values) and values[0] < 0:
            values = -values
        keys.append((indices.tobytes(), values.tobytes()))
    keep = []
    seen = set()
    for row, key in enumerate(keys):
        if key not in seen:
            seen.add(key)
            keep.append(row)
    return matrix[np.asarray(keep, dtype=np.int64)].tocsr()
