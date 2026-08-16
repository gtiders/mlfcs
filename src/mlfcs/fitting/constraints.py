from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import factorial

import numpy as np
from scipy import sparse
from scipy.linalg import qr

from mlfcs.core.constraints import (
    build_harmonic_rotational_constraints,
    build_translational_constraints,
)
from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.orbits import cluster_invariant_dimension


@dataclass(frozen=True, slots=True)
class JointConstraints:
    matrix: sparse.csr_matrix
    translational_rows: int
    rotational_rows: int
    rotational_mode: int


def build_joint_constraints(
    calculations,
    *,
    acoustic: bool,
    rotational_mode: int,
    covariance: np.ndarray | None = None,
) -> JointConstraints:
    """Build ASR and the complete adjacent-order rotation hierarchy.

    In a Wick fit the lowest identity couples the fitted Taylor FC1 to FC2.
    Both modes include that identity and every represented adjacent-order
    identity. Mode 2 leaves the upper boundary open; mode 3 closes it.
    """
    if rotational_mode not in (0, 2, 3):
        raise ValueError("rotational_mode must be 0, 2, or 3")
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
    rotational = []
    harmonic_index = None
    if rotational_mode:
        harmonic_index = next(
            (index for index, item in enumerate(calculations) if item.config.order == 2),
            None,
        )
        for index, (lower, upper) in enumerate(pairwise(calculations)):
            rotational.append(_adjacent_rotational_constraints(lower, upper, dimensions, index))
        if rotational_mode == 3:
            rotational.append(_highest_order_rotational_boundary(calculations[-1], dimensions))
    matrices = translational + rotational
    matrix = sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, total))
    lower_rows = 0
    if rotational_mode:
        if covariance is None:
            raise ValueError("covariance is required for rotational constraints in the Wick basis")
        transform = build_wick_to_taylor_transform(calculations, covariance)
        matrix = matrix @ transform
        if harmonic_index is not None:
            fc1 = build_wick_to_taylor_fc1_transform(calculations, covariance)
            lower = _lowest_order_rotational_constraints(
                calculations,
                dimensions,
                harmonic_index,
                transform,
                fc1,
            )
            lower = _independent_constraint_rows(
                lower, tolerance=calculations[harmonic_index].config.symprec
            )
            lower_rows = lower.shape[0]
            matrix = sparse.vstack([matrix, lower], format="csr")
    matrix = _compress_rows(matrix)
    return JointConstraints(
        matrix,
        sum(item.shape[0] for item in translational),
        lower_rows + sum(item.shape[0] for item in rotational),
        rotational_mode,
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


def _fc1_rotation_matrix(n_primitive: int, tolerance: float = 1e-12) -> sparse.csr_matrix:
    """Map Taylor FC1 to its term in the lowest FC1--FC2 identity."""
    matrix_rows = []
    matrix_columns = []
    matrix_data = []
    axes = np.eye(3)
    for atom in range(n_primitive):
        for force_direction in range(3):
            for rotation_axis in range(3):
                equation = (atom * 3 + force_direction) * 3 + rotation_axis
                for component in range(3):
                    # Harmonic moment minus (omega x FC1) equals zero.
                    value = -np.cross(axes[rotation_axis], axes[component])[force_direction]
                    if abs(value) > tolerance:
                        matrix_rows.append(equation)
                        matrix_columns.append(atom * 3 + component)
                        matrix_data.append(float(value))
    return sparse.coo_matrix(
        (matrix_data, (matrix_rows, matrix_columns)),
        shape=(n_primitive * 9, n_primitive * 3),
    ).tocsr()


def _parameter_count(calculation):
    return sum(orbit.dimension for orbit in calculation.orbit_space.orbits)


def _lowest_order_rotational_constraints(calculations, dimensions, harmonic_index, transform, fc1):
    """Apply the common FC1-FC2 rule, with a zero FC1 block when absent."""
    calculation = calculations[harmonic_index]
    harmonic = build_harmonic_rotational_constraints(
        calculation.orbit_space,
        calculation.supercell,
        index=calculation.index,
    )
    left = sum(dimensions[:harmonic_index])
    right = sum(dimensions[harmonic_index + 1 :])
    harmonic = sparse.hstack(
        [
            sparse.csr_matrix((harmonic.shape[0], left)),
            harmonic,
            sparse.csr_matrix((harmonic.shape[0], right)),
        ],
        format="csr",
    )
    fc1_rotation = _fc1_rotation_matrix(calculations[0].index.n_primitive)
    return harmonic @ transform + fc1_rotation @ fc1


def _independent_constraint_rows(matrix, tolerance=1e-11):
    """Keep a numerically independent row basis before row normalization."""
    matrix = sparse.csr_matrix(matrix)
    if not matrix.shape[0] or not matrix.nnz:
        return matrix
    _q, triangular, permutation = qr(
        matrix.toarray().T,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(triangular))
    threshold = (
        max(tolerance, tolerance * max(matrix.shape) * float(diagonal.max()))
        if len(diagonal)
        else 0.0
    )
    rank = int(np.count_nonzero(diagonal > threshold))
    return matrix[np.sort(permutation[:rank])]


def _image_columns(calculation):
    """Yield cluster and its dense Cartesian-component-to-pivot map."""
    offset = 0
    for orbit in calculation.orbit_space.orbits:
        representative = np.linalg.solve(orbit.basis[orbit.pivots].T, orbit.basis.T).T
        for image in orbit.images:
            yield image.cluster, image.action.apply_columns(representative), offset
        offset += orbit.dimension


def _physical_image_groups(calculation):
    """Group orbit images exactly as sparse IFC materialization groups them."""
    groups = {}
    for cluster, columns, local_offset in _image_columns(calculation):
        first = int(cluster[0])
        key = (int(calculation.index.primitive[first]), *map(int, cluster[1:]))
        groups.setdefault(key, []).append((columns, local_offset))
    return groups


def _adjacent_rotational_constraints(lower, upper, dimensions, lower_index, tolerance=1e-12):
    lower_order = lower.config.order
    upper_order = upper.config.order
    if upper_order != lower_order + 1:
        raise ValueError("rotational constraints require consecutive IFC orders")
    lower_global = sum(dimensions[:lower_index])
    upper_global = lower_global + dimensions[lower_index]
    equations = {}
    entries = {}

    def row(key):
        return equations.setdefault(key, len(equations))

    # Lower-order Cartesian tensor rotation term.
    for cluster, images in _physical_image_groups(lower).items():
        weight = 1.0 / len(images)
        for columns, local_offset in images:
            shaped = columns.reshape((3,) * lower_order + (-1,))
            for components in np.ndindex((3,) * lower_order):
                for mu in range(3):
                    for nu in range(mu + 1, 3):
                        equation = row((cluster, components, mu, nu))
                        for axis in range(lower_order):
                            if components[axis] == mu:
                                changed = (*components[:axis], nu, *components[axis + 1 :])
                                _add(
                                    entries,
                                    equation,
                                    lower_global + local_offset,
                                    shaped[changed],
                                    weight,
                                    tolerance,
                                )
                            if components[axis] == nu:
                                changed = (*components[:axis], mu, *components[axis + 1 :])
                                _add(
                                    entries,
                                    equation,
                                    lower_global + local_offset,
                                    shaped[changed],
                                    -weight,
                                    tolerance,
                                )

    # Upper-order moment term, summed over its final atom index.
    positions = upper.supercell.positions
    geometry = PeriodicGeometry(upper.supercell.cell, upper.supercell.pbc)
    for cluster, images in _physical_image_groups(upper).items():
        prefix = cluster[:-1]
        origin = positions[upper.index.representative(prefix[0])]
        vector, _ = geometry.mic(positions[cluster[-1]] - origin)
        weight = 1.0 / len(images)
        for columns, local_offset in images:
            shaped = columns.reshape((3,) * upper_order + (-1,))
            for components in np.ndindex((3,) * lower_order):
                for mu in range(3):
                    for nu in range(mu + 1, 3):
                        equation = row((prefix, components, mu, nu))
                        _add(
                            entries,
                            equation,
                            upper_global + local_offset,
                            shaped[components + (nu,)],
                            weight * vector[mu],
                            tolerance,
                        )
                        _add(
                            entries,
                            equation,
                            upper_global + local_offset,
                            shaped[components + (mu,)],
                            -weight * vector[nu],
                            tolerance,
                        )
    return _entries_to_matrix(entries, len(equations), sum(dimensions))


def _highest_order_rotational_boundary(calculation, dimensions, tolerance=1e-12):
    order = calculation.config.order
    global_offset = sum(dimensions[:-1])
    equations = {}
    entries = {}
    for cluster, images in _physical_image_groups(calculation).items():
        weight = 1.0 / len(images)
        for columns, local_offset in images:
            shaped = columns.reshape((3,) * order + (-1,))
            for components in np.ndindex((3,) * order):
                for mu in range(3):
                    for nu in range(mu + 1, 3):
                        key = (cluster, components, mu, nu)
                        equation = equations.setdefault(key, len(equations))
                        for axis in range(order):
                            if components[axis] == mu:
                                changed = (*components[:axis], nu, *components[axis + 1 :])
                                _add(
                                    entries,
                                    equation,
                                    global_offset + local_offset,
                                    shaped[changed],
                                    weight,
                                    tolerance,
                                )
                            if components[axis] == nu:
                                changed = (*components[:axis], mu, *components[axis + 1 :])
                                _add(
                                    entries,
                                    equation,
                                    global_offset + local_offset,
                                    shaped[changed],
                                    -weight,
                                    tolerance,
                                )
    return _entries_to_matrix(entries, len(equations), sum(dimensions))


def _add(entries, row, offset, values, factor, tolerance):
    for local in np.flatnonzero(np.abs(values * factor) > tolerance):
        key = (row, offset + int(local))
        entries[key] = entries.get(key, 0.0) + float(factor * values[local])


def _entries_to_matrix(entries, rows, columns):
    filtered = [(key, value) for key, value in entries.items() if abs(value) > 1e-12]
    return sparse.coo_matrix(
        (
            [value for _, value in filtered],
            ([key[0] for key, _ in filtered], [key[1] for key, _ in filtered]),
        ),
        shape=(rows, columns),
    ).tocsr()


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
