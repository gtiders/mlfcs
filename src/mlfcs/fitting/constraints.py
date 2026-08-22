from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import factorial

import numpy as np
from ase.geometry import find_mic
from scipy import sparse

from mlfcs.reconstruction.asr import build_translational_constraints


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
    """Build order-local ASR and adjacent-order Cartesian rotation constraints.

    ``rotational_mode=2`` follows ALAMODE ICONST=2 and omits the maximum-order
    boundary. ``rotational_mode=3`` additionally constrains the maximum order
    against a zero next-order tensor.
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
                    [sparse.csr_matrix((local.shape[0], left)), local,
                     sparse.csr_matrix((local.shape[0], right))],
                    format="csr",
                )
            )
    rotational = []
    if rotational_mode:
        for index, (lower, upper) in enumerate(pairwise(calculations)):
            rotational.append(
                _adjacent_rotational_constraints(lower, upper, dimensions, index)
            )
        if rotational_mode == 3:
            rotational.append(_highest_order_rotational_boundary(calculations[-1], dimensions))
    matrices = translational + rotational
    matrix = sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, total))
    if rotational_mode:
        if covariance is None:
            raise ValueError("covariance is required for rotational constraints in the Wick basis")
        matrix = matrix @ build_wick_to_taylor_transform(calculations, covariance)
    matrix = _compress_rows(matrix)
    return JointConstraints(
        matrix,
        sum(item.shape[0] for item in translational),
        sum(item.shape[0] for item in rotational),
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
    by_order = {calculation.config.order: (index, calculation) for index, calculation in enumerate(calculations)}
    image_maps = {
        order: {
            cluster: (int(offsets[index] + local_offset), columns)
            for cluster, columns, local_offset in _image_columns(calculation)
        }
        for order, (index, calculation) in by_order.items()
    }

    for target_order in by_order:
        for source_order in range(target_order + 2, max(by_order, default=0) + 1, 2):
            if source_order not in by_order:
                continue
            source_index, source = by_order[source_order]
            pairs = (source_order - target_order) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            contracted_by_target: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
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
                if target_cluster not in image_maps[target_order]:
                    raise ValueError(
                        f"Wick-to-Taylor contraction creates FC{target_order} cluster "
                        f"{target_cluster} outside its configured support"
                    )
                source_offset = int(offsets[source_index] + local_offset)
                contributions = contracted_by_target.setdefault(target_cluster, {})
                contribution = contracted.reshape(3**target_order, -1)
                contributions[source_offset] = contributions.get(
                    source_offset, np.zeros_like(contribution)
                ) + contribution
            for target_cluster, contributions in contracted_by_target.items():
                target_offset, target_columns = image_maps[target_order][target_cluster]
                source_offsets = sorted(contributions)
                contracted = np.concatenate(
                    [contributions[source_offset] for source_offset in source_offsets], axis=1
                )
                mapping = np.linalg.lstsq(target_columns, contracted, rcond=None)[0]
                begin = 0
                for source_offset in source_offsets:
                    width = contributions[source_offset].shape[1]
                    transform[
                        target_offset : target_offset + mapping.shape[0],
                        source_offset : source_offset + width,
                    ] += coefficient * mapping[:, begin : begin + width]
                    begin += width
    return transform.tocsr()


def _parameter_count(calculation):
    return sum(orbit.dimension for orbit in calculation.orbit_space.orbits)


def _image_columns(calculation):
    """Yield cluster and its dense Cartesian-component-to-pivot map."""
    offset = 0
    for orbit in calculation.orbit_space.orbits:
        representative = np.linalg.solve(
            orbit.basis[orbit.pivots].T, orbit.basis.T
        ).T
        for image in orbit.images:
            yield image.cluster, image.action.apply_columns(representative), offset
        offset += orbit.dimension


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
    for cluster, columns, local_offset in _image_columns(lower):
        shaped = columns.reshape((3,) * lower_order + (-1,))
        for components in np.ndindex((3,) * lower_order):
            for mu in range(3):
                for nu in range(mu + 1, 3):
                    equation = row((cluster, components, mu, nu))
                    for axis in range(lower_order):
                        if components[axis] == mu:
                            changed = (*components[:axis], nu, *components[axis + 1 :])
                            _add(entries, equation, lower_global + local_offset, shaped[changed], 1.0, tolerance)
                        if components[axis] == nu:
                            changed = (*components[:axis], mu, *components[axis + 1 :])
                            _add(entries, equation, lower_global + local_offset, shaped[changed], -1.0, tolerance)

    # Upper-order moment term, summed over its final atom index.
    positions = upper.supercell.positions
    for cluster, columns, local_offset in _image_columns(upper):
        prefix = cluster[:-1]
        origin = positions[prefix[0]]
        vector, _ = find_mic(
            positions[cluster[-1]] - origin,
            upper.supercell.cell,
            upper.supercell.pbc,
        )
        shaped = columns.reshape((3,) * upper_order + (-1,))
        for components in np.ndindex((3,) * lower_order):
            for mu in range(3):
                for nu in range(mu + 1, 3):
                    equation = row((prefix, components, mu, nu))
                    _add(entries, equation, upper_global + local_offset, shaped[components + (nu,)], vector[mu], tolerance)
                    _add(entries, equation, upper_global + local_offset, shaped[components + (mu,)], -vector[nu], tolerance)
    return _entries_to_matrix(entries, len(equations), sum(dimensions))


def _highest_order_rotational_boundary(calculation, dimensions, tolerance=1e-12):
    order = calculation.config.order
    global_offset = sum(dimensions[:-1])
    equations = {}
    entries = {}
    for cluster, columns, local_offset in _image_columns(calculation):
        shaped = columns.reshape((3,) * order + (-1,))
        for components in np.ndindex((3,) * order):
            for mu in range(3):
                for nu in range(mu + 1, 3):
                    key = (cluster, components, mu, nu)
                    equation = equations.setdefault(key, len(equations))
                    for axis in range(order):
                        if components[axis] == mu:
                            changed = (*components[:axis], nu, *components[axis + 1 :])
                            _add(entries, equation, global_offset + local_offset, shaped[changed], 1.0, tolerance)
                        if components[axis] == nu:
                            changed = (*components[:axis], mu, *components[axis + 1 :])
                            _add(entries, equation, global_offset + local_offset, shaped[changed], -1.0, tolerance)
    return _entries_to_matrix(entries, len(equations), sum(dimensions))


def _add(entries, row, offset, values, factor, tolerance):
    for local in np.flatnonzero(np.abs(values * factor) > tolerance):
        key = (row, offset + int(local))
        entries[key] = entries.get(key, 0.0) + float(factor * values[local])


def _entries_to_matrix(entries, rows, columns):
    filtered = [(key, value) for key, value in entries.items() if abs(value) > 1e-12]
    return sparse.coo_matrix(
        ([value for _, value in filtered],
         ([key[0] for key, _ in filtered], [key[1] for key, _ in filtered])),
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
