"""Cartesian acoustic equations shared by independent postprocessors."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from mlfcs.force_constants.lattice import rotate_basis
from mlfcs.force_constants.model import ForceConstants


def constraint_matrix(model: ForceConstants, order: int) -> sparse.csr_matrix:
    space = model.space
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


def relative_residual(matrix: sparse.csr_matrix, values: np.ndarray) -> tuple[float, float]:
    if matrix.shape[0] == 0 or values.size == 0:
        return 0.0, 0.0
    residual = np.asarray(matrix @ values)
    maximum = float(np.max(np.abs(residual), initial=0.0))
    row_norms = np.asarray(np.abs(matrix).sum(axis=1)).reshape(-1)
    equation_scale = float(np.max(row_norms, initial=0.0))
    parameter_scale = float(np.max(np.abs(values), initial=0.0))
    scale = equation_scale * parameter_scale
    relative = maximum / scale if scale else 0.0
    return maximum, relative


__all__ = ["constraint_matrix", "relative_residual"]
