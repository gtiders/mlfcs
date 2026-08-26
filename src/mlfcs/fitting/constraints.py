"""Joint physical constraints assembled in fitting coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mlfcs.constraints.translational import build_translational_constraints


def _parameter_count(calculation):
    return sum(orbit.dimension for orbit in calculation.realized_orbit_space.orbits)


@dataclass(frozen=True, slots=True)
class JointConstraints:
    matrix: sparse.csr_matrix
    translational_rows: int


def build_joint_constraints(
    calculations,
    *,
    acoustic: bool,
) -> JointConstraints:
    """Build per-order translational constraints in the fitting coordinates.

    Taylor uses these constraints directly.  Wick lowering contracts pairs of
    Cartesian indices with the reference covariance.  Such contractions
    commute with the translational sum over every uncontracted site index, so
    an ASR-satisfying Wick tensor lowers to ASR-satisfying Taylor tensors.  The
    same per-order null spaces are therefore sufficient for both backends and
    no covariance-dependent constraint branch is required.

    Born--Huang and Huang conditions deliberately live in the explicit FC2
    postprocessor. Applying them here would couple fitted FC2 to higher orders.
    """
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    total = sum(dimensions)
    translational = []
    if acoustic:
        for index, calculation in enumerate(calculations):
            primitive_space = getattr(calculation, "primitive_orbit_space", None)
            if primitive_space is None:
                primitive_space = calculation.interaction_space.primitive_orbit_space
            local = build_translational_constraints(primitive_space)
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
