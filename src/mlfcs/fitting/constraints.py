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

    Taylor coordinates use these constraints directly in physical parameter
    space; no covariance-dependent constraint branch is required.

    Born--Huang and Huang conditions deliberately live in the explicit FC2
    postprocessor. Applying them here would couple fitted FC2 to higher orders.
    """
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    total = sum(dimensions)
    blocks: list[sparse.csr_matrix] = []
    if acoustic:
        for index, calculation in enumerate(calculations):
            primitive_space = getattr(calculation, "primitive_orbit_space", None)
            if primitive_space is None:
                primitive_space = calculation.interaction_space.primitive_orbit_space
            local = build_translational_constraints(primitive_space)
            left = sum(dimensions[:index])
            right = total - left - dimensions[index]
            blocks.append(
                sparse.hstack(
                    [
                        sparse.csr_matrix((local.shape[0], left)),
                        local,
                        sparse.csr_matrix((local.shape[0], right)),
                    ],
                    format="csr",
                )
            )
    matrix = sparse.vstack(blocks, format="csr") if blocks else sparse.csr_matrix((0, total))
    matrix = _compress_rows(matrix)
    return JointConstraints(matrix, sum(block.shape[0] for block in blocks))


def _compress_rows(matrix, tolerance=1e-12):
    """Drop empty rows, normalize, and remove numerically identical constraints.

    The constraint rows are Cartesian components of algebraic numbers, so "the same
    constraint" is an algebraic statement that no integer key can decide; it is settled
    here by rounding, and again downstream by the rank tests.  The threshold stays
    explicit for that reason, and rows that are exactly empty are dropped without it.
    """
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    matrix = matrix[norms > tolerance]
    norms = norms[norms > tolerance]
    if not norms.size:
        return matrix
    matrix = sparse.diags(1.0 / norms) @ matrix
    rounded = matrix.copy()
    rounded.data = np.round(rounded.data, 12)
    keep: list[int] = []
    seen: set[tuple[bytes, bytes]] = set()
    for row in range(rounded.shape[0]):
        begin, end = rounded.indptr[row : row + 2]
        indices = rounded.indices[begin:end]
        values = rounded.data[begin:end]
        if len(values) and values[0] < 0.0:
            values = -values
        key = (indices.tobytes(), values.tobytes())
        if key not in seen:
            seen.add(key)
            keep.append(row)
    return matrix[np.asarray(keep, dtype=np.int64)].tocsr()
