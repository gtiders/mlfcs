"""Translational-invariance constraints for symmetry-reduced force constants.

For a force-constant tensor of order ``p``, the acoustic sum rule requires the
sum over its final atomic position and periodic images to vanish while the
other positions and Cartesian directions remain fixed. This module expresses
that condition in cluster-space coordinates and provides coordinates for the
complete real subspace of force constants that satisfy it.

Orbit-local maps relate lattice tensor coefficients to the canonical
Cartesian parameters. Sparse, exactly certified triangular equations define
the nullspace without storing a dense global basis.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import sparse

from mlfcs.cluster_space.acoustic_factorization import (
    checked_add,
    factor_acoustic_equations,
    lift_lattice_coordinates,
    restrict_lattice_rows,
)
from mlfcs.foundation.arrays import require_allocation
from mlfcs.foundation.tensors import apply_tensor_action, validate_tensor_action


@njit(cache=True)
def _compress(rows, columns, values):
    """Combine repeated integer contributions to one acoustic equation."""
    count = 0
    for i in range(len(values)):
        if count and rows[count - 1] == rows[i] and columns[count - 1] == columns[i]:
            values[count - 1] = checked_add(values[count - 1], values[i])
        else:
            rows[count], columns[count], values[count] = rows[i], columns[i], values[i]
            count += 1
    return rows[:count], columns[:count], values[:count]


def lattice_acoustic_equations(
    space, order: int
) -> tuple[sparse.csr_matrix, tuple[np.ndarray, ...]]:
    """Express one order's acoustic sum rule in lattice tensor coordinates.

    Each equation sums over the final atomic position and its periodic images,
    holding the preceding positions and Cartesian tensor directions fixed.
    ``matrix`` acts on lattice tensor coefficients ``c``; the accompanying
    orbit maps ``W`` convert them to canonical Cartesian parameters
    ``theta = W c``. The equation and mapped parameters use the same orbit
    order as the cluster space. The Cartesian conversion follows the
    cluster-space convention that lattice vectors are rows of the cell matrix.
    """
    block = space.block(order)
    width = block.parameters.stop - block.parameters.start
    prefixes, rows, columns, values, maps = {}, [], [], [], []
    frame = np.ones((1, 1))
    for _ in range(order):
        frame = np.kron(frame, space.cell.T)
    map_cache = {}
    offset, dimension = 0, 3**order
    for orbit in space.orbits[block.orbits]:
        key = (
            orbit.lattice_basis.shape,
            orbit.lattice_basis.tobytes(),
            orbit.observation_rows.tobytes(),
        )
        # Equal bases and observation rows define the same physical coordinate map.
        if key not in map_cache:
            map_cache[key] = (frame @ orbit.lattice_basis)[orbit.observation_rows]
            map_cache[key].setflags(write=False)
        maps.append(map_cache[key])
        magnitude = max((abs(int(v)) for v in orbit.lattice_basis.flat), default=0)
        for image, cluster in enumerate(orbit.clusters):
            rotation = space.symmetry.rotations[orbit.operations[image]]
            validate_tensor_action(rotation, order, coefficient=magnitude)
            action = apply_tensor_action(orbit.lattice_basis, rotation, orbit.permutations[image])
            prefix = prefixes.setdefault(cluster.labels[:-1], len(prefixes))
            component, column = np.nonzero(action)
            rows.append(prefix * dimension + component)
            columns.append(offset + column)
            values.append(action[component, column])
        offset += orbit.dimension
    rows, columns, values = (
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for chunks in (rows, columns, values)
    )
    sorting = np.lexsort((columns, rows))
    rows, columns, values = _compress(rows[sorting], columns[sorting], values[sorting])
    mask = values != 0
    matrix = sparse.coo_matrix(
        (values[mask], (rows[mask], columns[mask])), shape=(len(prefixes) * dimension, width)
    ).tocsr()
    return matrix, tuple(maps)


@dataclass(frozen=True, slots=True)
class AcousticCoordinates:
    """Coordinates for force constants satisfying the acoustic sum rule.

    For one force-constant order, the physical Cartesian coefficients are
    ``theta = W c``, where ``c`` are lattice tensor coefficients. The acoustic
    equations constrain ``c``; choosing its nonpivot entries freely determines
    a unique point in the complete real nullspace. ``lift`` maps those free
    entries back to the canonical physical coefficient vector.

    Exact sparse factors certify the nullspace dimension. These coordinates
    describe a real linear subspace; they are not a saturated integer basis
    for the combined global force-constant lattice. ``adjoint`` and
    ``restrict_rows`` apply the corresponding physical lift to observations.
    """

    indptr: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    pivots: np.ndarray
    free: np.ndarray
    physical_maps: tuple[np.ndarray, ...]
    width: int

    @property
    def dimension(self) -> int:
        """Number of independent force-constant parameters after imposing ASR."""
        return len(self.free)

    def lift(self, free_coordinates: object) -> np.ndarray:
        """Map free lattice coordinates to physical Cartesian coefficients.

        The returned order-local coefficient vector satisfies the acoustic
        equations by construction. ``free_coordinates`` contains the chosen
        nonpivot lattice coefficients, not canonical Cartesian parameters.
        """
        free_coordinates = np.asarray(free_coordinates, dtype=float)
        if free_coordinates.shape != (self.dimension,) or not np.all(np.isfinite(free_coordinates)):
            raise ValueError("invalid acoustic free coordinates")
        coordinates = lift_lattice_coordinates(
            free_coordinates,
            self.width,
            self.free,
            self.pivots,
            self.indptr,
            self.indices,
            self.values,
        )
        offset = 0
        for block in self.physical_maps:
            stop = offset + len(block)
            coordinates[offset:stop] = block @ coordinates[offset:stop]
            offset = stop
        if not np.all(np.isfinite(coordinates)):
            raise ArithmeticError("acoustic triangular lift is not finite")
        return coordinates

    def normal_basis(self) -> np.ndarray:
        """Return physical parameter directions perpendicular to the ASR subspace.

        The force-constant parameters satisfying ASR form a linear subspace in
        the canonical Cartesian parameter metric. The returned columns span
        its orthogonal complement, which lets rotational corrections remain
        inside the ASR subspace while minimizing changes to physical
        parameters.

        If ``U c = 0`` are the certified lattice equations and ``theta = W c``,
        then ``U W^-1`` spans the physical normal directions. QR orthonormalizes
        these directions using the certified equation rank; it does not infer
        rank from floating-point singular values. The basis is built on demand
        and contains one column per certified independent acoustic constraint.
        """
        rank = len(self.pivots)
        require_allocation("acoustic normal basis", (self.width, rank))
        rows = np.zeros((rank, self.width))
        for row in range(rank):
            start, stop = self.indptr[row : row + 2]
            rows[row, self.indices[start:stop]] = self.values[start:stop]
        # Row rescaling leaves the normal subspace unchanged.
        rows /= np.max(np.abs(rows), axis=1, initial=1.0)[:, None]
        offset = 0
        for block in self.physical_maps:
            stop = offset + len(block)
            rows[:, offset:stop] = np.linalg.solve(block.T, rows[:, offset:stop].T).T
            offset = stop
        if not np.all(np.isfinite(rows)):
            raise ArithmeticError("acoustic physical normal directions are not finite")
        return np.linalg.qr(rows.T, mode="reduced")[0]

    def extract(self, coefficients: object) -> np.ndarray:
        """Recover free coordinates from physical coefficients that satisfy ASR.

        Coefficients use the order-local canonical Cartesian parameter layout.
        A vector outside the prepared acoustic subspace raises ``ValueError``;
        this operation does not project an arbitrary force-constant model.
        """
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.width,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("invalid canonical coefficient vector")
        coordinates = np.empty(self.width)
        offset = 0
        for block in self.physical_maps:
            stop = offset + len(block)
            coordinates[offset:stop] = np.linalg.solve(block, coefficients[offset:stop])
            offset = stop
        free = coordinates[self.free]
        restored = self.lift(free)
        scale = max(1.0, float(np.max(np.abs(coefficients), initial=0.0)))
        if np.max(np.abs(restored - coefficients), initial=0.0) > 1e-10 * scale:
            raise ValueError("model coefficients do not belong to the prepared acoustic subspace")
        return free

    def adjoint(self, vector: object) -> np.ndarray:
        """Map a physical parameter vector through the adjoint of ``lift``."""
        return self.restrict_rows(np.asarray(vector).reshape(1, -1))[0]

    def restrict_rows(self, physical_rows: object) -> np.ndarray:
        """Express physical force observations in the free acoustic coordinates.

        Each row is composed with the physical lift, so predictions are
        unchanged while the columns correspond only to force constants that
        satisfy the acoustic sum rule.
        """
        physical_rows = np.asarray(physical_rows, dtype=float)
        if physical_rows.ndim != 2 or physical_rows.shape[1] != self.width:
            raise ValueError("invalid acoustic observation rows")
        rows = np.empty_like(physical_rows)
        offset = 0
        for block in self.physical_maps:
            stop = offset + len(block)
            rows[:, offset:stop] = physical_rows[:, offset:stop] @ block
            offset = stop
        result = restrict_lattice_rows(
            rows, self.free, self.pivots, self.indptr, self.indices, self.values
        )
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("acoustic row transformation is not finite")
        return result


def _pack(upper, pivots, maps, width):
    """Mark nonpivot lattice coefficients as free and attach their physical maps."""
    free_mask = np.ones(width, dtype=bool)
    free_mask[pivots] = False
    free = np.flatnonzero(free_mask)
    arrays = (*upper, pivots, free)
    for array in (*arrays, *maps):
        array.setflags(write=False)
    return AcousticCoordinates(*arrays, maps, width)


def prepare_coordinates(
    matrix: sparse.csr_matrix, physical_maps: tuple[np.ndarray, ...]
) -> AcousticCoordinates:
    """Build the physical acoustic subspace from its lattice equations.

    ``matrix`` expresses the translational-invariance conditions in lattice
    tensor coordinates. ``physical_maps`` converts each orbit's lattice
    coefficients to canonical Cartesian parameters. The returned coordinates
    cover the complete real nullspace while retaining the original physical
    coefficient layout.
    """
    upper, pivots = factor_acoustic_equations(matrix)
    return _pack(upper, pivots, physical_maps, matrix.shape[1])


def prepare_acoustic_coordinates(space):
    """Prepare the acoustic-invariant force-constant subspace for every order.

    Each included tensor order receives its own free coordinates and retains
    the cluster space's canonical Cartesian parameter layout. Sparse exact
    factorization certifies each subspace; unsupported reconstruction or
    certificate arithmetic raises ``OverflowError``.
    """
    return tuple(
        prepare_coordinates(*lattice_acoustic_equations(space, order)) for order in space.orders
    )
