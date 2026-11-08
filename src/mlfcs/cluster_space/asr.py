"""Acoustic-sum-rule constraints and their exact acoustic coordinates.

For a force-constant tensor of order ``p``, the acoustic sum rule requires the
sum over its final atomic position and periodic images to vanish while the
other positions and Cartesian directions remain fixed. This module expresses
that condition in cluster-space lattice coordinates and provides coordinates
for the complete real subspace of force constants satisfying it.

The construction has two exact stages. ``lattice_acoustic_equations`` assembles
the sparse integer system ``U c = 0`` on one force-constant order, together with
the per-orbit maps ``W`` that convert lattice tensor coefficients into canonical
physical parameters, ``theta = W c``. ``factor_acoustic_equations`` then returns
a certified sparse triangular factor of ``U``: modular elimination over two
primes proposes rational factors, and a cleared-denominator integer identity
together with independent pivot witnesses proves their rank and nullspace.
``AcousticCoordinates`` solves with that factor in floating point, so fitting and
observation rows never build a dense global basis.

The integer certificate is what makes the result trustworthy: equation rank is
never inferred from floating-point singular values. Lattice coordinates are used
for the certificate because they keep the equations integral; the physical
parameter coordinates follow the cluster-space convention that lattice vectors
are rows of the cell matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from numba import njit, types
from numba.typed import Dict, List
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr

from mlfcs.cluster_space.integer_kernel import reconstruct_rationals, scale_to_common_denominator
from mlfcs.foundation.arrays import require_allocation
from mlfcs.foundation.integer import RANK_PRIMES, modular_power
from mlfcs.foundation.tensors import apply_tensor_action, validate_tensor_action

LIMIT = np.int64(9223372036854775807)


@njit(cache=True)
def checked_add(left, right):
    """Add two certificate terms, raising instead of wrapping the int64 result."""
    if right > 0 and left > LIMIT - right:
        raise OverflowError("ASR certificate sum exceeds int64")
    if right < 0 and left < -LIMIT - right:
        raise OverflowError("ASR certificate sum exceeds int64")
    return left + right


@njit(cache=True)
def checked_multiply(left, right):
    """Multiply two certificate terms within the supported symmetric int64 range."""
    if left == -LIMIT - 1 or right == -LIMIT - 1:
        raise OverflowError("ASR certificate excludes int64 minimum")
    if right != 0 and abs(left) > LIMIT // abs(right):
        raise OverflowError("ASR certificate product exceeds int64")
    return left * right


@njit(cache=True)
def _factor(indptr, indices, values, width, prime, schedule):
    """Propose sparse triangular factors of the acoustic equations over one prime.

    The factors satisfy ``A = L U`` modulo ``prime``. Pivot rows are chosen from
    a shared schedule so factorizations over different primes describe the same
    chart, which is what makes rational reconstruction possible. Residue
    products stay inside the signed int64 range because ``prime**2 < 2**63``.

    ``schedule`` is filled in place: ``-2`` marks an unassigned row, later ``-1``
    marks a dependent row. Return ``(upper, lower, pivots, ok)``; ``ok`` is
    false when an unassigned row cannot realize its scheduled pivot.
    """
    upper, lower = List(), List()
    pivot_rows = np.full(width, -1, dtype=np.int64)
    pivots = List()
    counts = np.zeros(width, dtype=np.int64)
    for column in indices:
        counts[column] += 1
    row_order = np.argsort(indptr[1:] - indptr[:-1])
    for _ in range(len(indptr) - 1):
        lower.append(Dict.empty(types.int64, types.int64))
    for original in row_order:
        row = Dict.empty(types.int64, types.int64)
        for entry in range(indptr[original], indptr[original + 1]):
            residue = values[entry] % prime
            if residue:
                row[indices[entry]] = residue
        while True:
            previous, pivot = width, -1
            for column in row:
                rank = pivot_rows[column]
                if rank >= 0 and rank < previous:
                    previous, pivot = rank, column
            if pivot < 0:
                break
            multiplier = row[pivot]
            lower[original][previous] = multiplier
            for column, coefficient in upper[previous].items():
                updated = (row.get(column, np.int64(0)) - multiplier * coefficient % prime) % prime
                if updated:
                    row[column] = updated
                elif column in row:
                    del row[column]
        if not row:
            if schedule[original] >= 0:
                return upper, lower, np.asarray(pivots), False
            continue
        pivot = schedule[original]
        if pivot == -2:
            pivot, incidence = -1, len(indices) + 1
            for column in row:
                if counts[column] < incidence or (counts[column] == incidence and column < pivot):
                    pivot, incidence = column, counts[column]
            schedule[original] = pivot
        if pivot not in row:
            return upper, lower, np.asarray(pivots), False
        diagonal = row[pivot]
        lower[original][len(upper)] = diagonal
        inverse = modular_power(diagonal, prime - 2, prime)
        for column in row:
            row[column] = row[column] * inverse % prime
        pivot_rows[pivot] = len(upper)
        pivots.append(pivot)
        upper.append(row)
    for original in range(len(schedule)):
        if schedule[original] == -2:
            schedule[original] = -1
    return upper, lower, np.asarray(pivots), True


@njit(cache=True)
def _pack_rows(rows):
    """Pack sparse factor rows into CSR arrays, sorting each row's columns."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i in range(len(rows)):
        indptr[i + 1] = indptr[i] + len(rows[i])
    indices = np.empty(indptr[-1], dtype=np.int64)
    values = np.empty(indptr[-1], dtype=np.int64)
    for i in range(len(rows)):
        columns = np.empty(len(rows[i]), dtype=np.int64)
        for j, column in enumerate(rows[i]):
            columns[j] = column
        columns.sort()
        for j, column in enumerate(columns):
            indices[indptr[i] + j] = column
            values[indptr[i] + j] = rows[i][column]
    return indptr, indices, values


@njit(cache=True)
def _merge_structure(first, second):
    """Return the union of two CSR factor supports column by column.

    Residues are reconstructed from separate prime fields, so a coefficient that
    vanishes modulo one prime still has to be reconstructed as zero rather than
    dropped from the sparsity pattern.
    """
    left_ptr, left_col, _left_values = first
    right_ptr, right_col, _right_values = second
    indptr = np.zeros(len(left_ptr), dtype=np.int64)
    for row in range(len(indptr) - 1):
        left, right = left_ptr[row], right_ptr[row]
        count = 0
        while left < left_ptr[row + 1] or right < right_ptr[row + 1]:
            if right == right_ptr[row + 1] or (
                left < left_ptr[row + 1] and left_col[left] < right_col[right]
            ):
                left += 1
            elif left == left_ptr[row + 1] or right_col[right] < left_col[left]:
                right += 1
            else:
                left += 1
                right += 1
            count += 1
        indptr[row + 1] = indptr[row] + count
    indices = np.empty(indptr[-1], dtype=np.int64)
    for row in range(len(indptr) - 1):
        left, right, output = left_ptr[row], right_ptr[row], indptr[row]
        while left < left_ptr[row + 1] or right < right_ptr[row + 1]:
            if right == right_ptr[row + 1] or (
                left < left_ptr[row + 1] and left_col[left] < right_col[right]
            ):
                indices[output] = left_col[left]
                left += 1
            elif left == left_ptr[row + 1] or right_col[right] < left_col[left]:
                indices[output] = right_col[right]
                right += 1
            else:
                indices[output] = left_col[left]
                left += 1
                right += 1
            output += 1
    return indptr, indices


@njit(cache=True)
def _reconstruct_row(first, second, row, bound):
    """Reconstruct one rational factor row from its two modular residues.

    The common-denominator form ``(columns, coefficients, denominator)`` keeps
    the later integer identity free of divisions. Return values are only valid
    while every reconstructed numerator and denominator stays below ``bound``.
    """
    left, right = first[0][row], second[0][row]
    left_stop, right_stop = first[0][row + 1], second[0][row + 1]
    maximum = left_stop - left + right_stop - right
    columns = np.empty(maximum, dtype=np.int64)
    left_values = np.zeros((1, maximum), dtype=np.int64)
    right_values = np.zeros_like(left_values)
    count = 0
    while left < left_stop or right < right_stop:
        if right == right_stop or (left < left_stop and first[1][left] < second[1][right]):
            columns[count] = first[1][left]
            left_values[0, count] = first[2][left]
            left += 1
        elif left == left_stop or second[1][right] < first[1][left]:
            columns[count] = second[1][right]
            right_values[0, count] = second[2][right]
            right += 1
        else:
            columns[count] = first[1][left]
            left_values[0, count] = first[2][left]
            right_values[0, count] = second[2][right]
            left += 1
            right += 1
        count += 1
    numerator, denominator, ok = reconstruct_rationals(
        left_values[:, :count], right_values[:, :count], bound
    )
    if not ok:
        raise OverflowError("ASR rational factor reconstruction failed")
    scaled, common, ok = scale_to_common_denominator(numerator, denominator)
    if not ok:
        raise OverflowError("ASR rational factor denominator exceeds word domain")
    return columns[:count], scaled[0], common


@njit(cache=True)
def _reconstruct_rows(first, second, bound):
    """Reconstruct the rational upper factor one denominator-bounded row at a time."""
    indptr, indices = _merge_structure(first, second)
    values = np.empty(len(indices), dtype=np.int64)
    denominators = np.empty(len(indptr) - 1, dtype=np.int64)
    for row in range(len(denominators)):
        _columns, coefficients, denominator = _reconstruct_row(first, second, row, bound)
        values[indptr[row] : indptr[row + 1]] = coefficients
        denominators[row] = denominator
    return (indptr, indices, values), denominators


@njit(cache=True)
def _certify(indptr, indices, values, upper, uden, first_lower, second_lower, pivots, bound):
    """Certify the reconstructed factorization and nullspace over the integers.

    A cleared-denominator identity verifies ``A = L U`` exactly. Independent
    pivot columns establish that ``U`` has full row rank, while rows of ``L``
    with distinct final pivots establish that ``L`` has full column rank.
    Together these prove ``rank(A)`` and the dimension of its real nullspace.

    ``bound`` is the reconstruction window; every checked product must stay in
    the int64 domain, so an exhausted window is reported as a failed
    certificate rather than a wrapped identity.
    """
    uptr, ucol, uvalue = upper
    pivot_ranks = np.full(
        max(np.max(indices) if len(indices) else -1, np.max(pivots) if len(pivots) else -1) + 1,
        -1,
        dtype=np.int64,
    )
    for rank, pivot in enumerate(pivots):
        pivot_ranks[pivot] = rank
    for rank in range(len(pivots)):
        found = False
        for entry in range(uptr[rank], uptr[rank + 1]):
            column = ucol[entry]
            if column == pivots[rank]:
                found = uvalue[entry] == uden[rank]
            previous = pivot_ranks[column]
            if uvalue[entry] and previous >= 0 and previous < rank:
                return False
        if not found:
            return False
    seen = np.zeros(len(pivots), dtype=np.uint8)
    for original in range(len(indptr) - 1):
        lcol, lvalue, lower_denominator = _reconstruct_row(
            first_lower, second_lower, original, bound
        )
        greatest, common = -1, np.int64(1)
        for entry in range(len(lvalue)):
            if lvalue[entry]:
                rank = lcol[entry]
                greatest = max(greatest, rank)
                denominator = uden[rank]
                common = checked_multiply(common // gcd(common, denominator), denominator)
        if greatest >= 0:
            seen[greatest] = 1
        total_denominator = checked_multiply(common, lower_denominator)
        residual = Dict.empty(types.int64, types.int64)
        for lentry in range(len(lvalue)):
            rank, multiplier = lcol[lentry], lvalue[lentry]
            if not multiplier:
                continue
            scale = common // uden[rank]
            for uentry in range(uptr[rank], uptr[rank + 1]):
                column, value = ucol[uentry], uvalue[uentry]
                contribution = checked_multiply(checked_multiply(multiplier, value), scale)
                residual[column] = checked_add(residual.get(column, np.int64(0)), contribution)
        for entry in range(indptr[original], indptr[original + 1]):
            column = indices[entry]
            residual[column] = checked_add(
                residual.get(column, np.int64(0)),
                -checked_multiply(values[entry], total_denominator),
            )
        for value in residual.values():
            if value != 0:
                return False
    return np.all(seen)


@njit(cache=True)
def lift_lattice_coordinates(free_values, width, free, pivots, indptr, indices, values):
    """Solve the triangular factor for the nullspace vector fixed by free entries.

    ``free_values`` is assigned to the non-pivot columns and the pivot columns
    follow by back substitution, so the result satisfies ``U x = 0`` up to
    rounding. Return values are float64: the certificate is exact, the solve is
    not.
    """
    coordinates = np.zeros(width)
    coordinates[free] = free_values
    for rank in range(len(pivots) - 1, -1, -1):
        pivot, total, diagonal = pivots[rank], 0.0, 0.0
        for entry in range(indptr[rank], indptr[rank + 1]):
            column, value = indices[entry], values[entry]
            if column == pivot:
                diagonal = value
            else:
                total += float(value) * coordinates[column]
        coordinates[pivot] = -total / diagonal
    return coordinates


@njit(cache=True)
def restrict_lattice_rows(rows, free, pivots, indptr, indices, values):
    """Apply the adjoint triangular solve to observation rows.

    Each row is eliminated in place and the free columns are copied out, so the
    caller must pass storage it owns. Predictions are unchanged while the
    columns correspond only to force constants that satisfy the sum rule.
    """
    result = np.empty((len(rows), len(free)))
    for i in range(len(rows)):
        for rank, pivot in enumerate(pivots):
            diagonal = 0.0
            for entry in range(indptr[rank], indptr[rank + 1]):
                if indices[entry] == pivot:
                    diagonal = values[entry]
                    break
            multiplier = rows[i, pivot] / diagonal
            for entry in range(indptr[rank], indptr[rank + 1]):
                column = indices[entry]
                if column != pivot:
                    rows[i, column] -= multiplier * float(values[entry])
        for j, column in enumerate(free):
            result[i, j] = rows[i, column]
    return result


@njit(cache=True)
def _compress(rows, columns, values):
    """Sum adjacent duplicate entries of a lexicographically sorted triple list.

    The sum is a checked addition, so an overflowing acoustic equation is
    reported instead of silently wrapping.
    """
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
    holding the preceding positions and lattice tensor components fixed. A row
    is addressed by ``prefix * 3**order + component``, where ``prefix`` numbers
    the distinct tuples of labels of the first ``order - 1`` tensor slots, and
    the column is the orbit-local lattice coefficient. Equations for equal
    prefixes therefore share one physical sum rule.

    The common invertible tensor frame makes these equations equivalent to the
    Cartesian acoustic sum rule, while keeping every entry integral for the
    certificate. ``matrix`` acts on lattice tensor coefficients ``c``; the
    accompanying per-orbit maps ``W`` convert them into canonical Cartesian
    parameters ``theta = W c``. Both use the cluster-space orbit order, and the
    Cartesian conversion follows the cluster-space convention that lattice
    vectors are rows of the cell matrix. The row count includes every tensor
    component, so the system carries redundant equations on purpose: rank is
    established later by the exact certificate.

    Parameters
    ----------
    space
        Cluster space supplying the orbits, symmetry operations and cell.
    order
        Force-constant order to constrain. It must be present in ``space``.

    Returns
    -------
    matrix
        ``int64`` CSR matrix of shape ``(n_prefix * 3**order,
        block.parameters.stop - block.parameters.start)``.
    maps
        Tuple of ``(orbit.dimension, orbit.dimension)`` float64 matrices, one
        per orbit in cluster-space order, converting that orbit's lattice
        coefficients to canonical Cartesian parameters.

    Raises
    ------
    KeyError
        If the order is not part of the cluster space.
    OverflowError
        If an entry sum or a tensor-action bound exceeds the int64 domain.
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
        for image in range(len(orbit.labels)):
            rotation = space.symmetry.rotations[orbit.operations[image]]
            validate_tensor_action(rotation, order, coefficient=magnitude)
            action = apply_tensor_action(orbit.lattice_basis, rotation, orbit.permutations[image])
            prefix = prefixes.setdefault(
                tuple(map(tuple, orbit.labels[image, :-1].tolist())), len(prefixes)
            )
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


def factor_acoustic_equations(matrix):
    """Construct a certified triangular representation of acoustic equations.

    For an integer system ``A x = 0`` of acoustic sum rules, compute a sparse
    upper factor ``U`` with the same nullspace over the reals and the pivot
    columns that define its triangular solve. Modular elimination over two
    primes proposes the factor, then a cleared-denominator identity ``A = L U``
    and independent pivot witnesses certify ``rank(A)`` and therefore the
    nullspace dimension. The rank is never inferred from floating-point
    singular values.

    Parameters
    ----------
    matrix
        ``int64`` CSR matrix, as returned by ``lattice_acoustic_equations``,
        with shape ``(m, n)``. Entries must stay in the symmetric int64 domain.

    Returns
    -------
    upper
        ``(indptr, indices, values)`` arrays of the upper factor: row ``k``
        holds the cleared-denominator equation whose pivot is ``pivots[k]``.
    pivots
        ``int64`` column index of each factor row, in elimination order.

    Raises
    ------
    OverflowError
        If denominator reconstruction or the exact certificate exceeds the
        supported int64 range for every reconstruction window.
    ArithmeticError
        If the fixed pivot schedule is inconsistent between the reconstruction
        primes, which indicates a badly conditioned equation system rather
        than an arithmetic limit.
    """
    schedule = np.full(matrix.shape[0], -2, dtype=np.int64)
    indptr = np.asarray(matrix.indptr, dtype=np.int64)
    indices = np.asarray(matrix.indices, dtype=np.int64)
    choices = []
    for prime in RANK_PRIMES:
        upper, lower, pivots, ok = _factor(
            indptr, indices, matrix.data, matrix.shape[1], np.int64(prime), schedule
        )
        if not ok:
            raise ArithmeticError("ASR fixed pivot schedule is singular at a reconstruction prime")
        choices.append((_pack_rows(upper), _pack_rows(lower), pivots))
        # Hash tables are necessary for elimination, but not for stored residues.
        del upper, lower
    if not np.array_equal(choices[0][2], choices[1][2]):
        raise ArithmeticError("ASR fixed pivot schedules disagree")
    pivots = choices[0][2]
    for exponent in range(31):
        try:
            upper, uden = _reconstruct_rows(choices[0][0], choices[1][0], 1 << exponent)
            valid = _certify(
                indptr,
                indices,
                matrix.data,
                upper,
                uden,
                choices[0][1],
                choices[1][1],
                pivots,
                1 << exponent,
            )
        except OverflowError:
            continue
        if valid:
            return upper, pivots
    raise OverflowError("ASR factor reconstruction or exact identity exceeds int64 domain")


def _finite_vector(values, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Return ``values`` as a finite float64 array with the required shape.

    Raise ``ValueError`` naming the expected shape when it differs, or the
    presence of non-finite entries, so callers see which contract failed.
    """
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}: got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite: got non-finite entries")
    return array


@dataclass(frozen=True, slots=True)
class AcousticCoordinates:
    """Coordinates of the force-constant subspace satisfying the acoustic sum rule.

    For one force-constant order the physical Cartesian coefficients are
    ``theta = W c``, where ``c`` are lattice tensor coefficients. The acoustic
    equations constrain ``c``; choosing its nonpivot entries freely determines a
    unique point of the complete real nullspace, and ``lift`` maps those free
    entries back to the canonical physical coefficient vector.

    ``indptr``, ``indices`` and ``values`` store the certified triangular factor,
    ``pivots`` its pivot columns and ``free`` the remaining columns. These
    coordinates describe a real linear subspace; they are not a saturated
    integer basis for the combined global force-constant lattice.

    ``extract`` and ``project`` move between physical coefficients and free
    coordinates, while ``adjoint`` and ``restrict_rows`` apply the physical lift
    to observation rows. ``normal_basis`` spans the directions that violate the
    sum rule, which keeps rotational corrections inside the subspace.
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
        free_coordinates = _finite_vector(
            free_coordinates, (self.dimension,), "acoustic free coordinates"
        )
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
        coefficients = _finite_vector(coefficients, (self.width,), "canonical coefficient vector")
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

    def project(self, coefficients: object) -> np.ndarray:
        """Project physical coefficients onto the prepared acoustic subspace.

        Minimize ``||lift(z) - coefficients||_2`` in canonical physical
        parameter coordinates and return ``lift(z)``. The finite input must
        have shape ``(width,)``. Only lift and adjoint actions are used; no
        dense nullspace is formed. Invalid input raises ``ValueError`` and
        failure of the least-squares iteration raises ``RuntimeError``.
        """
        coefficients = _finite_vector(coefficients, (self.width,), "canonical coefficient vector")
        if not self.dimension:
            return np.zeros(self.width)
        operator = LinearOperator(
            (self.width, self.dimension),
            matvec=self.lift,
            rmatvec=self.adjoint,
            dtype=np.float64,
        )
        solution = lsmr(
            operator,
            coefficients,
            atol=1e-12,
            btol=1e-12,
            maxiter=max(1000, 2 * self.dimension),
        )
        if solution[1] not in (0, 1, 2, 4, 5):
            raise RuntimeError("acoustic constraint projection did not converge")
        return self.lift(solution[0])

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
            raise ValueError(
                f"acoustic observation rows must have shape (n, {self.width}): "
                f"got {physical_rows.shape}"
            )
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


def _pack_acoustic_coordinates(upper, pivots, physical_maps, width):
    """Assemble free acoustic coordinates from a certified factor.

    The free columns are the complement of the certified pivot columns; every
    stored array is frozen because the coordinates are shared through
    ``ClusterSpace``.
    """
    free_mask = np.ones(width, dtype=bool)
    free_mask[pivots] = False
    free = np.flatnonzero(free_mask)
    arrays = (*upper, pivots, free)
    for array in (*arrays, *physical_maps):
        array.setflags(write=False)
    return AcousticCoordinates(*arrays, physical_maps, width)


def prepare_coordinates(
    matrix: sparse.csr_matrix, physical_maps: tuple[np.ndarray, ...]
) -> AcousticCoordinates:
    """Build the physical acoustic subspace from one order's lattice equations.

    ``matrix`` expresses the translational-invariance conditions in lattice
    tensor coordinates and ``physical_maps`` converts each orbit's lattice
    coefficients to canonical Cartesian parameters. The returned coordinates
    cover the complete real nullspace while retaining the original physical
    coefficient layout.

    Parameters
    ----------
    matrix
        ``int64`` acoustic equation system of shape ``(m, n)``.
    physical_maps
        Per-orbit parameter maps whose column counts sum to ``n``.

    Returns
    -------
    AcousticCoordinates
        Free coordinates plus the certified factor of ``matrix``.

    Raises
    ------
    ArithmeticError
        If the fixed pivot schedule is inconsistent between reconstruction
        primes.
    OverflowError
        If exact reconstruction or its certificate exceeds the int64 domain.
    """
    upper, pivots = factor_acoustic_equations(matrix)
    return _pack_acoustic_coordinates(upper, pivots, physical_maps, matrix.shape[1])


def prepare_acoustic_coordinates(space):
    """Prepare the acoustic-invariant force-constant subspace for every order.

    Each included tensor order receives its own free coordinates while keeping
    the cluster space's canonical Cartesian parameter layout, so fitted and
    observed coefficients stay in the same coordinates as without ASR. Sparse
    exact factorization certifies every subspace; preparation can have
    substantial sparse fill for high orders.

    Parameters
    ----------
    space
        Cluster space whose orders should be constrained.

    Returns
    -------
    tuple of AcousticCoordinates
        One entry per order, in ``space.orders`` order.

    Raises
    ------
    ArithmeticError
        If a pivot schedule is inconsistent between reconstruction primes.
    OverflowError
        If exact reconstruction or its certificate exceeds the int64 domain.
    """
    return tuple(
        prepare_coordinates(*lattice_acoustic_equations(space, order)) for order in space.orders
    )


__all__ = [
    "prepare_acoustic_coordinates",
    "prepare_coordinates",
]
