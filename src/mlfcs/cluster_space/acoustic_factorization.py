"""Exact nullspace coordinates for acoustic force-constant constraints.

For acoustic equations ``A x = 0``, ``A`` acts on lattice force-constant
coordinates. Its rank determines how many independent coordinates are removed
by translational invariance, and its nullspace is the complete allowed
parameter space.

Sparse modular elimination proposes rational triangular factors. Cleared-
denominator integer identities certify their rank and nullspace before
floating-point triangular solves map between free acoustic coordinates and
full lattice coordinates.
"""

import math

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from mlfcs.cluster_space.integer_kernel import reconstruct_rationals, scale_to_common_denominator
from mlfcs.foundation.integer import RANK_PRIMES, modular_power

LIMIT = np.int64(9223372036854775807)


@njit(cache=True)
def checked_add(left, right):
    """Add two signed certificate terms without wrapping the integer result."""
    if right > 0 and left > LIMIT - right:
        raise OverflowError("ASR certificate sum exceeds int64")
    if right < 0 and left < -LIMIT - right:
        raise OverflowError("ASR certificate sum exceeds int64")
    return left + right


@njit(cache=True)
def checked_multiply(left, right):
    """Multiply certificate terms within the supported symmetric int64 range."""
    if left == -LIMIT - 1 or right == -LIMIT - 1:
        raise OverflowError("ASR certificate excludes int64 minimum")
    if right != 0 and abs(left) > LIMIT // abs(right):
        raise OverflowError("ASR certificate product exceeds int64")
    return left * right


@njit(cache=True)
def _factor(indptr, indices, values, width, prime, schedule):
    """Factor the acoustic equations over one prime field.

    The factors satisfy ``A = L U`` modulo ``prime``. A shared pivot schedule
    aligns factorizations over different primes for rational reconstruction.
    Residue products remain within the signed int64 range.
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
    """Store sparse factor rows in sorted column order."""
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
    """Unite factor supports so a coefficient zero in one field is retained."""
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
    """Reconstruct one rational factor row from its two modular residues."""
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
    numerator, denominator, ok = reconstruct_rationals(left_values[:, :count], right_values[:, :count], bound)
    if not ok:
        raise OverflowError("ASR rational factor reconstruction failed")
    scaled, common, ok = scale_to_common_denominator(numerator, denominator)
    if not ok:
        raise OverflowError("ASR rational factor denominator exceeds word domain")
    return columns[:count], scaled[0], common


@njit(cache=True)
def _reconstruct_rows(first, second, bound):
    """Reconstruct the rational upper factor one bounded row at a time."""
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
                common = checked_multiply(common // math.gcd(common, denominator), denominator)
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
    """Map free acoustic coordinates to a full lattice vector in the nullspace."""
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
    """Apply the adjoint acoustic coordinate lift to observation rows."""
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


def factor_acoustic_equations(matrix):
    """Construct a certified triangular representation of acoustic equations.

    ``matrix`` represents translational-invariance constraints ``A x = 0`` in
    lattice force-constant coordinates. The returned sparse upper factor and
    pivot columns have the same nullspace as ``A``. Modular elimination and
    rational reconstruction propose the factor; a cleared-denominator exact
    identity and independent pivot witnesses certify its rank and nullspace.

    Raises
    ------
    OverflowError
        If reconstruction or its exact certificate exceeds the supported
        integer range.
    ArithmeticError
        If the fixed pivot schedule is inconsistent across reconstruction
        primes.
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
