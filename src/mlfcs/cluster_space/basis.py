"""Construct symmetry-invariant tensor spaces and physical Cartesian parameterizations.

The module first builds the tensor subspace compatible with repeated-site
index symmetry and the stabilizer of a representative cluster. The resulting
invariant space is represented exactly in lattice coordinates and can then be
converted to Cartesian coordinates and reparameterized by selected physical
tensor components.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numba import njit, types
from numba.typed import Dict, List
from scipy.linalg import qr

from mlfcs.cluster_space.integer_kernel import integer_kernel_basis
from mlfcs.foundation.arrays import as_int64_array, require_allocation
from mlfcs.foundation.tensors import apply_tensor_action, tensor_dimension, validate_tensor_action

# Invariant bases repeat heavily across orbits: a cluster space groups thousands
# of orbits into a few dozen distinct (repeated-site seed, stabilizer action)
# signatures. Entries hold frozen arrays shared between callers.
_INVARIANT_CACHE: dict[tuple[bytes, bytes], np.ndarray] = {}
_PARAMETERIZATION_CACHE: dict[tuple[tuple[int, ...], bytes], tuple] = {}
_CACHE_LIMIT = 256


def _action_signature(rotations: np.ndarray, permutations: np.ndarray) -> bytes:
    """Return an action-order-insensitive byte signature of stabilizer actions."""
    if not len(rotations):
        return b""
    stacked = np.hstack((rotations.reshape(len(rotations), -1), permutations))
    return np.ascontiguousarray(stacked[np.lexsort(stacked.T[::-1])]).tobytes()


def _label_symmetric_basis(labels: Sequence[object]) -> np.ndarray:
    """Return the tensor basis imposed by repeated periodic sites.

    Tensor slots carrying equal lattice-site labels are interchangeable:
    Tensor components related only by permutations among equal slots share
    one independent component.
    """
    dimension = tensor_dimension(len(labels))
    require_allocation("label basis", (dimension, dimension))
    equal = np.asarray([[a == b for b in labels] for a in labels], dtype=np.uint8).reshape(
        len(labels), len(labels)
    )
    return label_basis(equal)


def compute_invariant_basis(
    labels: Sequence[object],
    stabilizers: Sequence[tuple[np.ndarray, tuple[int, ...]]],
) -> np.ndarray:
    """Construct the exact tensor basis invariant under the cluster stabilizer.

    Let ``p = len(labels)`` and let ``E`` denote the repeated-site tensor
    basis induced by repeated lattice sites. For every stabilizer operation
    ``g`` with lattice rotation ``R_g`` and tensor-axis permutation ``P_g``,
    the allowed tensors satisfy ``T_g Phi = Phi``, where ``T_g`` is the
    corresponding order-``p`` tensor action. In the coordinates of ``E``, the
    invariant subspace is therefore the integer kernel of the stacked
    constraints ``(T_g - I) E c = 0`` over all stabilizer operations.

    Parameters
    ----------
    labels
        Lattice-site labels of the representative cluster, one per tensor
        index.
    stabilizers
        Stabilizer operations of the representative cluster. Each entry is
        ``(rotation, permutation)``, where ``rotation`` is a ``(3, 3)``
        integer lattice-coordinate tensor action and ``permutation`` has
        length ``p``.

    Returns
    -------
    ndarray
        Saturated, readonly ``int64`` basis of shape ``(3**p, d)``. Its columns
        span exactly the tensor subspace satisfying both repeated-site index
        symmetry and all stabilizer constraints. The array may be shared with
        other calls that receive the same repeated-site seed and the same
        stabilizer action set, so callers must treat it as immutable.

    Raises
    ------
    ValueError
        If the stabilizer buffers have inconsistent shapes or an axis action
        is not a permutation.
    OverflowError
        If a rotation's tensor-action bound cannot be represented.

    Notes
    -----
    Stabilizer action sets repeat across orbits, so results are memoized on the
    ``(repeated-site seed, action set)`` signature; the action set is compared
    without regard to order. Constraint rows are hash-deduplicated while they
    are produced, which avoids materializing and sorting the full stack of
    ``|stabilizers| * 3**p`` rows.
    """
    repeated_site_basis = _label_symmetric_basis(labels)
    if stabilizers:
        rotations = as_int64_array([rotation for rotation, _ in stabilizers])
        permutations = as_int64_array([permutation for _, permutation in stabilizers])
        if rotations.shape != (len(stabilizers), 3, 3) or permutations.shape != (
            len(stabilizers),
            len(labels),
        ):
            raise ValueError("stabilizer buffers have inconsistent shapes")
        if not np.all(np.sort(permutations, axis=1) == np.arange(len(labels))):
            raise ValueError("stabilizer axes must be permutations")
        for rotation in rotations:
            validate_tensor_action(rotation, len(labels), subtraction=1)
        key = (repeated_site_basis.tobytes(), _action_signature(rotations, permutations))
        cached = _INVARIANT_CACHE.get(key)
        if cached is not None:
            return cached
        constraints = unique_constraint_rows(repeated_site_basis, rotations, permutations)
    else:
        key = (repeated_site_basis.tobytes(), b"")
        cached = _INVARIANT_CACHE.get(key)
        if cached is not None:
            return cached
        constraints = np.empty((0, repeated_site_basis.shape[1]), dtype=np.int64)
    # Row order is canonicalized before the integer kernel so that identical
    # signatures produce identical bases regardless of stabilizer ordering.
    constraints = np.unique(constraints, axis=0)
    kernel = integer_kernel_basis(constraints)
    # Each repeated-site basis row selects exactly one column; this avoids an
    # unnecessary integer dot product and its coefficient-growth bound.
    result = kernel[np.argmax(repeated_site_basis, axis=1)]
    result.setflags(write=False)
    if len(_INVARIANT_CACHE) >= _CACHE_LIMIT:
        _INVARIANT_CACHE.clear()
    _INVARIANT_CACHE[key] = result
    return result


__all__: list[str] = []


@njit(cache=True)
def label_basis(equal):
    """Construct the tensor basis generated by repeated-site index symmetry.

    ``equal[i, j]`` indicates whether tensor slots ``i`` and ``j`` refer to
    the same lattice site. Cartesian tensor components that differ only by
    permutations among equal slots are identified as one independent
    component.

    For tensor order ``p``, Cartesian components are flattened in C order,
    giving ``3**p`` rows. The returned matrix has shape ``(3**p, d)``, where
    ``d`` is the number of resulting equivalence classes. Each row contains
    exactly one unit entry, assigning that Cartesian component to its class.
    """
    order = equal.shape[0]
    dimension = 3**order
    classes = np.full(dimension, -1, dtype=np.int64)
    column_of_row = np.empty(dimension, dtype=np.int64)
    count = 0
    for row in range(dimension):
        directions = np.empty(order, dtype=np.int64)
        for i in range(order):
            directions[i] = row // (3 ** (order - 1 - i)) % 3
        for i in range(order):
            for j in range(i + 1, order):
                if equal[i, j] and directions[j] < directions[i]:
                    directions[i], directions[j] = directions[j], directions[i]
        key = 0
        for i in range(order):
            key = 3 * key + directions[i]
        if classes[key] < 0:
            classes[key] = count
            count += 1
        column_of_row[row] = classes[key]
    result = np.zeros((dimension, count), dtype=np.int64)
    for row in range(dimension):
        result[row, column_of_row[row]] = 1
    return result


@njit(cache=True)
def unique_constraint_rows(seed, rotations, permutations):
    """Collect the distinct stabilizer constraint rows without stacking them.

    For each stabilizer operation ``g`` the rows are ``(T_g - I) seed``; across
    all operations the stack repeats the same rows thousands of times, so rows
    are hash-deduplicated as they are produced instead of being materialized
    and sorted afterwards. Inputs are preserved and not validated: allocation
    and rotation-bound admission belong to the caller.

    The returned rows are the exact row set of the stacked constraints in
    first-occurrence order; callers canonicalize the order when the resulting
    basis must not depend on stabilizer ordering. Hash collisions are detected
    by comparing stored rows, so no distinct row is ever dropped.
    """
    rows, columns = seed.shape
    table = Dict.empty(types.int64, types.int64)
    unique = List()
    buffer = np.empty(columns, dtype=np.int64)
    for operation in range(len(rotations)):
        transformed = apply_tensor_action(seed, rotations[operation], permutations[operation])
        for row in range(rows):
            mixed = np.int64(0)
            for column in range(columns):
                value = transformed[row, column] - seed[row, column]
                buffer[column] = value
                mixed = (mixed ^ value) * np.int64(1099511628211)
            slot = mixed
            duplicate = False
            while slot in table:
                candidate = unique[table[slot]]
                equal = True
                for column in range(columns):
                    if candidate[column] != buffer[column]:
                        equal = False
                        break
                if equal:
                    duplicate = True
                    break
                slot += np.int64(1)
            if not duplicate:
                table[slot] = len(unique)
                unique.append(buffer.copy())
    result = np.empty((len(unique), columns), dtype=np.int64)
    for index in range(len(unique)):
        result[index] = unique[index]
    return result


def component_parameterization(cartesian: object) -> tuple[np.ndarray, np.ndarray, float]:
    """Reparameterize a Cartesian tensor subspace by physical tensor components.

    Let ``cartesian`` span a ``d``-dimensional subspace of the flattened
    Cartesian tensor space. This function selects ``d`` tensor-component rows
    whose restriction is nonsingular and constructs an equivalent basis
    ``B`` satisfying ``B[rows, :] = I``. Consequently, the parameter vector
    in ``Phi_flat = B @ coefficients`` is exactly the set of selected
    Cartesian tensor components ``Phi_flat[rows]``.

    Parameters
    ----------
    cartesian
        Finite two-dimensional array with ``d`` columns whose columns span
        the Cartesian tensor subspace.

    Returns
    -------
    basis
        Equivalent floating-point basis with the same shape as ``cartesian``
        satisfying ``basis[rows] = I`` up to numerical roundoff.
    rows
        Integer array of shape ``(d,)`` containing the selected Cartesian
        component indices, sorted.
    condition
        Spectral condition number of the selected row block before
        reparameterization.

    Raises
    ------
    ValueError
        If the input is not a finite two-dimensional array or the supplied
        subspace loses its expected numerical rank.

    Notes
    -----
    Rows are selected greedily from an orthonormal basis using modified
    Gram-Schmidt residual norms. The condition number is reported as a
    diagnostic and does not change the subspace dimension.

    The result is memoized on the input bytes: within one cluster space the same
    Cartesian subspace recurs once per symmetry-equivalent orbit. Returned
    arrays are readonly and may be shared between calls, so callers must not
    mutate them.
    """
    values = np.asarray(cartesian, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Cartesian basis must be a finite two-dimensional array")
    rows, dimension = values.shape
    if dimension == 0:
        return np.empty((rows, 0)), np.empty(0, dtype=np.int64), 1.0
    key = (values.shape, values.tobytes())
    cached = _PARAMETERIZATION_CACHE.get(key)
    if cached is not None:
        return cached
    orthonormal, _ = qr(values, mode="economic", check_finite=False)
    selected = select_observation_rows(orthonormal)
    observed = orthonormal[selected]
    basis = np.linalg.solve(observed.T, orthonormal.T).T
    condition = float(np.linalg.cond(observed, 2))
    basis.setflags(write=False)
    selected.setflags(write=False)
    if len(_PARAMETERIZATION_CACHE) >= _CACHE_LIMIT:
        _PARAMETERIZATION_CACHE.clear()
    result = (basis, selected, condition)
    _PARAMETERIZATION_CACHE[key] = result
    return result


@njit(cache=True)
def select_observation_rows(orthonormal: np.ndarray) -> np.ndarray:
    """Select independent observation rows by greedy Gram-Schmidt residuals.

    At each step, the available row with the largest squared residual norm is
    selected, then its normalized direction is projected from all remaining
    rows. The procedure selects exactly one row per subspace dimension and
    raises ValueError if fewer than that many numerically independent rows
    can be selected. Inputs have orthonormal columns; the returned row
    indices are sorted.
    """
    rows, dimension = orthonormal.shape
    residuals = orthonormal.copy()
    available = np.ones(rows, dtype=np.uint8)
    selected = np.empty(dimension, dtype=np.int64)
    for column in range(dimension):
        best_row = -1
        best_score = -1.0
        for row in range(rows):
            if available[row] == 0:
                continue
            score = 0.0
            for component in range(dimension):
                score += residuals[row, component] * residuals[row, component]
            if score > best_score:
                best_score = score
                best_row = row
        if best_row < 0 or not best_score > 0.0:
            raise ValueError("Cartesian invariant subspace lost its expected dimension")
        selected[column] = best_row
        available[best_row] = 0
        inverse_norm = 1.0 / np.sqrt(best_score)
        direction = np.empty(dimension, dtype=np.float64)
        for component in range(dimension):
            direction[component] = residuals[best_row, component] * inverse_norm
        for row in range(rows):
            overlap = 0.0
            for component in range(dimension):
                overlap += residuals[row, component] * direction[component]
            for component in range(dimension):
                residuals[row, component] -= overlap * direction[component]
    return np.sort(selected)
