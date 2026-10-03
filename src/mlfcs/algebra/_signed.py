"""Saturated kernels for signed incidence constraints."""

import numpy as np
from numba import njit


@njit(cache=True)
def signed_components(a):
    """Signed incidence constraints have a saturated component basis."""
    n = a.shape[1]
    parent = np.arange(n, dtype=np.int64)
    signs = np.ones(n, dtype=np.int64)
    dead = np.zeros(n, dtype=np.uint8)
    for i in range(a.shape[0]):
        first, second, count = -1, -1, 0
        for j in range(n):
            if a[i, j]:
                if count == 0:
                    first = j
                elif count == 1:
                    second = j
                count += 1
        if count > 2:
            return parent, signs, dead, np.empty(0, dtype=np.int64), -1
        if count == 0:
            continue
        x, sx = first, 1
        while parent[x] != x:
            sx *= signs[x]
            x = parent[x]
        if count == 1:
            dead[x] = 1
            continue
        if abs(a[i, first]) != abs(a[i, second]):
            return parent, signs, dead, np.empty(0, dtype=np.int64), -1
        relation = -1 if a[i, first] == a[i, second] else 1
        y, sy = second, 1
        while parent[y] != y:
            sy *= signs[y]
            y = parent[y]
        if x == y:
            if sx != relation * sy:
                dead[x] = 1
        else:
            parent[y] = x
            signs[y] = relation * sx * sy
            dead[x] = max(dead[x], dead[y])
    columns = np.full(n, -1, dtype=np.int64)
    dimension = 0
    for j in range(n):
        root = j
        while parent[root] != root:
            root = parent[root]
        if not dead[root] and columns[root] < 0:
            columns[root] = dimension
            dimension += 1
    return parent, signs, dead, columns, dimension


@njit(cache=True)
def component_basis(parent, signs, dead, columns, dimension):
    n = len(parent)
    result = np.zeros((n, dimension), dtype=np.int64)
    for j in range(n):
        root, sign = j, 1
        while parent[root] != root:
            sign *= signs[root]
            root = parent[root]
        if not dead[root]:
            result[j, columns[root]] = sign
    return result
