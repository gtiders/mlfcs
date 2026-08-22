"""Covariance-orthogonalized Wick features and basis conversion helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def wick(displacement, covariance, coordinates, order):
    """Evaluate a multivariate Wick monomial recursively."""
    flattened = displacement.reshape(-1)
    values = flattened[coordinates]
    if order == 0:
        return jnp.ones(coordinates.shape[:-1], dtype=displacement.dtype)
    if order == 1:
        return values[..., 0]
    first = coordinates[..., 0]
    result = values[..., 0] * wick(displacement, covariance, coordinates[..., 1:], order - 1)
    for partner in range(1, order):
        remaining = np.delete(np.arange(order), (0, partner))
        result -= covariance[first, coordinates[..., partner]] * wick(
            displacement, covariance, coordinates[..., remaining], order - 2
        )
    return result


def wick_axis_derivatives(displacement, covariance, coordinates, order):
    """Return all leave-one-axis Wick features with shared subset recursion."""
    flattened = displacement.reshape(-1)
    cache = {}

    def subset(axes):
        axes = tuple(axes)
        if axes in cache:
            return cache[axes]
        if not axes:
            value = jnp.ones(coordinates.shape[:-1], dtype=displacement.dtype)
        elif len(axes) == 1:
            value = flattened[coordinates[..., axes[0]]]
        else:
            first = axes[0]
            value = flattened[coordinates[..., first]] * subset(axes[1:])
            for position in range(1, len(axes)):
                partner = axes[position]
                remaining = axes[1:position] + axes[position + 1 :]
                value -= covariance[coordinates[..., first], coordinates[..., partner]] * subset(
                    remaining
                )
        cache[axes] = value
        return value

    return tuple(
        subset(tuple(axis for axis in range(order) if axis != omitted)) for omitted in range(order)
    )
