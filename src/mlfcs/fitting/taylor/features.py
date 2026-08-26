"""Ordinary Taylor monomials used as fitting coordinates."""

from __future__ import annotations

import jax.numpy as jnp


def taylor(displacement, _state, coordinates, order):
    """Evaluate one multivariate Taylor monomial."""
    values = displacement.reshape(-1)[coordinates]
    return jnp.prod(values[..., :order], axis=-1)


def taylor_axis_derivatives(displacement, _state, coordinates, order):
    """Return the leave-one-axis monomial for every tensor axis."""
    values = displacement.reshape(-1)[coordinates]
    return tuple(
        jnp.prod(
            values[..., jnp.asarray([other for other in range(order) if other != axis])],
            axis=-1,
        )
        for axis in range(order)
    )


__all__ = ["taylor", "taylor_axis_derivatives"]
