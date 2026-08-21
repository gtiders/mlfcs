"""Covariance-orthogonalized Wick features and basis conversion helpers."""

from __future__ import annotations

from math import factorial

import jax
import jax.numpy as jnp
import numpy as np

from mlfcs.ifc.model import SparseOrderForceConstants

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


def symmetrized_covariance(displacements, calculation):
    """Average the empirical displacement covariance over lattice symmetries."""
    flattened = displacements.reshape(len(displacements), -1)
    covariance = flattened.T @ flattened / len(flattened)
    covariance = covariance.reshape(len(calculation.supercell), 3, len(calculation.supercell), 3)
    result = np.zeros_like(covariance)
    count = 0
    translations = np.unique(calculation.index.translations, axis=0)
    for shift in translations:
        translated = np.asarray(
            [
                calculation.index.translate_atom(atom, shift)
                for atom in range(len(calculation.supercell))
            ]
        )
        translation_inverse = np.argsort(translated)
        translated_covariance = covariance[translation_inverse][:, :, translation_inverse, :]
        for permutation, rotation in zip(
            calculation.symmetry.atom_permutations,
            calculation.symmetry.cartesian_rotations,
            strict=True,
        ):
            rotated = np.einsum(
                "ag,igjd,bd->iajb", rotation, translated_covariance, rotation, optimize=True
            )
            inverse = np.argsort(permutation)
            result += rotated[inverse][:, :, inverse, :]
            count += 1
    result = result.reshape(flattened.shape[1], flattened.shape[1]) / count
    return (result + result.T) * 0.5


def convert_sparse_wick_reference(force_constants, covariance):
    """Reference tensor-space Wick-to-Taylor conversion used for verification."""
    wick_values = {
        order: SparseOrderForceConstants(
            values.order,
            values.n_primitive,
            values.n_supercell,
            values.clusters.copy(),
            values.tensors.copy(),
            None if values.sites is None else values.sites.copy(),
            (
                None
                if values.translations is None
                else values.translations.copy()
            ),
        )
        for order, values in force_constants.items()
    }
    result = {
        order: SparseOrderForceConstants(
            values.order,
            values.n_primitive,
            values.n_supercell,
            values.clusters.copy(),
            values.tensors.copy(),
            None if values.sites is None else values.sites.copy(),
            (
                None
                if values.translations is None
                else values.translations.copy()
            ),
        )
        for order, values in wick_values.items()
    }
    covariance = np.asarray(covariance).reshape(
        next(iter(result.values())).n_supercell,
        3,
        next(iter(result.values())).n_supercell,
        3,
    )
    maximum_order = max(result, default=0)
    for target_order in sorted(result):
        target = result[target_order]
        labelled = target.is_lattice_labelled
        def physical_key(values, row, order):
            if values.is_lattice_labelled:
                return (
                    tuple(map(int, values.sites[row, :order])),
                    tuple(
                        tuple(map(int, vector))
                        for vector in values.translations[row, : order - 1]
                    ),
                )
            return tuple(map(int, values.clusters[row, :order]))

        tensors_by_cluster = {}
        clusters_by_key = {}
        for row, (cluster, tensor) in enumerate(
            zip(target.clusters, target.tensors, strict=True)
        ):
            key = physical_key(target, row, target_order)
            tensors_by_cluster[key] = tensors_by_cluster.get(
                key, np.zeros((3,) * target_order)
            ) + tensor
            clusters_by_key.setdefault(key, np.asarray(cluster, dtype=np.int32))
        for source_order in range(target_order + 2, maximum_order + 1, 2):
            if source_order not in wick_values:
                continue
            pairs = (source_order - target_order) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            source = wick_values[source_order]
            for row, (cluster, tensor) in enumerate(
                zip(source.clusters, source.tensors, strict=True)
            ):
                contracted = tensor
                for pair in range(pairs):
                    left = target_order + 2 * pair
                    atom_left, atom_right = int(cluster[left]), int(cluster[left + 1])
                    contracted = np.einsum(
                        "...ab,ab->...",
                        contracted,
                        covariance[atom_left, :, atom_right, :],
                        optimize=True,
                    )
                key = physical_key(source, row, target_order)
                clusters_by_key.setdefault(key, np.asarray(cluster[:target_order], dtype=np.int32))
                tensors_by_cluster[key] = (
                    tensors_by_cluster.get(key, np.zeros((3,) * target_order))
                    + coefficient * contracted
                )
        keys = tuple(tensors_by_cluster)
        clusters = np.asarray([clusters_by_key[key] for key in keys], dtype=np.int32).reshape(
            (-1, target_order)
        )
        tensors = np.asarray(
            [tensors_by_cluster[key] for key in keys], dtype=float
        ).reshape((-1,) + (3,) * target_order)
        original = result[target_order]
        sites = translations = None
        if labelled:
            sites = np.asarray([key[0] for key in keys], dtype=np.int32)
            translations = np.asarray([key[1] for key in keys], dtype=np.int32).reshape(
                (-1, target_order - 1, 3)
            )
        result[target_order] = SparseOrderForceConstants(
            target_order,
            original.n_primitive,
            original.n_supercell,
            clusters,
            tensors,
            sites,
            translations,
        )
    return result
