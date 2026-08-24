"""Reference-symmetrized covariance owned by the Wick backend."""

from __future__ import annotations

import numpy as np


def symmetrized_covariance(displacements, calculation):
    """Average the empirical displacement covariance over lattice symmetries."""
    flattened = displacements.reshape(len(displacements), -1)
    covariance = flattened.T @ flattened / len(flattened)
    covariance = covariance.reshape(len(calculation.supercell), 3, len(calculation.supercell), 3)
    result = np.zeros_like(covariance)
    count = 0
    translations = calculation.index.cell_representatives
    translated_atoms = calculation.index.translate_atoms(
        np.arange(len(calculation.supercell), dtype=np.int32), translations
    )
    for translated in translated_atoms:
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
