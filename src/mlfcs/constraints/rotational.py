"""Physical FC2 Born--Huang and Huang constraint projection.

This module deliberately works on lattice-labelled sparse IFCs rather than
orbit parameters.  It is therefore shared by finite differences and fitting,
does not depend on a particular regression basis, and never changes FC3 or
higher orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants


@dataclass(frozen=True, slots=True)
class RotationalSumRuleResult:
    """Corrected FC2 and the projection's physical residuals."""

    force_constants: ForceConstants
    strength: float
    tolerance: float
    length_scale: float
    retained_rank: int
    acoustic_before: float
    acoustic_after: float
    born_huang_before: float | None
    born_huang_after: float | None
    huang_before: float | None
    huang_after: float | None
    relative_fc2_correction: float
    maximum_fc2_correction: float


def enforce_rotational_sum_rules(
    force_constants: ForceConstants,
    *,
    born_huang: bool = False,
    huang: bool = False,
    strength: float = 1.0,
    tolerance: float = 1e-8,
) -> RotationalSumRuleResult:
    """Return an FC2-projected copy with strict ASR and selected conditions.

    ``strength=1`` is the default strict projection.  Values in ``[0, 1)``
    retain that fraction of the minimum-norm Born--Huang/Huang correction,
    while ASR remains exact.  ``tolerance`` is one dimensionless spectral
    cutoff after all pair vectors are normalized by a characteristic nearest
    neighbour distance.

    The conditions are evaluated on exact primitive-lattice pair vectors.
    No Wigner--Seitz image reconstruction is needed because every sparse FC2
    entry carries its physical integer translation explicitly.
    """
    if not born_huang and not huang:
        raise ValueError("select born_huang=True and/or huang=True")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if force_constants.relation is None:
        raise ValueError("rotational sum rules require a verified StructureRelation")
    if 2 not in force_constants.sparse:
        raise ValueError("rotational sum rules require lattice-labelled sparse FC2")

    relation = force_constants.relation
    fc2 = force_constants.sparse[2]
    sites, translations = _lattice_labels(fc2, relation)
    keys, values, multiplicities, vectors, dyadics, nearest_lengths = _physical_fc2_values(
        fc2, sites, translations, relation
    )
    nonzero_lengths = nearest_lengths[nearest_lengths > 1e-12]
    if len(nonzero_lengths) == 0:
        raise ValueError("Born--Huang/Huang constraints require a non-onsite FC2 pair")
    length_scale = float(np.median(nonzero_lengths))

    initial = values.reshape(-1)
    weights = np.repeat(1.0 / multiplicities, 9)
    inverse_weights = 1.0 / weights
    asr = _asr_matrix(keys, relation.index.n_primitive)
    bh = _born_huang_matrix(keys, vectors / length_scale, relation.index.n_primitive)
    hu = _huang_matrix(keys, dyadics / length_scale**2, relation.index.n_primitive)
    selected = [matrix for enabled, matrix in ((born_huang, bh), (huang, hu)) if enabled]
    constraints = sparse.vstack(selected, format="csr")

    asr_before = _maximum_residual(asr, initial)
    bh_before = _maximum_residual(bh, initial) * length_scale if born_huang else None
    hu_before = _maximum_residual(hu, initial) * length_scale**2 if huang else None

    asr_projected = _metric_project(asr, initial, inverse_weights, tolerance)[0]
    correction, rank = _null_asr_constraint_correction(
        asr, constraints, asr_projected, inverse_weights, tolerance
    )
    projected = asr_projected - strength * correction
    # Algebraically the second correction is in ASR's null space.  Repeat the
    # exact ASR projection to remove only floating-point round-off.
    projected = _metric_project(asr, projected, inverse_weights, tolerance)[0]

    asr_after = _maximum_residual(asr, projected)
    bh_after = _maximum_residual(bh, projected) * length_scale if born_huang else None
    hu_after = _maximum_residual(hu, projected) * length_scale**2 if huang else None
    delta = projected - initial
    denominator = max(float(np.linalg.norm(initial)), np.finfo(float).eps)
    result_values = {
        "strength": float(strength),
        "tolerance": float(tolerance),
        "length_scale": length_scale,
        "retained_rank": rank,
        "acoustic_before": asr_before,
        "acoustic_after": asr_after,
        "born_huang_before": bh_before,
        "born_huang_after": bh_after,
        "huang_before": hu_before,
        "huang_after": hu_after,
        "relative_fc2_correction": float(np.linalg.norm(delta) / denominator),
        "maximum_fc2_correction": float(np.max(np.abs(delta), initial=0.0)),
    }
    corrected = _replace_fc2(
        force_constants, fc2, keys, projected.reshape((-1, 3, 3)), sites, translations
    )
    corrected.metadata = {
        **force_constants.metadata,
        "harmonic_constraints": {
            "born_huang": born_huang,
            "huang": huang,
            "strength": float(strength),
            "tolerance": float(tolerance),
            "length_scale": length_scale,
            "retained_rank": rank,
        },
    }
    return RotationalSumRuleResult(force_constants=corrected, **result_values)


def _lattice_labels(fc2: SparseOrderForceConstants, relation) -> tuple[np.ndarray, np.ndarray]:
    return fc2.sites.copy(), fc2.translations[:, 0, :].copy()


def _physical_fc2_values(fc2, sites, translations, relation):
    """Aggregate duplicate exact-R sparse rows and derive pair moments."""
    grouped: dict[tuple[int, int, int, int, int], list[int]] = {}
    for row, (site, translation) in enumerate(zip(sites, translations, strict=True)):
        key = (int(site[0]), int(site[1]), *(int(value) for value in translation))
        grouped.setdefault(key, []).append(row)
    keys = tuple(grouped)
    values = np.empty((len(keys), 3, 3), dtype=float)
    multiplicities = np.empty(len(keys), dtype=float)
    vectors = np.empty((len(keys), 3), dtype=float)
    dyadics = np.empty((len(keys), 3, 3), dtype=float)
    nearest_lengths = np.empty(len(keys), dtype=float)
    primitive = relation.primitive
    cell = np.asarray(primitive.cell)
    for location, key in enumerate(keys):
        rows = grouped[key]
        values[location] = np.mean(fc2.tensors[rows], axis=0)
        first, second, *translation = key
        vector = (
            primitive.positions[second]
            - primitive.positions[first]
            + np.asarray(translation, dtype=np.int32) @ cell
        )
        multiplicities[location] = 1.0
        vectors[location] = vector
        dyadics[location] = np.outer(vector, vector)
        nearest_lengths[location] = float(np.linalg.norm(vector))
    return keys, values, multiplicities, vectors, dyadics, nearest_lengths


def _asr_matrix(keys, n_primitive: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for block, (first, _second, *_translation) in enumerate(keys):
        for alpha in range(3):
            for beta in range(3):
                rows.append((first * 3 + alpha) * 3 + beta)
                columns.append(block * 9 + alpha * 3 + beta)
                data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_primitive * 9, len(keys) * 9)
    ).tocsr()


def _born_huang_matrix(keys, vectors: np.ndarray, n_primitive: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for block, ((first, _second, *_translation), vector) in enumerate(
        zip(keys, vectors, strict=True)
    ):
        for alpha in range(3):
            for pair, (beta, gamma) in enumerate(((0, 1), (0, 2), (1, 2))):
                row = (first * 3 + alpha) * 3 + pair
                rows.extend((row, row))
                columns.extend((block * 9 + alpha * 3 + beta, block * 9 + alpha * 3 + gamma))
                data.extend((float(vector[gamma]), -float(vector[beta])))
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_primitive * 9, len(keys) * 9)
    ).tocsr()


def _huang_matrix(keys, dyadics: np.ndarray, n_primitive: int) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for block, ((first, _second, *_translation), dyadic) in enumerate(
        zip(keys, dyadics, strict=True)
    ):
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    for delta in range(3):
                        row = ((first * 3 + alpha) * 3 + beta) * 9 + gamma * 3 + delta
                        rows.extend((row, row))
                        columns.extend(
                            (block * 9 + alpha * 3 + beta, block * 9 + gamma * 3 + delta)
                        )
                        data.extend((float(dyadic[gamma, delta]), -float(dyadic[alpha, beta])))
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(n_primitive * 81, len(keys) * 9)
    ).tocsr()


def _spectral_pinv(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    maximum = float(np.max(np.abs(values), initial=0.0))
    if maximum == 0.0:
        return np.zeros_like(matrix), 0
    retained = values > tolerance * max(1.0, maximum)
    inverse = np.zeros_like(values)
    inverse[retained] = 1.0 / values[retained]
    return (vectors * inverse) @ vectors.T, int(np.count_nonzero(retained))


def _metric_project(matrix, values, inverse_weights, tolerance):
    if matrix.shape[0] == 0:
        return values.copy(), 0
    weighted = matrix.multiply(inverse_weights)
    gram = (weighted @ matrix.T).toarray()
    inverse, rank = _spectral_pinv(gram, tolerance)
    correction = inverse_weights * np.asarray(matrix.T @ (inverse @ (matrix @ values))).reshape(-1)
    return values - correction, rank


def _null_asr_constraint_correction(asr, constraints, values, inverse_weights, tolerance):
    weighted_asr = asr.multiply(inverse_weights)
    asr_gram = (weighted_asr @ asr.T).toarray()
    asr_inverse, _ = _spectral_pinv(asr_gram, tolerance)
    weighted_constraints = constraints.multiply(inverse_weights)
    gram = (weighted_constraints @ constraints.T).toarray()
    cross = (weighted_constraints @ asr.T).toarray()
    null_gram = gram - cross @ asr_inverse @ cross.T
    inverse, rank = _spectral_pinv(null_gram, tolerance)
    residual = np.asarray(constraints @ values).reshape(-1)
    multipliers = inverse @ residual
    direct = inverse_weights * np.asarray(constraints.T @ multipliers).reshape(-1)
    asr_part = inverse_weights * np.asarray(
        asr.T @ (asr_inverse @ (cross.T @ multipliers))
    ).reshape(-1)
    return direct - asr_part, rank


def _maximum_residual(matrix, values: np.ndarray) -> float:
    if matrix.shape[0] == 0:
        return 0.0
    return float(np.max(np.abs(matrix @ values), initial=0.0))


def _replace_fc2(force_constants, fc2, keys, values, sites, translations) -> ForceConstants:
    lookup = {key: value for key, value in zip(keys, values, strict=True)}
    tensors = np.empty_like(fc2.tensors)
    for row, (site, translation) in enumerate(zip(sites, translations, strict=True)):
        key = (int(site[0]), int(site[1]), *(int(value) for value in translation))
        tensors[row] = lookup[key]
    sparse_values = dict(force_constants.sparse)
    sparse_values[2] = SparseOrderForceConstants(
        order=fc2.order,
        sites=fc2.sites.copy(),
        translations=fc2.translations.copy(),
        tensors=tensors,
    )
    arrays = {order: array.copy() for order, array in force_constants.arrays.items() if order != 2}
    return ForceConstants(
        arrays=arrays,
        supercell=force_constants.supercell.copy(),
        metadata=dict(force_constants.metadata),
        sparse=sparse_values,
        relation=force_constants.relation,
    )


__all__ = [
    "RotationalSumRuleResult",
    "enforce_rotational_sum_rules",
]
