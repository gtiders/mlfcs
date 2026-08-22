from __future__ import annotations

import numpy as np
import pytest

from mlfcs.physics.temperature import normalize_temperature_schedule
from mlfcs.structure.integer_lattice import adjugate_3x3, determinant_3x3, residue_key
from mlfcs.structure.reciprocal import (
    quotient_qpoints,
    reciprocal_quotient_grid,
)


def _point_set(points: np.ndarray) -> set[tuple[float, float, float]]:
    return {tuple(np.round(point, 12)) for point in points}


def test_diagonal_reciprocal_quotient_is_the_expected_regular_grid():
    points = quotient_qpoints(np.diag([2, 3, 1]))
    expected = np.asarray([[i / 2, j / 3, 0.0] for i in range(2) for j in range(3)], dtype=float)
    assert _point_set(points) == _point_set(expected)


def test_general_reciprocal_quotient_has_exact_count_and_periodic_phases():
    matrix = np.asarray([[2, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
    points = quotient_qpoints(matrix)
    assert len(points) == abs(determinant_3x3(matrix))
    np.testing.assert_allclose(matrix @ points.T, np.rint(matrix @ points.T), atol=1e-12)
    assert len(_point_set(points)) == len(points)


def test_general_reciprocal_quotient_uses_hnf_canonical_order():
    matrix = np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=int)
    points = quotient_qpoints(matrix)
    expected = np.asarray([[0.0, 0.0, 0.0], [0.75, 0.5, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]])
    np.testing.assert_allclose(points, expected, atol=1e-12, rtol=0.0)


def test_sheared_reciprocal_quotient_matches_legacy_set_not_legacy_order():
    matrix = np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64)
    determinant = abs(determinant_3x3(matrix))
    current = quotient_qpoints(matrix)

    legacy_representatives = {}
    for values in np.ndindex((determinant, determinant, determinant)):
        candidate = np.asarray(values, dtype=np.int64)
        legacy_representatives.setdefault(residue_key(candidate, matrix.T), candidate)
        if len(legacy_representatives) == determinant:
            break
    legacy_numerators = np.asarray(list(legacy_representatives.values())) @ adjugate_3x3(matrix).T
    legacy = np.mod(legacy_numerators, determinant).astype(float) / determinant

    assert len(current) == determinant == 8
    assert len(_point_set(current)) == determinant
    assert _point_set(current) == _point_set(legacy)
    np.testing.assert_allclose(current @ matrix.T, np.rint(current @ matrix.T), atol=1e-12)


def test_exact_reciprocal_labels_pair_q_and_negative_q_without_float_rounding():
    matrix = np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64)
    grid = reciprocal_quotient_grid(matrix)
    labels = {tuple(int(value) for value in label) for label in grid.labels}
    assert len(labels) == grid.denominator == abs(determinant_3x3(matrix))
    for label in grid.labels:
        assert grid.negative_label(label) in labels
    np.testing.assert_array_equal(
        grid.labels.astype(float) / grid.denominator,
        grid.points,
    )


def test_temperature_schedule_sorts_and_rejects_duplicates():
    assert normalize_temperature_schedule([600, 300, 450]) == (300.0, 450.0, 600.0)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_temperature_schedule([300, 300])
