from __future__ import annotations

import numpy as np
import pytest

from mlfcs.anharmonic.common.schedule import normalize_temperature_schedule
from mlfcs.anharmonic.common.thermodynamics import quotient_qpoints
from mlfcs.core.integer_lattice import determinant_3x3


def _point_set(points: np.ndarray) -> set[tuple[float, float, float]]:
    return {tuple(np.round(point, 12)) for point in points}


def test_diagonal_reciprocal_quotient_is_the_expected_regular_grid():
    points = quotient_qpoints(np.diag([2, 3, 1]))
    expected = np.asarray(
        [[i / 2, j / 3, 0.0] for i in range(2) for j in range(3)], dtype=float
    )
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
    expected = np.asarray(
        [[0.0, 0.0, 0.0], [0.75, 0.5, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]]
    )
    np.testing.assert_allclose(points, expected, atol=1e-12, rtol=0.0)


def test_temperature_schedule_sorts_and_rejects_duplicates():
    assert normalize_temperature_schedule([600, 300, 450]) == (300.0, 450.0, 600.0)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_temperature_schedule([300, 300])
