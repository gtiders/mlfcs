from __future__ import annotations

import numpy as np
import pytest
from test_analytic_morse_fc4_oracle import (
    R0,
    error_metrics,
    finite_difference_fc4,
    relaxed_primitive,
)


def test_ASE_relaxation_recovers_analytic_FCC_Morse_equilibrium():
    atoms, steps = relaxed_primitive()
    conventional_lattice_constant = np.sqrt(2.0) * np.linalg.norm(atoms.cell[0])

    assert steps < 10
    assert conventional_lattice_constant == pytest.approx(np.sqrt(2.0) * R0, abs=1.0e-12)
    assert atoms.get_potential_energy() == pytest.approx(-6.0, abs=1.0e-12)
    assert np.max(np.abs(atoms.get_stress())) < 1.0e-11


def test_FC4_matches_independent_JAX_Morse_derivative_and_converges_quadratically():
    errors = []
    for displacement in (0.01, 0.005, 0.0025):
        actual, exact = finite_difference_fc4(displacement)
        errors.append(error_metrics(actual, exact))

    finest = errors[-1]
    assert finest["maximum"] < 4.8
    assert finest["rms"] < 0.21
    assert finest["relative_l2"] < 2.1e-4
    assert finest["correlation"] > 0.9999999

    for metric in ("maximum", "rms", "relative_l2"):
        ratios = [errors[index][metric] / errors[index + 1][metric] for index in (0, 1)]
        assert np.allclose(ratios, 4.0, atol=0.06, rtol=0)
