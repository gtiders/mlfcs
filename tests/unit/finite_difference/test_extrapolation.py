from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from mlfcs import ForceConstantCalculation
from mlfcs.finite_difference.extrapolation import ExtrapolationBackend


class NonFiniteCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.full((len(atoms), 3), np.nan)


def test_extrapolation_recovers_zero_step_derivative_and_reports_error():
    backend = ExtrapolationBackend(0.03, 0.005, side_steps=2, degree=1)
    expected = np.arange(6, dtype=float).reshape(2, 3)
    derivatives = [{((0, 0),): expected + 4.0 * step**2} for step in backend.grid]

    result, metrics = backend.extrapolate(derivatives)

    np.testing.assert_allclose(result[((0, 0),)], expected, atol=1e-12)
    assert metrics.maximum_correction > 0
    assert metrics.maximum_fit_residual < 1e-12


def test_extrapolation_grid_must_remain_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        ExtrapolationBackend(0.01, 0.005, side_steps=2)


def test_extrapolation_is_available_only_through_direct_calculator_run():
    calculation = ForceConstantCalculation(
        bulk("Al", "fcc", a=4.05),
        order=2,
        supercell=(2, 2, 2),
        cutoff=-1,
        displacement=0.02,
        jax_platform="cpu",
        verbose=False,
    )
    central_count = len(calculation.plan)

    result = calculation.run(
        EMT(),
        derivative_backend="extrapolate",
        extrapolation_spacing=0.005,
        extrapolation_side_steps=1,
        acoustic_sum_rule=False,
    )

    assert result.metadata["derivative_backend"] == "extrapolate"
    assert result.metadata["extrapolation_grid_angstrom"] == [0.015, 0.02, 0.025]
    assert result.metadata["extrapolation_degree"] == 1
    assert result.metadata["configurations"] == 3 * central_count


@pytest.mark.parametrize("backend", ["central", "extrapolate"])
def test_direct_calculator_rejects_nonfinite_forces(backend):
    calculation = ForceConstantCalculation(
        bulk("Al", "fcc", a=4.05),
        order=2,
        supercell=(2, 2, 2),
        cutoff=-1,
        verbose=False,
    )
    options = {"derivative_backend": backend}
    if backend == "extrapolate":
        options["extrapolation_spacing"] = 0.002
    with pytest.raises(ValueError, match="configuration 0.*NaN or infinite"):
        calculation.run(NonFiniteCalculator(), acoustic_sum_rule=False, **options)
