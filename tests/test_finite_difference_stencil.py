import numpy as np

from mlfcs.finite_difference.stencil import CentralDifferenceStencil


def test_recursive_stencil_sizes_and_denominator():
    third = CentralDifferenceStencil.for_force_constant(3, 0.01)
    fourth = CentralDifferenceStencil.for_force_constant(4, 0.01)
    assert third.signs.shape == (4, 2)
    assert fourth.signs.shape == (8, 3)
    assert third.denominator == (0.02) ** 2
    assert fourth.denominator == (0.02) ** 3


def test_mixed_derivative_of_polynomial():
    stencil = CentralDifferenceStencil(3, 0.02)
    values = np.prod(stencil.signs * stencil.step, axis=1)
    assert np.isclose(stencil.contract(values), 1.0)
