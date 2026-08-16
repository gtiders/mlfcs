from mlfcs.io._text import zero_small_scalar


def test_zero_small_scalar_has_an_explicit_strict_text_tolerance():
    assert zero_small_scalar(9e-9, tolerance=1e-8) == 0.0
    assert zero_small_scalar(-9e-9, tolerance=1e-8) == 0.0
    assert zero_small_scalar(1e-8, tolerance=1e-8) == 1e-8
