"""Regression guards for the host/JAX execution boundary."""

from __future__ import annotations

import subprocess
import sys


def test_base_and_finite_difference_imports_do_not_initialize_jax():
    code = """
import sys
import mlfcs
from mlfcs import FiniteDifferenceCalculation
assert 'jax' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
