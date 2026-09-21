"""Compiled reciprocal kernels against their NumPy oracle.

A compiled kernel is only worth having if it computes the same thing, so every test here
compares it with the tuple-based NumPy path it replaces -- on the crystal coverage the plan asks
for, including a non-diagonal supercell and zincblende GaAs.  The kernel also has to be in
nopython mode, has to be independent of the numba thread count, and has to stay correct with
``NUMBA_DISABLE_JIT`` set, which is how a machine without a working cache is simulated.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from ase.build import bulk
from test_reciprocal_full_grid_oracle import pair_bond_force_constants, scph_case

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.kernels import dynamical_matrices_compiled, kernel_is_nopython
from mlfcs.reciprocal.plan import FourierPlan
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies

CASES = ("cubic_2x1x1", "hcp_2x1x1", "diamond_2x1x1", "cubic_nondiagonal")


def _case(name: str):
    """Return one case's force constants, terms and q batch."""
    if name == "GaAs":
        primitive = bulk("GaAs", "zincblende", a=5.653)
        matrix = np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
        force_constants = pair_bond_force_constants(
            primitive, matrix, cutoff=3.2, spring=1.0, bend=0.5
        )[0]
    else:
        force_constants = scph_case(name)[0]
    relation = force_constants.relation
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    terms = fourier_terms(lattice_fc2(force_constants), relation.primitive)
    rng = np.random.default_rng(0)
    points = rng.random((11, 3)) - 0.5
    return terms, masses, points


@pytest.mark.parametrize("name", (*CASES, "GaAs"))
def test_the_plan_flattens_the_terms_exactly(name: str) -> None:
    """The flattened plan stores what the term tuple stores, with the mass weight folded in."""
    terms, masses, _ = _case(name)
    plan = FourierPlan.from_terms(terms, masses)
    np.testing.assert_array_equal(plan.first, [term.first for term in terms])
    np.testing.assert_array_equal(plan.second, [term.second for term in terms])
    np.testing.assert_array_equal(plan.images, [term.images for term in terms])
    np.testing.assert_array_equal(plan.tensors, [term.tensor for term in terms])
    np.testing.assert_allclose(
        plan.mass_weight,
        [1.0 / np.sqrt(masses[term.first] * masses[term.second]) for term in terms],
        rtol=0.0,
        atol=0.0,
    )
    assert plan.n_sites == len(masses)


@pytest.mark.parametrize("name", (*CASES, "GaAs"))
def test_the_compiled_kernel_matches_the_numpy_oracle(name: str) -> None:
    """Same matrices as the tuple path, to roundoff, on every required crystal."""
    terms, masses, points = _case(name)
    oracle = dynamical_matrices(terms, masses, points)
    compiled = dynamical_matrices_compiled(FourierPlan.from_terms(terms, masses), points)
    scale = float(np.max(np.abs(oracle)))
    np.testing.assert_allclose(compiled, oracle, rtol=1e-13, atol=1e-13 * scale)
    np.testing.assert_allclose(compiled, compiled.conj().swapaxes(-1, -2), rtol=0.0, atol=1e-15 * scale)


def test_the_kernel_is_nopython_and_thread_independent() -> None:
    """Object mode would be slower than NumPy, and a serial kernel cannot depend on threads."""
    import numba

    terms, masses, points = _case("diamond_2x1x1")
    plan = FourierPlan.from_terms(terms, masses)
    dynamical_matrices_compiled(plan, points)
    assert kernel_is_nopython(), _dynamical_signatures()

    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = dynamical_matrices_compiled(plan, points)
        numba.set_num_threads(min(4, previous))
        several = dynamical_matrices_compiled(plan, points)
    finally:
        numba.set_num_threads(previous)
    np.testing.assert_array_equal(single, several)


def test_the_frequency_path_keeps_its_eigensolver_batch(monkeypatch) -> None:
    """Compiling the matrix build must not change which matrices reach the eigensolver."""
    force_constants = scph_case("diamond_2x1x1")[0]
    shapes: list[tuple[int, ...]] = []
    original = np.linalg.eigvalsh

    def counting(values, *args, **kwargs):
        shapes.append(np.asarray(values).shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", counting)
    mesh = harmonic_frequencies(force_constants, 2)
    n_modes = 3 * len(force_constants.relation.primitive)
    assert shapes == [(mesh.n_irreducible, n_modes, n_modes)]


def test_the_kernel_stays_correct_with_jit_disabled() -> None:
    """``NUMBA_DISABLE_JIT=1`` is the portable path: the same numbers, no compiled cache."""
    code = (
        "import sys; sys.path.insert(0, 'tests');"
        "import numpy as np;"
        "from test_reciprocal_performance_kernels import _case;"
        "from mlfcs.reciprocal.fourier import dynamical_matrices;"
        "from mlfcs.reciprocal.kernels import dynamical_matrices_compiled;"
        "from mlfcs.reciprocal.plan import FourierPlan;"
        "terms, masses, points = _case('hcp_2x1x1');"
        "oracle = dynamical_matrices(terms, masses, points);"
        "compiled = dynamical_matrices_compiled(FourierPlan.from_terms(terms, masses), points);"
        "residual = float(np.max(np.abs(compiled - oracle)));"
        "scale = float(np.max(np.abs(oracle)));"
        "assert residual <= 1e-13 * scale, (residual, scale);"
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def _dynamical_signatures() -> list[str]:
    from mlfcs.reciprocal import kernels

    return [str(signature) for signature in kernels._dynamical_matrices_kernel.signatures]
