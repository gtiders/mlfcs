"""The harmonic sampler and the SCPH kernel must share one Fourier gauge.

The sampler used to build ``exp[2 pi i q . R]`` while SCPH used
``exp[2 pi i q . (R + tau_b - tau_a)]``.  Eigenvalues agree, eigenvectors differ by a
per-atom phase, so an eigenvector expanded by one consumer would be meaningless to the
other.  This test pins the unified gauge through the two independent kernels: the compact
supercell form used by the sampler and the exact-lattice form used by SCPH must agree at
every q point of the supercell grid, on diagonal, non-diagonal and anisotropic supercells.
"""

from __future__ import annotations

import numpy as np
import pytest
from reciprocal_helpers import CASES, SUPERCELLS, crystal_case, relation_reference

from mlfcs.force_constants.dense import lattice_fc2
from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.reciprocal.fourier import dynamical_matrix, fourier_terms
from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("supercell", sorted(SUPERCELLS))
def test_harmonic_sampler_and_fourier_kernel_agree(case_name, supercell):
    """The sampler's compact kernel is the same gauge as the lattice kernel.

    The force constants come from a single source supercell and only the *target* of the
    realization is parametrized: a fold that is too small to identify its own parameters is
    refused by the realization instead of being compared, which is the subject of
    ``test_reciprocal_boundaries.py`` rather than of a gauge statement.
    """
    _, force_constants, primitive, _ = crystal_case(case_name)
    reference = relation_reference(primitive, SUPERCELLS[supercell])
    compact = realize_force_constants(
        force_constants, reference, primitive=primitive
    ).materialize(2, max_bytes=None)
    # The gauge is measured on the matrices, not on the stability of the model: a
    # Lennard-Jones diamond is not at equilibrium, so imaginary modes are expected.
    sampler = HarmonicSampler(
        primitive, reference, compact, temperature=300.0, imaginary_modes="absolute"
    )
    terms = fourier_terms(lattice_fc2(force_constants), primitive)
    masses = np.asarray(primitive.get_masses(), dtype=float)
    matrix_from_lattice = np.asarray(
        [dynamical_matrix(terms, masses, q) for q in sampler.qpoints]
    )
    matrix_from_compact = np.asarray([sampler._dynamical_matrix(q) for q in sampler.qpoints])
    np.testing.assert_allclose(matrix_from_compact, matrix_from_lattice, rtol=1e-10, atol=1e-12)
