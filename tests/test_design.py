"""Numerical behavior of design."""

import numpy as np
from ase.build import bulk
from numba import get_num_threads, set_num_threads

from mlfcs import ClusterMap, ClusterSpace
from mlfcs.fitting.design import ForceDesign


def test_force_design_matches_between_thread_counts():
    """Verify force design matches between thread counts."""
    atoms = bulk("Ar", "sc", a=1)
    space = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 1.1}, max_body_orders={2: 2})
    mapping = ClusterMap(
        space, atoms.repeat((3,) * 3), supercell_matrix=3 * np.eye(3, dtype=np.int64)
    )
    design = ForceDesign(mapping)
    previous = get_num_threads()
    try:
        set_num_threads(1)
        workspace = design.allocate_workspace()
        displacement = np.random.default_rng(81).normal(size=(27, 3))
        single = design.matrix(displacement, workspace=workspace)
        if previous >= 2:
            set_num_threads(2)
            parallel = design.matrix(displacement)
            np.testing.assert_array_equal(single, parallel)
    finally:
        set_num_threads(previous)
