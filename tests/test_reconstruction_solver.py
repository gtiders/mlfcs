import numpy as np
import pytest
from ase.build import bulk

from mlfcs.clusters.orbits import build_orbit_space
from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.reconstruction.solver import reconstruct_sparse
from mlfcs.structure.geometry import make_supercell, resolve_cutoff


@pytest.mark.parametrize("order", [3, 4])
def test_reconstructs_every_orbit_from_independent_components(order):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    symmetry = SymmetryOperations.from_atoms(primitive, supercell)
    cutoff = resolve_cutoff(supercell, index, -1)
    space = build_orbit_space(
        supercell,
        index,
        symmetry,
        order=order,
        cutoff=cutoff,
    )

    derivatives = {
        key: np.zeros((len(supercell), 3), dtype=float) for key in space.displacement_keys
    }
    expected = {}
    for orbit_number, orbit in enumerate(space.orbits, start=1):
        coefficients = np.arange(1, orbit.dimension + 1, dtype=float) / orbit_number
        representative = orbit.basis @ coefficients
        for pivot in orbit.pivots:
            components = np.unravel_index(int(pivot), (3,) * order)
            key = tuple(
                (orbit.representative[axis], int(components[axis])) for axis in range(order - 1)
            )
            derivatives[key][orbit.representative[-1], components[-1]] = representative[pivot]
        for image in orbit.images:
            expected[image.cluster] = image.action.apply_flat(representative).reshape((3,) * order)

    compact = reconstruct_sparse(space, index, derivatives, enforce_asr=False).to_dense()
    for cluster, tensor in expected.items():
        np.testing.assert_allclose(compact[cluster], tensor, atol=1e-9)
