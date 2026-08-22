import numpy as np
import pytest
from ase.build import bulk
from supercell_helpers import make_supercell

from mlfcs.finite_difference.reconstruction import reconstruct_sparse
from mlfcs.interactions.enumerate import build_primitive_interaction_space
from mlfcs.interactions.realization import realize_orbit_space


@pytest.mark.parametrize("order", [3, 4])
def test_reconstructs_every_orbit_from_independent_components(order):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    primitive_space = build_primitive_interaction_space(
        primitive,
        order=order,
        cutoff=-1,
        max_body_order=None,
        symprec=1e-5,
    )
    space = realize_orbit_space(primitive_space, index)

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
            tensor = image.action.apply_flat(representative).reshape((3,) * order)
            expected[image.cluster] = expected.get(image.cluster, 0.0) + tensor

    sparse = reconstruct_sparse(
        space,
        index,
        derivatives,
        enforce_asr=False,
        primitive_interaction_space=primitive_space,
    )
    from mlfcs.force_constants.data import ForceConstants
    from mlfcs.structure.relation import StructureRelation

    compact = ForceConstants(
        {},
        supercell,
        sparse={order: sparse},
        relation=StructureRelation.from_atoms(primitive, supercell),
    ).materialize(order)
    for cluster, tensor in expected.items():
        dense_key = (index.primitive[cluster[0]], *cluster[1:])
        np.testing.assert_allclose(compact[dense_key], tensor, atol=1e-9)
