import numpy as np
from ase import Atoms

from mlfcs import ForceConstantCalculation, build_supercell
from mlfcs.core.real_space import InteractionKey, build_primitive_interaction_space


def test_primitive_fc2_space_keeps_exact_nearest_neighbor_translations():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.1,
        max_body_order=None,
        symprec=1e-5,
    )

    images = {image.key for orbit in space.orbits for image in orbit.images}
    expected = {
        InteractionKey((0, 0), ((0, 0, 0),)),
        InteractionKey((0, 0), ((1, 0, 0),)),
        InteractionKey((0, 0), ((-1, 0, 0),)),
        InteractionKey((0, 0), ((0, 1, 0),)),
        InteractionKey((0, 0), ((0, -1, 0),)),
        InteractionKey((0, 0), ((0, 0, 1),)),
        InteractionKey((0, 0), ((0, 0, -1),)),
    }
    assert images == expected


def test_exact_ifcs_realize_into_a_different_supercell_size():
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    source = build_supercell(primitive, (3, 3, 3))
    calculation = ForceConstantCalculation(
        primitive,
        reference=source,
        order=2,
        cutoff=4.1,
        verbose=False,
    )
    result = calculation.reap(
        np.zeros((len(calculation.plan), len(source), 3)), acoustic_sum_rule=False
    )
    target = build_supercell(primitive, (2, 2, 2))
    realized = result.realize(target)

    assert len(result.sparse[2].translations) == 7
    assert realized.materialize(2, max_bytes=None).shape == (1, 8, 3, 3)
