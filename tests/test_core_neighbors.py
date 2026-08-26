import numpy as np
from ase.build import bulk
from ase.geometry import minkowski_reduce

from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
from mlfcs.structure.periodic_geometry import unique_periodic_distances


def test_shells_merge_numerically_split_distances():
    shells = unique_periodic_distances(np.array([0.0, 2.9997637, 2.9997681, 3.66109]))
    assert shells == [2.9997637, 3.66109]


def test_negative_cutoff_selects_a_primitive_neighbor_shell():
    primitive = bulk("Si", "diamond", a=5.43)
    first = resolve_primitive_cutoff(primitive, -1)
    second = resolve_primitive_cutoff(primitive, -2)
    assert second > first > 0


def test_none_cutoff_selects_maximum_unambiguous_reference_radius():
    primitive = bulk("Si", "diamond", a=5.43)
    reference = primitive.repeat((2, 2, 2))
    cutoff = resolve_primitive_cutoff(primitive, None, reference=reference)
    assert cutoff > resolve_primitive_cutoff(primitive, -1)
    assert cutoff < min(np.linalg.norm(reference.cell, axis=1))
    with np.testing.assert_raises_regex(ValueError, "explicit reference"):
        resolve_primitive_cutoff(primitive, None)


def test_none_cutoff_uses_shortest_translation_of_skew_reference():
    primitive = bulk("Si", "diamond", a=5.43)
    reference = primitive.repeat((2, 2, 2))
    expected = resolve_primitive_cutoff(primitive, None, reference=reference)
    reference.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 0]]) @ np.asarray(reference.cell),
        scale_atoms=False,
    )
    reduced_cell, _operation = minkowski_reduce(reference.cell, pbc=reference.pbc)
    assert np.isclose(resolve_primitive_cutoff(primitive, None, reference=reference), expected)
    assert expected < float(np.min(np.linalg.norm(reduced_cell, axis=1))) - 0.01
