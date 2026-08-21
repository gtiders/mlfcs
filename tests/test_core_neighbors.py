import numpy as np
from ase.build import bulk

from mlfcs.core.geometry import _unique_distances
from mlfcs.core.real_space import resolve_primitive_cutoff


def test_shells_merge_numerically_split_distances():
    shells = _unique_distances(np.array([0.0, 2.9997637, 2.9997681, 3.66109]))
    assert shells == [2.9997637, 3.66109]


def test_negative_cutoff_selects_a_primitive_neighbor_shell():
    primitive = bulk("Si", "diamond", a=5.43)
    first = resolve_primitive_cutoff(primitive, -1)
    second = resolve_primitive_cutoff(primitive, -2)
    assert second > first > 0


def test_none_cutoff_is_rejected():
    primitive = bulk("Si", "diamond", a=5.43)
    with np.testing.assert_raises_regex(ValueError, "cutoff must be explicit"):
        resolve_primitive_cutoff(primitive, None)
