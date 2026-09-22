import numpy as np
from ase.build import bulk

from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
from mlfcs.structure.periodic_geometry import unique_periodic_distances


def test_shells_merge_numerically_split_distances():
    shells = unique_periodic_distances(np.array([0.0, 2.9997637, 2.9997681, 3.66109]), symprec=1e-5)
    assert shells == [2.9997637, 3.66109]


def test_shell_identity_uses_one_absolute_angstrom_precision():
    values = np.array([0.0, 3.0, 3.0 + 0.5e-5, 3.0 + 1.5e-5])
    assert unique_periodic_distances(values, symprec=1e-5) == [3.0, 3.0 + 1.5e-5]


def test_shell_precision_must_be_physical():
    with np.testing.assert_raises_regex(ValueError, "finite positive"):
        unique_periodic_distances(np.array([1.0]), symprec=0.0)

    with np.testing.assert_raises_regex(ValueError, "finite positive"):
        resolve_primitive_cutoff(bulk("Si", "diamond", a=5.43), 3.0, symprec=np.nan)


def test_negative_cutoff_selects_a_primitive_neighbor_shell():
    primitive = bulk("Si", "diamond", a=5.43)
    first = resolve_primitive_cutoff(primitive, -1, symprec=1e-5)
    second = resolve_primitive_cutoff(primitive, -2, symprec=1e-5)
    assert second > first > 0


def test_none_cutoff_is_rejected():
    """A radius belongs to the primitive model, so `None` is never accepted."""
    primitive = bulk("Si", "diamond", a=5.43)

    with np.testing.assert_raises_regex(ValueError, "negative neighbour-shell"):
        resolve_primitive_cutoff(primitive, None, symprec=1e-5)
