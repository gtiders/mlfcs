import numpy as np
from ase.build import bulk

from mlfcs.structure.geometry import (
    _unique_distances,
    make_supercell,
    neighbor_shell_limit,
    resolve_cutoff,
)


def test_shells_merge_numerically_split_distances():
    shells = _unique_distances(np.array([0.0, 2.9997637, 2.9997681, 3.66109]))
    assert shells == [2.9997637, 3.66109]


def test_selected_and_maximum_shell_radii_are_reported(capsys):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index)
    selected_radius = resolve_cutoff(supercell, index, -2)
    output = capsys.readouterr().out
    assert f"maximum shell = {maximum_shell}" in output
    assert f"maximum cutoff radius = {maximum_radius:.10f} Å" in output
    assert "Selected neighbor cutoff: shell = 2" in output
    assert f"cutoff radius = {selected_radius:.10f} Å" in output


def test_shell_above_supercell_limit_is_rejected():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    maximum_shell, _ = neighbor_shell_limit(supercell, index)
    with np.testing.assert_raises_regex(ValueError, "exceeds this supercell"):
        resolve_cutoff(supercell, index, -(maximum_shell + 1), report=False)


def test_none_selects_the_supercell_maximum_cutoff(capsys):
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, index = make_supercell(primitive, (2, 2, 2))
    maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index)

    selected = resolve_cutoff(supercell, index, None)

    assert selected == maximum_radius
    output = capsys.readouterr().out
    assert f"maximum shell = {maximum_shell}" in output
    assert "Selected maximum cutoff radius:" in output
