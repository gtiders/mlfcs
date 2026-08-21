import numpy as np
import pytest
from ase import Atoms
from supercell_helpers import make_supercell

from mlfcs.api import ForceConstantCalculation
from mlfcs.io.export import build_export_view
from mlfcs.io.hdf5 import read_hdf5


def _result():
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference = primitive.repeat((2, 1, 1))[[1, 0]]
    calculation = ForceConstantCalculation(primitive, reference=reference, order=2, cutoff=3.0)
    return calculation.reap(np.zeros((len(calculation.plan), len(reference), 3)))


def test_export_view_relabels_an_equivalent_reordered_reference(tmp_path):
    result = _result()
    target = result.supercell[[1, 0]]
    view = build_export_view(result, supercell=target)

    np.testing.assert_array_equal(view.force_constants.supercell.numbers, target.numbers)
    assert view.relation is not None
    assert view.force_constants.sparse[2].sites.shape == result.sparse[2].sites.shape
    output = tmp_path / "relabelled.h5"
    result.write(output, format="hdf5", supercell=target)
    restored = read_hdf5(output)
    np.testing.assert_array_equal(restored.supercell.numbers, result.relation.primitive.numbers)
    np.testing.assert_allclose(
        restored.realize(target).materialize(2),
        view.force_constants.materialize(2),
    )
    result.write(tmp_path / "FORCE_CONSTANTS", format="phonopy", supercell=target)
    assert (tmp_path / "FORCE_CONSTANTS").is_file()
    result.write(tmp_path / "phonopy.hdf5", format="phonopy_hdf5", supercell=target)
    assert (tmp_path / "phonopy.hdf5").is_file()


def test_export_view_reuses_cached_source_and_target_views(capsys):
    result = _result()
    source_first = build_export_view(result)
    source_second = build_export_view(result)
    target = result.supercell[[1, 0]]
    target_first = build_export_view(result, supercell=target)
    target_second = build_export_view(result, supercell=target)

    assert source_first is source_second
    assert target_first is target_second
    assert target_first is not source_first
    messages = capsys.readouterr().out
    assert messages.count("cache miss; constructing new view") == 2
    assert messages.count("cache hit; reusing existing view") == 2


def test_export_view_realizes_into_a_different_supercell_translation_lattice(tmp_path):
    result = _result()
    target = result.relation.primitive.repeat((1, 2, 1))
    realized = result.realize(target)

    np.testing.assert_array_equal(realized.supercell.cell, target.cell)
    np.testing.assert_array_equal(realized.sparse[2].sites, result.sparse[2].sites)
    np.testing.assert_array_equal(
        realized.sparse[2].translations,
        result.sparse[2].translations,
    )
    result.write(tmp_path / "realized.h5", format="hdf5", supercell=target)
    assert (tmp_path / "realized.h5").is_file()


def test_export_view_accepts_unimodular_primitive_basis_change():
    result = _result()
    source = result.relation.primitive
    change = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    target_primitive = source.copy()
    target_primitive.set_cell(change @ source.cell, scale_atoms=False)

    view = build_export_view(result, primitive=target_primitive)
    assert view.relation is not None
    np.testing.assert_array_equal(view.relation.primitive.cell, target_primitive.cell)


def test_export_view_roundtrips_when_primitive_basis_and_reference_are_both_represented_anew():
    result = _result()
    source_primitive = result.relation.primitive
    change = np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    target_primitive = source_primitive.copy()
    target_primitive.set_cell(change @ source_primitive.cell, scale_atoms=False)
    target_matrix = result.relation.supercell_matrix @ np.linalg.inv(change)
    target_supercell, _ = make_supercell(target_primitive, target_matrix)
    target_supercell = target_supercell[[1, 0]]

    target_view = build_export_view(
        result,
        primitive=target_primitive,
        supercell=target_supercell,
    )
    returned = build_export_view(
        target_view.force_constants,
        primitive=source_primitive,
        supercell=result.supercell,
    ).force_constants.sparse[2]
    source = result.sparse[2]
    np.testing.assert_array_equal(returned.sites, source.sites)
    np.testing.assert_array_equal(
        returned.translations, source.translations
    )
    np.testing.assert_allclose(returned.tensors, source.tensors)


def test_export_view_rejects_cartesian_rotation():
    primitive = Atoms(
        "NaCl", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]], cell=np.eye(3) * 4, pbc=True
    )
    reference = primitive.repeat((2, 1, 1))
    calculation = ForceConstantCalculation(primitive, reference=reference, order=2, cutoff=3.0)
    result = calculation.reap(np.zeros((len(calculation.plan), len(reference), 3)))
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = primitive.copy()
    rotated.set_cell(rotation @ primitive.cell, scale_atoms=False)
    rotated.positions = primitive.positions @ rotation.T

    with pytest.raises(ValueError, match="cannot be mapped|equivalent representation"):
        build_export_view(result, primitive=rotated)
