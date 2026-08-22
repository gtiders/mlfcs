from importlib.metadata import version

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from phono3py.file_IO import read_fc3_from_hdf5
from phonopy.file_IO import read_force_constants_hdf5
from supercell_helpers import make_supercell

from mlfcs import (
    ForceConstantCalculation,
    ForceConstants,
    SparseOrderForceConstants,
    write_force_constants,
)
from mlfcs.structure.relation import StructureRelation


def test_reap_keeps_sparse_clusters_and_hdf5_writes_them(tmp_path):
    primitive = bulk("Si", "diamond", a=5.43)
    calculation = ForceConstantCalculation(
        primitive,
        order=3,
        reference=make_supercell(primitive, (2, 2, 2))[0],
        cutoff=-1,
    )
    forces = np.zeros((len(calculation.plan), len(calculation.supercell), 3))
    result = calculation.reap(forces)
    assert result.arrays == {}
    assert result.orders == (3,)

    target = tmp_path / "fc.h5"
    write_force_constants(result, target, format="hdf5")
    with h5py.File(target) as handle:
        group = handle["force_constants/3"]
        assert handle.attrs["schema_version"] == 3
        assert group.attrs["representation"] == "lattice-labelled-sparse"
        assert group["sites"].shape[1] == 3
        assert group["translations"].shape[1:] == (2, 3)
        assert group["tensors"].shape[1:] == (3, 3, 3)


def test_dense_materialization_warns_but_continues():
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    relation = StructureRelation.from_atoms(primitive, primitive)
    sparse = SparseOrderForceConstants(
        order=2,
        sites=np.empty((0, 2), dtype=np.int32),
        translations=np.empty((0, 1, 3), dtype=np.int32),
        tensors=np.empty((0, 3, 3)),
    )
    with pytest.warns(RuntimeWarning, match="materialization will continue"):
        dense = ForceConstants({}, primitive, sparse={2: sparse}, relation=relation).materialize(
            2, max_bytes=1
        )
    assert dense.shape == (1, 1, 3, 3)


def test_phonopy_hdf5_is_readable_and_preserves_reference_order(tmp_path):
    primitive = Atoms(
        "NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    compact = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3) / 7
    result = ForceConstants({2: compact}, supercell)
    target = tmp_path / "fc2.hdf5"

    write_force_constants(result, target, format="phonopy_hdf5")
    full, unit = read_force_constants_hdf5(target, return_physical_unit=True)

    assert full.shape == (4, 4, 3, 3)
    assert unit == "eV/angstrom^2"
    np.testing.assert_allclose(full[1, 2], compact[0, 3], atol=0, rtol=0)
    with h5py.File(target) as handle:
        np.testing.assert_array_equal(handle["p2s_map"], [0, 2])
        assert handle["version"][()].decode() == f"mlfcs {version('mlfcs')}"


def test_phono3py_hdf5_streams_full_fc3_and_is_readable(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    compact = np.arange(1 * 2 * 2 * 27, dtype=float).reshape(1, 2, 2, 3, 3, 3)
    clusters = np.asarray([[0, first, second] for first in range(2) for second in range(2)])
    sparse = SparseOrderForceConstants(
        order=3,
        sites=np.zeros((4, 3), dtype=np.int32),
        translations=np.asarray(
            [[[first, 0, 0], [second, 0, 0]] for first in range(2) for second in range(2)]
        ),
        tensors=compact[tuple(clusters.T)],
    )
    result = ForceConstants(
        {},
        supercell,
        sparse={3: sparse},
        relation=StructureRelation.from_atoms(primitive, supercell),
    )
    target = tmp_path / "fc3.hdf5"

    write_force_constants(result, target, format="phono3py_hdf5")
    full = read_fc3_from_hdf5(target)

    assert full.shape == (2, 2, 2, 3, 3, 3)
    np.testing.assert_allclose(full[0], compact[0], atol=0, rtol=0)
    np.testing.assert_allclose(full[1, 0, 1], compact[0, 1, 0], atol=0, rtol=0)
    with h5py.File(target) as handle:
        np.testing.assert_array_equal(handle["p2s_map"], [0])


def test_external_hdf5_rejects_wrong_order(tmp_path):
    primitive = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    supercell, _ = make_supercell(primitive, (1, 1, 1))
    result = ForceConstants({3: np.zeros((1, 1, 1, 3, 3, 3))}, supercell)
    with pytest.raises(ValueError, match="only for order 2"):
        write_force_constants(result, tmp_path / "fc2.hdf5", format="phonopy_hdf5")
