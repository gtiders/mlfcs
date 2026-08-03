import h5py
import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ForceConstantCalculation, SparseOrderForceConstants


def test_reap_keeps_sparse_clusters_and_hdf5_writes_them(tmp_path):
    calculation = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=3,
        supercell=(2, 2, 2),
        cutoff=-1,
    )
    forces = np.zeros((len(calculation.plan), len(calculation.supercell), 3))
    result = calculation.reap(forces)
    assert result.arrays == {}
    assert result.orders == (3,)

    target = tmp_path / "fc.h5"
    result.write(target, format="hdf5")
    with h5py.File(target) as handle:
        group = handle["force_constants/3"]
        assert group.attrs["representation"] == "sparse-cluster"
        assert group["clusters"].shape[1] == 3
        assert group["tensors"].shape[1:] == (3, 3, 3)


def test_dense_materialization_warns_but_continues():
    sparse = SparseOrderForceConstants(
        order=2,
        n_primitive=1,
        n_supercell=1,
        clusters=np.empty((0, 2), dtype=np.int32),
        tensors=np.empty((0, 3, 3)),
    )
    with pytest.warns(RuntimeWarning, match="materialization will continue"):
        dense = sparse.to_dense(max_bytes=1)
    assert dense.shape == (1, 1, 3, 3)
