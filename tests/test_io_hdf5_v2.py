import h5py
import numpy as np
import pytest
from ase import Atoms

from mlfcs import read_hdf5 as public_read_hdf5
from mlfcs.api import ForceConstantCalculation
from mlfcs.io.hdf5 import read_hdf5


def test_native_hdf5_v3_roundtrip_preserves_exact_lattice_labelled_sparse_ifcs(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference = primitive.repeat((2, 1, 1))[[1, 0]]
    calculation = ForceConstantCalculation(primitive, reference=reference, order=2, cutoff=3.0)
    result = calculation.reap(np.zeros((len(calculation.plan), len(reference), 3)))
    target = tmp_path / "fc-v3.h5"
    result.write(target, format="hdf5")
    restored = public_read_hdf5(target)

    assert restored.relation is not None
    assert restored.orders == result.orders
    np.testing.assert_array_equal(restored.relation.reference.numbers, primitive.numbers)
    for order in result.orders:
        assert result.sparse[order].is_lattice_labelled
        assert restored.sparse[order].is_lattice_labelled
        np.testing.assert_array_equal(restored.sparse[order].sites, result.sparse[order].sites)
        np.testing.assert_array_equal(
            restored.sparse[order].translations,
            result.sparse[order].translations,
        )
        np.testing.assert_allclose(restored.sparse[order].tensors, result.sparse[order].tensors)
    realized = restored.realize(reference)
    np.testing.assert_allclose(realized.materialize(2), result.materialize(2))


def test_native_hdf5_rejects_v2_schema_without_guessing_atom_semantics(tmp_path):
    source = tmp_path / "legacy.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["format"] = "mlfcs-force-constants"
        handle.attrs["schema_version"] = 2
    with pytest.raises(ValueError, match="only v3 is supported"):
        read_hdf5(source)


def test_native_hdf5_contains_no_source_supercell_mapping(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference = primitive.repeat((2, 1, 1))
    calculation = ForceConstantCalculation(primitive, reference=reference, order=2, cutoff=3.0)
    result = calculation.reap(np.zeros((len(calculation.plan), len(reference), 3)))
    target = tmp_path / "tampered.hdf5"
    result.write(target, format="hdf5")
    with h5py.File(target) as handle:
        assert "reference_mapping" not in handle
        assert "reference_supercell" not in handle["structures"]
