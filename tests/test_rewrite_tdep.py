"""Read TDEP exports with the record layout used by its Fortran readers."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from ase.build import bulk

from mlfcs.cluster_space import build_cluster_space
from mlfcs.core import PrimitiveCell
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.export import clean
from mlfcs.force_constants.lattice import expand
from mlfcs.supercell import ClusterMap, Supercell


def _read_tdep(file, *, order: int, size: int):
    """Mirror TDEP's list-directed FC2/3/4 readers, including the FC2 flag."""
    lines = iter(file.read_text().splitlines())
    assert int(next(lines)) == size
    assert float(next(lines)) > 0.0
    found = {}
    for first in range(size):
        for _ in range(int(next(lines))):
            sites = (first,)
            if order == 2:
                sites += (int(next(lines)) - 1,)
            else:
                sites = tuple(int(next(lines)) - 1 for _ in range(order))
                assert sites[0] == first
            count = order - 1 if order == 2 else order
            vectors = tuple(
                tuple(int(value) for value in next(lines).split()) for _ in range(count)
            )
            assert all(len(vector) == 3 for vector in vectors)
            if order == 2:
                translations = vectors
            else:
                assert vectors[0] == (0, 0, 0)
                translations = vectors[1:]
            tensor = np.empty((3,) * order)
            for directions in product(range(3), repeat=order - 1):
                tensor[directions] = [float(value) for value in next(lines).split()]
            key = sites, translations
            assert key not in found
            found[key] = tensor
    if order == 2:
        assert next(lines) == "0"  # Required nonpolar flag.
    assert list(lines) == []
    return found


@pytest.mark.parametrize("order,cutoff", [(2, 1.1), (3, 1.1), (4, 0.1)])
def test_tdep_export_preserves_primitive_clusters_and_tensors(tmp_path, order, cutoff) -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = build_cluster_space(
        primitive,
        cutoffs={order: cutoff},
        max_body_orders={order: order if order != 4 else 1},
    )
    block = space.block(order)
    model = ForceConstants(
        space, {order: np.linspace(0.5e-8, 2e-7, block.parameters.stop - block.parameters.start)}
    )
    supercell = Supercell.from_atoms(primitive, atoms.repeat((2, 2, 2)), matrix=np.diag([2, 2, 2]))
    mapping = ClusterMap.build(space, supercell)
    file = model.write(tmp_path / f"infile.forceconstant{order}", format="tdep", order=order)
    assert (
        file.read_bytes()
        == model.write(
            tmp_path / f"mapped.forceconstant{order}", mapping, format="tdep", order=order
        ).read_bytes()
    )
    found = _read_tdep(file, order=order, size=primitive.size)

    expanded = expand(model, order)
    expected = {
        (sites, translations): clean(tensor, 1e-8)
        for sites, translations, tensor in zip(
            expanded.sites, expanded.translations, expanded.tensors, strict=True
        )
    }
    assert found.keys() == expected.keys()
    assert len(found) == len(expanded.sites)
    for key, values in found.items():
        np.testing.assert_allclose(values, expected[key], rtol=0, atol=1e-20)
    assert any(np.any(values) for values in found.values())
    assert all(np.all((values == 0.0) | (np.abs(values) >= 1e-8)) for values in found.values())


def test_tdep_rejects_binary_storage(tmp_path) -> None:
    atoms = bulk("Ar", "sc", a=1.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    space = build_cluster_space(primitive, cutoffs={2: 0.1}, max_body_orders={2: 1})
    block = space.block(2)
    model = ForceConstants(space, {2: np.ones(block.parameters.stop - block.parameters.start)})
    supercell = Supercell.from_atoms(primitive, atoms, matrix=np.eye(3, dtype=int))
    mapping = ClusterMap.build(space, supercell)
    with pytest.raises(ValueError, match="text storage"):
        model.write(tmp_path / "invalid.hdf5", mapping, format="tdep", order=2, storage="hdf5")
