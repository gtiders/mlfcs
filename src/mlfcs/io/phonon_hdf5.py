from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from mlfcs.model import ForceConstants


def write_phonon_hdf5(
    target: str | Path,
    force_constants: ForceConstants,
    *,
    order: int,
) -> None:
    """Write full-supercell FC2/FC3 in phonopy/phono3py HDF5 conventions.

    MLFCS keeps a translation-reduced first atomic axis. The external files
    use every supercell atom on every atomic axis and group translated images
    by primitive atom. Slabs are expanded and written one first atom at a time
    so the full FC3 is never materialized in memory.
    """
    if order not in {2, 3}:
        raise ValueError("phonopy/phono3py HDF5 output supports only orders 2 and 3")
    if order not in force_constants.orders:
        raise ValueError(f"order {order} is not present in force constants")

    sparse = force_constants.sparse.get(order)
    compact = sparse.to_dense() if sparse is not None else np.asarray(force_constants.arrays[order])
    supercell = force_constants.supercell
    primitive = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
    translations = np.asarray(supercell.arrays["cell_translation"], dtype=np.int64)
    n_supercell = len(supercell)
    n_primitive = int(primitive.max()) + 1
    expected = (n_primitive,) + (n_supercell,) * (order - 1) + (3,) * order
    if compact.shape != expected:
        raise ValueError(f"compact FC{order} must have shape {expected}, got {compact.shape}")

    repeats = translations.max(axis=0) + 1
    atom_by_key = {
        (int(p), *(int(value) for value in translation)): atom
        for atom, (p, translation) in enumerate(zip(primitive, translations, strict=True))
    }
    grouped = np.lexsort((translations[:, 0], translations[:, 1], translations[:, 2], primitive))
    shape = (n_supercell,) * order + (3,) * order
    chunks = (1,) + shape[1:]
    dataset_name = "force_constants" if order == 2 else "fc3"

    with h5py.File(target, "w") as handle:
        dataset = handle.create_dataset(
            dataset_name,
            shape=shape,
            dtype=compact.dtype,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        for target_first, source_first in enumerate(grouped):
            relative = np.mod(translations - translations[source_first], repeats)
            anchored = np.fromiter(
                (
                    atom_by_key[(int(p), *(int(value) for value in translation))]
                    for p, translation in zip(primitive, relative, strict=True)
                ),
                dtype=np.int64,
                count=n_supercell,
            )
            tails = anchored[grouped]
            if order == 2:
                slab = compact[int(primitive[source_first]), tails]
            else:
                slab = compact[int(primitive[source_first])][np.ix_(tails, tails)]
            dataset[target_first] = slab

        n_cells = n_supercell // n_primitive
        handle.create_dataset("p2s_map", data=np.arange(n_primitive, dtype=np.int64) * n_cells)
        try:
            release = version("mlfcs")
        except PackageNotFoundError:
            release = "unknown"
        handle.create_dataset("version", data=np.bytes_(f"mlfcs {release}"))
        if order == 2:
            handle.create_dataset("physical_unit", data=np.asarray([b"eV/angstrom^2"]))


__all__ = ["write_phonon_hdf5"]
