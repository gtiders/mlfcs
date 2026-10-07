"""Stream Cartesian force-constant blocks into external phonon file formats."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from numba import njit

from mlfcs.foundation.log import get_logger
from mlfcs.geometry.primitive import LatticeSite

logger = get_logger(__name__)
DEFAULT_THRESHOLD = 1e-8


def threshold_value(value):
    """Require a finite nonnegative cleanup threshold in the selected order's units."""
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")
    return threshold


def threshold_components(values, threshold):
    """Copy finite tensors and zero components below the physical output threshold."""
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    _clean_components(result.reshape(-1), threshold_value(threshold))
    return result


@njit(cache=True)
def _clean_components(values, threshold):
    """Validate and clean a private output buffer without full-sized temporary masks."""
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            raise ValueError("expanded force constants contain NaN or infinite values")
        if abs(values[index]) < threshold:
            values[index] = 0


def write(tensors, file, *, format, order, storage=None, threshold=DEFAULT_THRESHOLD):
    """Write a declared tensor order, rejecting formats lacking the required information."""
    if order not in tensors.orders:
        raise ValueError(f"force constants do not contain order {order}")
    threshold = threshold_value(threshold)
    if format == "phonopy":
        if order != 2 or storage not in {None, "text", "hdf5"}:
            raise ValueError("phonopy supports FC2 with text or hdf5 storage")
    elif format == "phono3py":
        if order != 3 or storage not in {None, "hdf5"}:
            raise ValueError("phono3py supports FC3 with hdf5 storage")
    elif format == "shengbte":
        if order not in {3, 4} or storage not in {None, "text"}:
            raise ValueError("ShengBTE supports FC3/FC4 with text storage")
    else:
        raise ValueError("supported force-constant formats are phonopy, phono3py, and shengbte")
    if format in {"phonopy", "phono3py"} and tensors.cluster_map is None:
        raise ValueError("periodic tensor export requires a ClusterMap")
    path = Path(file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    logger.info(
        "Export started: path=%s format=%s order=%d storage=%s threshold=%.6g",
        path,
        format,
        order,
        storage or "default",
        threshold,
    )
    if format == "phonopy" and storage != "hdf5":
        _write_phonopy_text(path, tensors, threshold)
    elif format in {"phonopy", "phono3py"}:
        _write_hdf5(path, tensors, order, threshold)
    else:
        _write_shengbte(path, tensors, order, threshold)
    logger.info(
        "Export complete: path=%s format=%s order=%d elapsed_s=%.2f",
        path,
        format,
        order,
        perf_counter() - started,
    )
    return path


def _write_hdf5(path, tensors, order, threshold):
    """Write full-supercell HDF5 slices without collecting a full or compact array."""
    mapping = tensors.cluster_map
    n = mapping.n_atoms
    shape = (n,) * order + (3,) * order
    blocks = tensors.full_blocks(order)
    slices, block = next(blocks)
    chunks = block.shape
    name = "force_constants" if order == 2 else "fc3"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            name,
            shape=shape,
            dtype=np.float64,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        dataset[slices] = threshold_components(block, threshold)
        del block
        for slices, block in blocks:
            dataset[slices] = threshold_components(block, threshold)
            del block
        anchors = [
            mapping.atom_index(LatticeSite(i, (0, 0, 0)))
            for i in range(tensors.cluster_space.n_atoms)
        ]
        handle.create_dataset("p2s_map", data=np.asarray(anchors, dtype=np.int64))
        try:
            release = version("mlfcs")
        except PackageNotFoundError:
            release = "unknown"
        handle.create_dataset("version", data=np.bytes_(f"mlfcs {release}"))
        if order == 2:
            handle.create_dataset("physical_unit", data=np.asarray([b"eV/angstrom^2"]))
        handle.attrs["mlfcs_threshold"] = threshold


def _write_phonopy_text(path, tensors, threshold):
    """Write full FC2 rows in supercell order, including zero Cartesian blocks."""
    n = tensors.cluster_map.n_atoms
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{n:4d} {n:4d}\n")
        for slices, block in tensors.full_blocks(2):
            cleaned = threshold_components(block, threshold)
            first = slices[0].start
            for local, second in enumerate(range(slices[1].start, slices[1].stop)):
                handle.write(f"{first + 1:d} {second + 1:d}\n")
                for row in cleaned[0, local]:
                    handle.write(("%22.15f" * 3) % tuple(row) + "\n")
            del block, cleaned, row


def _lattice_records(tensors, order, threshold):
    """Yield canonical lattice labels and cleaned tensors in primitive-site order."""
    for sites, translations, values in tensors.lattice_blocks(order):
        for index in range(len(values)):
            yield sites[index], translations[index], threshold_components(values[index], threshold)
        del values


def _write_shengbte(path, tensors, order, threshold):
    """Count and stream nonzero anchored FC3/FC4 blocks with Cartesian translations."""
    count = sum(np.any(value) for _, _, value in _lattice_records(tensors, order, threshold))
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{count:>5}\n")
        number = 0
        for sites, translations, value in _lattice_records(tensors, order, threshold):
            if not np.any(value):
                continue
            number += 1
            handle.write(f"\n{number:>5}\n")
            for translation in translations:
                vector = translation @ tensors.cluster_space.cell
                handle.write(" ".join(f"{entry:>15.10e}" for entry in vector) + "\n")
            handle.write(" ".join(f"{site + 1:>6d}" for site in sites) + "\n")
            for directions in product(range(3), repeat=order):
                text = " ".join(f"{direction + 1:>2d}" for direction in directions)
                handle.write(f"{text} {value[directions]:>20.10e}\n")


__all__ = ["write"]
