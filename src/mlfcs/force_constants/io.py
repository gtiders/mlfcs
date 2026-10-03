"""External force-constant formats and their writers."""

from __future__ import annotations

import os
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path

import h5py
import numpy as np

from mlfcs.force_constants.export import (
    DEFAULT_THRESHOLD,
    clean,
    compact,
    full_fc2,
    primitive_to_supercell,
    threshold_value,
    translated_atoms,
    validate,
)
from mlfcs.force_constants.lattice import expand
from mlfcs.force_constants.model import ForceConstants
from mlfcs.mapping import ClusterMap


def write(
    model: ForceConstants,
    file: str | os.PathLike[str],
    cluster_map: ClusterMap | None,
    *,
    format: str,
    order: int,
    storage: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Write one explicitly selected order in one supported external format."""
    path = Path(file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = threshold_value(threshold)
    if format == "phonopy":
        if order != 2:
            raise ValueError("phonopy output supports only order 2")
        selected_storage = "text" if storage is None else storage
        if selected_storage not in {"text", "hdf5"}:
            raise ValueError("phonopy storage must be 'text' or 'hdf5'")
        validate(model, cluster_map, order)
        _write_phonopy(path, model, cluster_map, storage=selected_storage, threshold=threshold)
    elif format == "phono3py":
        if order != 3:
            raise ValueError("phono3py output supports only order 3")
        if storage not in {None, "hdf5"}:
            raise ValueError("phono3py output uses HDF5 storage")
        validate(model, cluster_map, order)
        _write_phono3py(path, model, cluster_map, threshold=threshold)
    elif format == "shengbte":
        if order not in {3, 4}:
            raise ValueError("ShengBTE output supports only orders 3 and 4")
        if storage not in {None, "text"}:
            raise ValueError("ShengBTE output uses text storage")
        validate(model, cluster_map, order)
        _write_shengbte(path, model, cluster_map, order=order, threshold=threshold)
    elif format == "tdep":
        if order not in {2, 3, 4}:
            raise ValueError("TDEP output supports only orders 2, 3 and 4")
        if storage not in {None, "text"}:
            raise ValueError("TDEP output uses text storage")
        if order not in model.coefficients:
            raise ValueError(f"force constants do not contain order {order}")
        if cluster_map is not None:
            validate(model, cluster_map, order)
        _write_tdep(path, model, order=order, threshold=threshold)
    else:
        raise ValueError(
            f"unsupported force-constant format {format!r}; supported formats are "
            "phonopy, phono3py, shengbte, and tdep"
        )
    return path


def _write_hdf5(
    path: Path,
    model: ForceConstants,
    cluster_map: ClusterMap,
    *,
    order: int,
    threshold: float,
) -> None:
    """Write full-supercell FC2 or FC3 in phonon HDF5 conventions."""
    values = compact(model, cluster_map, order, threshold=threshold)
    size = len(cluster_map.atomic_numbers)
    shape = (size,) * order + (3,) * order
    name = "force_constants" if order == 2 else "fc3"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            name,
            shape=shape,
            dtype=np.float64,
            chunks=(1,) + shape[1:],
            compression="gzip",
            compression_opts=4,
        )
        for first in range(size):
            tails = translated_atoms(cluster_map, first)
            primitive = int(cluster_map.primitive_site_indices[first])
            if order == 2:
                dataset[first] = values[primitive, tails]
            else:
                dataset[first] = values[primitive][np.ix_(tails, tails)]
        handle.create_dataset("p2s_map", data=primitive_to_supercell(cluster_map))
        try:
            release = version("mlfcs")
        except PackageNotFoundError:
            release = "unknown"
        handle.create_dataset("version", data=np.bytes_(f"mlfcs {release}"))
        if order == 2:
            handle.create_dataset("physical_unit", data=np.asarray([b"eV/angstrom^2"]))
        handle.attrs["mlfcs_threshold"] = threshold


def _write_phonopy(
    path: Path,
    model: ForceConstants,
    cluster_map: ClusterMap,
    *,
    storage: str,
    threshold: float,
) -> None:
    """Write FC2 as phonopy text or HDF5 in explicit supercell order."""
    if storage == "hdf5":
        _write_hdf5(path, model, cluster_map, order=2, threshold=threshold)
        return
    values = full_fc2(compact(model, cluster_map, 2, threshold=threshold), cluster_map)
    size = len(cluster_map.atomic_numbers)
    lines = [f"{size:4d} {size:4d}"]
    for first in range(size):
        for second in range(size):
            lines.append(f"{first + 1:d} {second + 1:d}")
            lines.extend(("%22.15f" * 3) % tuple(row) for row in values[first, second])
    path.write_text("\n".join(lines) + "\n")


def _write_phono3py(
    path: Path,
    model: ForceConstants,
    cluster_map: ClusterMap,
    *,
    threshold: float,
) -> None:
    """Write full-supercell FC3 in phono3py HDF5 convention."""
    _write_hdf5(path, model, cluster_map, order=3, threshold=threshold)


def _vector_line(vector: np.ndarray) -> str:
    return " ".join(f"{value:>15.10e}" for value in vector)


def _write_shengbte(
    path: Path,
    model: ForceConstants,
    cluster_map: ClusterMap,
    *,
    order: int,
    threshold: float,
) -> None:
    """Write nonzero exact-lattice FC3 or FC4 blocks in ShengBTE style."""
    validate(model, cluster_map, order)
    expanded = expand(model, order)
    physical: dict[
        tuple[int, ...], tuple[tuple[int, ...], tuple[tuple[int, int, int], ...], np.ndarray]
    ] = {}
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        key = (*sites, *(value for translation in translations for value in translation))
        if key in physical:
            raise RuntimeError("cluster-space expansion produced a duplicate lattice cluster")
        physical[key] = (sites, translations, np.array(tensor, copy=True))

    blocks = []
    primitive_cell = model.cluster_space.cell
    for key in sorted(physical):
        sites, translations, tensor = physical[key]
        tensor = clean(tensor, threshold)
        if not np.any(tensor):
            continue
        lines = ["", f"{len(blocks) + 1:>5}"]
        lines.extend(
            _vector_line(np.asarray(translation) @ primitive_cell) for translation in translations
        )
        lines.append(" ".join(f"{site + 1:>6d}" for site in sites))
        for directions in product(range(3), repeat=order):
            direction_text = " ".join(f"{direction + 1:>2d}" for direction in directions)
            lines.append(f"{direction_text} {tensor[directions]:>20.10e}")
        blocks.append("\n".join(lines) + "\n")
    path.write_text(f"{len(blocks):>5}\n" + "".join(blocks))


def _write_tdep(path: Path, model: ForceConstants, *, order: int, threshold: float) -> None:
    """Write TDEP's per-primitive-atom list of Cartesian energy derivatives.

    Indices are one-based; lattice vectors are integer coefficients of the
    primitive cell. TDEP's FC2 reader additionally requires a polar flag, so
    this writer marks FC2 as nonpolar (no Born-charge correction is exported).
    """
    primitive = model.cluster_space
    expanded = expand(model, order)
    grouped: dict[
        int, list[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...], np.ndarray]]
    ] = defaultdict(list)
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = set()
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        key = (sites, translations)
        if key in seen:
            raise RuntimeError("cluster-space expansion produced a duplicate lattice cluster")
        seen.add(key)
        grouped[sites[0]].append((sites, translations, clean(tensor, threshold)))

    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{primitive.n_atoms}\n{model.cluster_space.block(order).cutoff:.17g}\n")
        for first in range(primitive.n_atoms):
            blocks = sorted(grouped[first], key=lambda block: (block[0], block[1]))
            handle.write(f"{len(blocks)}\n")
            for sites, translations, tensor in blocks:
                if order == 2:
                    handle.write(f"{sites[1] + 1}\n")
                    vectors = translations
                else:
                    for site in sites:
                        handle.write(f"{site + 1}\n")
                    vectors = ((0, 0, 0), *translations)
                for vector in vectors:
                    handle.write(" ".join(str(int(value)) for value in vector) + "\n")
                for directions in product(range(3), repeat=order - 1):
                    handle.write(" ".join(f"{value:.17e}" for value in tensor[directions]) + "\n")
        if order == 2:
            handle.write("0\n")


__all__ = ["write"]
