"""Version-five HDF5 storage of primitive models and physical coefficients."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from mlfcs.cluster_space.model import Cluster, ClusterSpace, Orbit, OrderBlock
from mlfcs.foundation.arrays import readonly
from mlfcs.geometry.primitive import validate_primitive_arrays
from mlfcs.geometry.symmetry import PrimitiveSymmetry

_FORMAT = "mlfcs.force_constants"
_VERSION = 5
_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _array(group, name, kind, ndim):
    """Read a typed dataset without coercing invalid stored values.

    ``group`` is an HDF5 group, ``name`` a required dataset, ``kind`` is
    'i' for signed int64 or 'f' for float64, and ``ndim`` is its required
    rank. Return a NumPy array preserving shape and entries. Raise ValueError
    for missing datasets or incompatible dtype/rank; floating entries must be
    finite. Mathematical shape constraints are checked by the model constructors.
    """
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"{name} must be a dataset")
    if dataset.dtype.kind != kind or dataset.dtype.itemsize != 8 or dataset.ndim != ndim:
        raise ValueError(
            f"{name} must be a rank-{ndim} {'int64' if kind == 'i' else 'float64'} dataset"
        )
    values = dataset[...]
    if kind == "f" and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def _cluster(labels):
    """Decode anchored int64 labels of shape (p, 4) into a tensor-index cluster.

    Each row is (primitive site, tx, ty, tz), where p >= 2 and the first
    translation is zero. Raise ValueError for malformed or unanchored labels.
    """
    if labels.ndim != 2 or labels.shape[1:] != (4,) or len(labels) < 2:
        raise ValueError("cluster labels must have shape (order >= 2, 4)")
    if np.any(labels[0, 1:] != 0):
        raise ValueError("stored cluster labels must be anchored")
    return Cluster.from_labels(labels)


def _write_space(group, space):
    """Write a ClusterSpace snapshot, including bases and masses, without object encoding.

    ``group`` is an empty HDF5 group. Lattice vectors are rows in angstrom;
    positions are wrapped fractional rows, masses are in atomic mass units,
    tensor bases use C-order Cartesian components, and labels are integer
    lattice addresses. All ndarray datasets are int64 or float64.
    """
    primitive = group.create_group("primitive")
    for name in ("cell", "scaled_positions", "atomic_numbers", "masses"):
        primitive.create_dataset(name, data=getattr(space, name))
    primitive.attrs["symprec"] = space.symprec
    symmetry = group.create_group("symmetry")
    for name in (
        "rotations",
        "translations",
        "cartesian_rotations",
        "site_permutations",
        "site_shifts",
    ):
        symmetry.create_dataset(name, data=getattr(space.symmetry, name))
    symmetry.attrs["symbol"] = space.symmetry.symbol
    symmetry.attrs["symprec"] = space.symmetry.symprec
    blocks = group.create_group("blocks")
    for index, block in enumerate(space.blocks):
        entry = blocks.create_group(str(index))
        for name in ("order", "cutoff", "max_body_order"):
            entry.attrs[name] = getattr(block, name)
        for name in ("orbits", "parameters"):
            interval = getattr(block, name)
            entry.create_dataset(
                name, data=np.array([interval.start, interval.stop], dtype=np.int64)
            )
    orbits = group.create_group("orbits")
    for index, orbit in enumerate(space.orbits):
        entry = orbits.create_group(str(index))
        entry.create_dataset(
            "representative", data=np.array(orbit.representative.labels, dtype=np.int64)
        )
        entry.create_dataset(
            "clusters",
            data=np.array([cluster.labels for cluster in orbit.clusters], dtype=np.int64).reshape(
                -1, orbit.representative.order, 4
            ),
        )
        for name in (
            "lattice_basis",
            "component_basis",
            "observation_rows",
            "operations",
            "permutations",
        ):
            entry.create_dataset(name, data=getattr(orbit, name))
        entry.attrs["observation_condition"] = orbit.observation_condition


def _read_space(group):
    """Restore a validated ClusterSpace from explicit HDF5 model datasets.

    Data follow the units, shapes and tensor-index conventions of ``_write_space``.
    Returns an immutable model without neighbor enumeration, symmetry discovery
    or integer-kernel solving. Primitive validation still certifies the motif
    through spglib. Raise ValueError for invalid geometry, masses or layout.
    """
    primitive = group["primitive"]
    cell, positions, numbers, symprec = validate_primitive_arrays(
        _array(primitive, "cell", "f", 2),
        _array(primitive, "scaled_positions", "f", 2),
        _array(primitive, "atomic_numbers", "i", 1),
        primitive.attrs["symprec"],
    )
    masses = _array(primitive, "masses", "f", 1)
    if masses.shape != numbers.shape or np.any(masses <= 0):
        raise ValueError("stored atomic masses must be positive and match the motif")
    source = group["symmetry"]
    symmetry = PrimitiveSymmetry(
        rotations=_array(source, "rotations", "i", 3),
        translations=_array(source, "translations", "f", 2),
        cartesian_rotations=_array(source, "cartesian_rotations", "f", 3),
        site_permutations=_array(source, "site_permutations", "i", 2),
        site_shifts=_array(source, "site_shifts", "i", 3),
        symbol=source.attrs["symbol"],
        symprec=source.attrs["symprec"],
    )
    if symmetry.symprec != symprec or symmetry.site_permutations.shape[1] != len(numbers):
        raise ValueError("stored symmetry does not act on the declared primitive motif")
    blocks = []
    source = group["blocks"]
    for index in range(len(source)):
        entry = source[str(index)]
        intervals = []
        for name in ("orbits", "parameters"):
            bounds = _array(entry, name, "i", 1)
            if bounds.shape != (2,):
                raise ValueError(f"{name} must contain two slice endpoints")
            intervals.append(slice(int(bounds[0]), int(bounds[1])))
        blocks.append(
            OrderBlock(
                entry.attrs["order"],
                entry.attrs["cutoff"],
                entry.attrs["max_body_order"],
                *intervals,
            )
        )
    orbits = []
    source = group["orbits"]
    for index in range(len(source)):
        entry = source[str(index)]
        representative = _cluster(_array(entry, "representative", "i", 2))
        images = _array(entry, "clusters", "i", 3)
        if images.shape[1:] != (representative.order, 4):
            raise ValueError("stored image labels differ from the representative order")
        orbits.append(
            Orbit(
                representative=representative,
                lattice_basis=_array(entry, "lattice_basis", "i", 2),
                component_basis=_array(entry, "component_basis", "f", 2),
                observation_rows=_array(entry, "observation_rows", "i", 1),
                observation_condition=entry.attrs["observation_condition"],
                clusters=tuple(_cluster(labels) for labels in images),
                operations=_array(entry, "operations", "i", 1),
                permutations=_array(entry, "permutations", "i", 2),
            )
        )
    space = object.__new__(ClusterSpace)
    for name, value in (
        ("cell", cell),
        ("scaled_positions", positions),
        ("atomic_numbers", numbers),
        ("symprec", symprec),
        ("_masses", readonly(masses, np.float64)),
        ("symmetry", symmetry),
        ("blocks", tuple(blocks)),
        ("orbits", tuple(orbits)),
    ):
        object.__setattr__(space, name, value)
    space.__post_init__()
    return space


def save(model, path):
    """Atomically save a ForceConstants model to a version-five HDF5 file.

    ``path`` is a path-like target; return its absolute Path. Store all primitive
    model arrays and each present physical coefficient vector (eV/angstrom**p).
    Flush and fsync a same-directory temporary file before replacing the target.
    I/O errors propagate and the temporary file is removed on failure.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = _FORMAT
            handle.attrs["version"] = _VERSION
            _write_space(handle.create_group("cluster_space"), model.cluster_space)
            coefficients = handle.create_group("coefficients")
            for order, values in model.coefficients.items():
                coefficients.create_dataset(str(order), data=values)
            handle.flush()
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load(path):
    """Load a version-five HDF5 ForceConstants model without executable object decoding.

    ``path`` is a path-like input. Return a model with readonly primitive arrays,
    masses and physical coefficients. Raise ValueError for unsupported formats,
    missing fields or invalid stored model data. File-access errors propagate.
    Old pickle-native files are rejected; no legacy decoder is invoked.
    """
    from mlfcs.force_constants.model import ForceConstants

    with open(path, "rb") as handle:
        if handle.read(8) != _SIGNATURE:
            raise ValueError("unsupported force-constant file; HDF5 version 5 is required")
    with h5py.File(path, "r") as handle:
        try:
            version = handle.attrs["version"]
            if (
                handle.attrs["format"] != _FORMAT
                or not isinstance(version, (int, np.integer))
                or version != _VERSION
            ):
                raise ValueError("unsupported force-constant format; HDF5 version 5 is required")
            space = _read_space(handle["cluster_space"])
            source = handle["coefficients"]
            coefficients = {}
            for name in source:
                order = int(name)
                if str(order) != name:
                    raise ValueError("coefficient dataset names must be canonical integer orders")
                coefficients[order] = _array(source, name, "f", 1)
            return ForceConstants(space, coefficients)
        except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as error:
            raise ValueError(f"invalid native force-constant model: {error}") from error
