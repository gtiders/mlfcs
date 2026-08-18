from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs.ifc.model import SparseOrderForceConstants
from mlfcs.io._text import zero_small_scalar
from mlfcs.structure.geometry import PeriodicGeometry, PeriodicIndex

_TEXT_ZERO_TOLERANCE = 1e-8


def write_shengbte(
    target: str | Path,
    force_constants: SparseOrderForceConstants,
    supercell: Atoms,
) -> None:
    """Write an order-parameterized ShengBTE-style force-constant file.

    Atomic axes precede Cartesian axes. A block contains ``order - 1``
    lattice translations, ``order`` primitive atom indices, and ``3**order``
    Cartesian components. Values use scientific notation at every order.
    """
    order = force_constants.order
    if force_constants.n_supercell != len(supercell):
        raise ValueError("sparse force constants and supercell sizes differ")
    if order not in {3, 4}:
        raise ValueError("ShengBTE output supports only third- and fourth-order tensors")
    Path(target).write_text(_format_sparse_force_constants(force_constants, supercell))


def _format_sparse_force_constants(
    fc: SparseOrderForceConstants,
    supercell: Atoms,
) -> str:
    """Serialize sparse lattice-labelled clusters without atom-number arithmetic."""
    try:
        index = PeriodicIndex(
            np.asarray(supercell.arrays["primitive_index"]),
            np.asarray(supercell.arrays["cell_translation"]),
            np.asarray(supercell.info["mlfcs_supercell_matrix"]),
        )
    except KeyError as error:
        raise ValueError("supercell is missing MLFCS periodic mapping metadata") from error
    primitive_cell = np.linalg.inv(index.supercell_matrix) @ np.asarray(supercell.cell)
    if fc.is_lattice_labelled:
        sites = np.asarray(fc.sites)
        relative = np.asarray(fc.translation_representatives)
    else:
        sites = index.primitive[fc.clusters]
        translations = index.translations[fc.clusters]
        relative = translations[:, 1:] - translations[:, :1]
    try:
        primitive_scaled = np.asarray(supercell.arrays["primitive_scaled_position"])
    except KeyError as error:
        raise ValueError("supercell is missing primitive-site position metadata") from error
    basis_scaled = np.empty((fc.n_primitive, 3), dtype=float)
    for site in range(fc.n_primitive):
        atoms = np.flatnonzero(index.primitive == site)
        if not len(atoms):
            raise ValueError(f"supercell has no image of primitive site {site}")
        basis_scaled[site] = primitive_scaled[atoms[0]]
        if not np.allclose(primitive_scaled[atoms], basis_scaled[site], atol=1e-10, rtol=0.0):
            raise ValueError("primitive-site position metadata is inconsistent")

    geometry = PeriodicGeometry(supercell.cell, supercell.pbc)
    physical: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site_labels, translations, tensor in zip(sites, relative, fc.tensors, strict=True):
        translations = np.atleast_2d(translations)
        basis_vectors = (
            basis_scaled[site_labels[1:]] - basis_scaled[site_labels[0]]
        ) @ primitive_cell
        vectors = basis_vectors + translations @ primitive_cell
        for shifts in geometry.joint_closest_image_shifts(vectors):
            resolved = translations + shifts @ index.supercell_matrix
            key = tuple(int(value) for value in np.concatenate((site_labels, resolved.ravel())))
            previous = physical.get(key)
            if previous is not None:
                if not np.allclose(previous[2], tensor, atol=1e-8, rtol=1e-10):
                    raise ValueError("duplicate ShengBTE cluster has inconsistent force constants")
                continue
            physical[key] = (site_labels.copy(), resolved.copy(), tensor)

    # Writer order is physical and therefore independent of source reference
    # atom ordering. Block order is not part of ShengBTE's IFC semantics.
    blocks: list[str] = []
    for site_labels, translations, tensor in (physical[key] for key in sorted(physical)):
        lines = [
            "",
            f"{len(blocks) + 1:>5}",
            *[_vector_line(np.asarray(vector) * 0.1) for vector in translations @ primitive_cell],
            " ".join(f"{site + 1:>6d}" for site in site_labels),
        ]
        for directions in product(range(3), repeat=fc.order):
            direction_text = " ".join(f"{direction + 1:>2d}" for direction in directions)
            value = zero_small_scalar(tensor[directions], tolerance=_TEXT_ZERO_TOLERANCE)
            lines.append(f"{direction_text} {value:>20.10e}")
        blocks.append("\n".join(lines) + "\n")
    return f"{len(blocks):>5}\n" + "".join(blocks)


def _vector_line(vector: np.ndarray) -> str:
    vector_angstrom = 10.0 * vector
    return f"{vector_angstrom[0]:>15.10e} {vector_angstrom[1]:>15.10e} {vector_angstrom[2]:>15.10e}"
