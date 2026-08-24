from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs.force_constants.representation import SparseOrderForceConstants
from mlfcs.io.numeric_text import zero_small_scalar
from mlfcs.structure.supercell_mapping import PeriodicIndex

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
    sites = np.asarray(fc.sites)
    relative = np.asarray(fc.translations)
    physical: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for site_labels, translations, tensor in zip(sites, relative, fc.tensors, strict=True):
        translations = np.atleast_2d(translations)
        key = tuple(int(value) for value in np.concatenate((site_labels, translations.ravel())))
        previous = physical.get(key)
        if previous is not None:
            if not np.allclose(previous[2], tensor, atol=1e-8, rtol=1e-10):
                raise ValueError("duplicate ShengBTE cluster has inconsistent force constants")
            continue
        physical[key] = (site_labels.copy(), translations.copy(), tensor)

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
