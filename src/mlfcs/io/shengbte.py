from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist


def write_shengbte(
    target: str | Path,
    force_constants: np.ndarray,
    supercell: Atoms,
    *,
    cutoff: float,
) -> None:
    """Write an order-parameterized ShengBTE-style force-constant file.

    Atomic axes precede Cartesian axes. A block contains ``order - 1``
    lattice translations, ``order`` primitive atom indices, and ``3**order``
    Cartesian components. Values use scientific notation at every order.
    """
    order = force_constants.ndim // 2
    if order not in {3, 4} or force_constants.ndim != 2 * order:
        raise ValueError("ShengBTE output supports only third- and fourth-order tensors")
    n_primitive = force_constants.shape[0]
    n_supercell = len(supercell)
    expected = (n_primitive,) + (n_supercell,) * (order - 1) + (3,) * order
    if force_constants.shape != expected:
        raise ValueError(f"force constants must have shape {expected}")

    geometry = _nanometre_geometry(supercell)
    cutoff_nm = cutoff * 0.1
    distances, counts, shifts = _periodic_distances(geometry)
    text = _format_force_constants(
        force_constants,
        geometry,
        n_primitive,
        distances,
        counts,
        shifts,
        cutoff_nm,
        order,
    )
    Path(target).write_text(text)


def _nanometre_geometry(supercell: Atoms) -> Atoms:
    geometry = supercell.copy()
    geometry.set_cell(np.asarray(geometry.cell) * 0.1, scale_atoms=False)
    required = {"primitive_scaled_position", "cell_translation"}
    if required <= geometry.arrays.keys():
        translations = geometry.arrays["cell_translation"]
        repeats = translations.max(axis=0) + 1
        fractional = (geometry.arrays["primitive_scaled_position"] + translations) / repeats
        geometry.positions = fractional @ geometry.cell
    else:
        geometry.positions *= 0.1
    return geometry


def _periodic_distances(
    supercell: Atoms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shift_vectors = np.asarray(list(product(range(-1, 2), repeat=3)), dtype=np.int32)
    cartesian_shifts = shift_vectors @ supercell.cell
    positions = supercell.positions
    squared = np.asarray(
        [cdist(positions, positions + shift, "sqeuclidean") for shift in cartesian_shifts]
    )
    minimum = squared.min(axis=0)
    degenerate = np.abs(squared - minimum) < 1e-4
    counts = degenerate.sum(axis=0, dtype=np.int32)
    maximum = int(counts.max())
    sorting = np.argsort(~degenerate, axis=0)
    shift_indices = np.transpose(sorting[:maximum], (1, 2, 0)).astype(np.int32)
    return np.sqrt(minimum), counts, shift_indices


def _format_force_constants(
    fc: np.ndarray,
    supercell: Atoms,
    n_primitive: int,
    distances: np.ndarray,
    counts: np.ndarray,
    shifts: np.ndarray,
    cutoff: float,
    order: int,
) -> str:
    shift_vectors = np.asarray(list(product(range(-1, 2), repeat=3)))
    blocks: list[str] = []
    block_number = 0
    for first in range(n_primitive):
        for remaining in product(range(len(supercell)), repeat=order - 1):
            atom_indices = (first,) + remaining
            tensor = fc[atom_indices]
            has_force_constants = bool(np.any(tensor != 0.0))
            if not has_force_constants and any(
                distances[first, atom] >= cutoff for atom in remaining
            ):
                continue
            possible_shifts = [
                shift_vectors[shifts[first, atom, : counts[first, atom]]] for atom in remaining
            ]
            best_distance, best_shifts = _best_joint_images(supercell, remaining, possible_shifts)
            if not has_force_constants and best_distance >= cutoff * cutoff:
                continue
            primitive_atoms = (first,) + tuple(atom % n_primitive for atom in remaining)
            translations = tuple(
                _translation(supercell, atom, primitive_atom, shift)
                for atom, primitive_atom, shift in zip(
                    remaining, primitive_atoms[1:], best_shifts, strict=True
                )
            )
            block_number += 1
            lines = [
                "",
                f"{block_number:>5}",
                *[_vector_line(vector) for vector in translations],
                " ".join(f"{atom + 1:>6d}" for atom in primitive_atoms),
            ]
            for directions in product(range(3), repeat=order):
                direction_text = " ".join(f"{direction + 1:>2d}" for direction in directions)
                value = fc[atom_indices + directions]
                lines.append(f"{direction_text} {value:>20.10e}")
            blocks.append("\n".join(lines) + "\n")
    return f"{block_number:>5}\n" + "".join(blocks)


def _best_joint_images(
    supercell: Atoms,
    atoms: tuple[int, ...],
    possible_shifts: list[np.ndarray],
) -> tuple[float, tuple[np.ndarray, ...]]:
    best_distance = np.inf
    best_shifts = tuple(shifts[0] for shifts in possible_shifts)
    pairs = tuple(combinations(range(len(atoms)), 2))
    for selected_shifts in product(*possible_shifts):
        positions = tuple(
            supercell.positions[atom] + shift @ supercell.cell
            for atom, shift in zip(atoms, selected_shifts, strict=True)
        )
        distance = max(
            float(np.sum((positions[left] - positions[right]) ** 2)) for left, right in pairs
        )
        if distance < best_distance:
            best_distance = distance
            best_shifts = selected_shifts
    return best_distance, best_shifts


def _translation(
    supercell: Atoms,
    atom: int,
    primitive_atom: int,
    shift: np.ndarray,
) -> np.ndarray:
    return supercell.positions[atom] + shift @ supercell.cell - supercell.positions[primitive_atom]


def _vector_line(vector: np.ndarray) -> str:
    vector_angstrom = 10.0 * vector
    return f"{vector_angstrom[0]:>15.10e} {vector_angstrom[1]:>15.10e} {vector_angstrom[2]:>15.10e}"
