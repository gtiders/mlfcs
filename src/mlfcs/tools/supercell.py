"""Standalone ASE supercell construction helpers.

This module deliberately has no dependency on MLFCS core or calculation
modules.  It prepares a reference ASE structure; the calculation layer later
validates that structure against the user-supplied primitive.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
from ase import Atoms

SupercellOrdering = Literal["phonopy", "phonopy_snf", "thirdorder"]


def _normalize_matrix(matrix: object) -> np.ndarray:
    values = np.asarray(matrix)
    if values.shape == (3,):
        values = np.diag(values)
    if values.shape != (3, 3):
        raise ValueError("supercell_matrix must be three repeats or an integer 3x3 matrix")
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, atol=1e-10, rtol=0.0):
        raise ValueError("supercell_matrix must contain integers")
    if round(float(np.linalg.det(rounded))) == 0:
        raise ValueError("supercell_matrix must be nonsingular")
    return rounded.astype(np.int32)


def _phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    kwargs = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.asarray(atoms.cell),
        "scaled_positions": atoms.get_scaled_positions(wrap=True),
    }
    masses = atoms.get_masses()
    if masses is not None:
        kwargs["masses"] = masses
    return PhonopyAtoms(**kwargs)


def _from_phonopy(atoms: Atoms, matrix: np.ndarray, *, ordering: str, symprec: float) -> Atoms:
    from phonopy.structure.cells import get_supercell

    is_old_style = ordering != "phonopy_snf"
    phonopy_matrix = matrix if ordering == "phonopy_snf" else matrix.T
    result = get_supercell(
        _phonopy_atoms(atoms), phonopy_matrix, is_old_style=is_old_style, symprec=symprec
    )
    return Atoms(
        symbols=result.symbols,
        cell=np.asarray(result.cell),
        scaled_positions=np.asarray(result.scaled_positions),
        pbc=True,
    )


def _translation_label(translation: np.ndarray, matrix: np.ndarray) -> tuple[int, int, int]:
    determinant = abs(round(float(np.linalg.det(matrix))))
    adjugate = np.rint(np.linalg.det(matrix) * np.linalg.inv(matrix)).astype(np.int64)
    residue = np.mod(np.asarray(translation, dtype=np.int64) @ adjugate, determinant)
    return tuple(int(value) for value in residue)


def _coset_translations(matrix: np.ndarray) -> np.ndarray:
    count = abs(round(float(np.linalg.det(matrix))))
    zero = np.zeros(3, dtype=np.int32)
    found = {_translation_label(zero, matrix): zero}
    pending = deque([zero])
    generators = np.eye(3, dtype=np.int32)
    while pending and len(found) < count:
        current = pending.popleft()
        for generator in generators:
            candidate = current + generator
            residue = _translation_label(candidate, matrix)
            if residue not in found:
                found[residue] = candidate
                pending.append(candidate)
    if len(found) != count:
        raise RuntimeError("could not enumerate supercell translation cosets")
    return np.asarray(
        sorted(found.values(), key=lambda value: (value[2], value[1], value[0])), dtype=np.int32
    )


def _thirdorder(atoms: Atoms, matrix: np.ndarray) -> Atoms:
    translations = _coset_translations(matrix)
    positions = np.concatenate(
        [atoms.positions + shift @ atoms.cell for shift in translations]
    )
    return Atoms(
        numbers=np.tile(atoms.numbers, len(translations)),
        positions=positions,
        cell=matrix @ np.asarray(atoms.cell),
        pbc=True,
    )


def _fallback_phonopy_old_style(atoms: Atoms, matrix: np.ndarray, *, symprec: float) -> Atoms:
    determinant = round(float(np.linalg.det(matrix)))
    if determinant <= 0:
        raise ValueError("phonopy ordering requires a positive determinant")
    phonopy_matrix = matrix.T
    corners = np.asarray(
        (
            (0, 0, 0),
            phonopy_matrix[:, 0],
            phonopy_matrix[:, 1],
            phonopy_matrix[:, 2],
            phonopy_matrix[:, 1] + phonopy_matrix[:, 2],
            phonopy_matrix[:, 2] + phonopy_matrix[:, 0],
            phonopy_matrix[:, 0] + phonopy_matrix[:, 1],
            phonopy_matrix[:, 0] + phonopy_matrix[:, 1] + phonopy_matrix[:, 2],
        ),
        dtype=np.int64,
    )
    multiplicities = np.max(corners, axis=0) - np.min(corners, axis=0)
    if np.any(multiplicities <= 0):
        raise ValueError("phonopy surrounding frame has a zero multiplicity")
    simple_matrix = np.diag(multiplicities)
    simple_cell = simple_matrix @ np.asarray(atoms.cell)
    trim_frame = phonopy_matrix / multiplicities[:, None]
    target_cell = trim_frame.T @ simple_cell
    b, c, a = np.meshgrid(
        range(int(multiplicities[1])),
        range(int(multiplicities[2])),
        range(int(multiplicities[0])),
    )
    lattice_points = np.c_[a.ravel(), b.ravel(), c.ravel()]
    images = len(lattice_points)
    scaled = atoms.get_scaled_positions(wrap=True)
    positions = (
        np.tile(lattice_points, (len(atoms), 1))
        + np.repeat(scaled, images, axis=0)
    ) @ np.linalg.inv(simple_matrix).T
    positions = positions @ np.linalg.inv(trim_frame).T
    positions -= np.floor(positions)
    numbers = np.repeat(atoms.numbers, images)
    selected: list[int] = []
    for atom, position in enumerate(positions):
        if selected:
            delta = positions[np.asarray(selected)] - position
            delta -= np.rint(delta)
            distance = np.linalg.norm(delta @ target_cell, axis=1)
            if np.any((distance < symprec) & (numbers[np.asarray(selected)] == numbers[atom])):
                continue
        selected.append(atom)
    return Atoms(
        numbers=numbers[np.asarray(selected)],
        scaled_positions=positions[np.asarray(selected)],
        cell=target_cell,
        pbc=True,
    )


def build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    ordering: SupercellOrdering = "phonopy",
    symprec: float = 1e-5,
) -> Atoms:
    """Build an ASE reference supercell without invoking MLFCS core logic."""
    if not isinstance(primitive, Atoms):
        raise TypeError("primitive must be an ASE Atoms object")
    if not np.all(primitive.pbc):
        raise ValueError("primitive must be periodic")
    if ordering not in {"phonopy", "phonopy_snf", "thirdorder"}:
        raise ValueError("invalid supercell ordering")
    matrix = _normalize_matrix(supercell_matrix)
    if ordering == "phonopy_snf":
        raise NotImplementedError("phonopy_snf is reserved and not implemented")
    if ordering == "thirdorder":
        return _thirdorder(primitive, matrix)
    try:
        return _from_phonopy(primitive, matrix, ordering=ordering, symprec=symprec)
    except ImportError:
        return _fallback_phonopy_old_style(primitive, matrix, symprec=symprec)


__all__ = ["SupercellOrdering", "build_supercell"]
