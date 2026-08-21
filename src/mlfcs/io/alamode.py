"""ALAMODE FCSXML output with MLFCS-controlled atom mappings.

Third-party provenance
----------------------
The XML layout, unit conversion, and 27-image closest-mirror convention are
adapted from ``alm.fcsxml.Fcsxml`` in ttadano/ALM revision
``f1d668f210d3e95355643132144f3fd1ec10d4d7``. MLFCS replaces ALM's primitive
discovery with the deterministic ``primitive_index`` and ``cell_translation``
arrays attached to its reference supercell.

The MIT License (MIT)

Copyright (c) 2014,2015,2016 Terumasa Tadano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from itertools import pairwise, product
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import numpy as np
from ase import Atoms
from ase.units import Bohr, Rydberg

from mlfcs.core.geometry import PeriodicGeometry, PeriodicIndex
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.io._text import zero_small_scalar

_MIRROR_SHIFTS = np.asarray(
    [(0, 0, 0), *(shift for shift in product((-1, 0, 1), repeat=3) if shift != (0, 0, 0))],
    dtype=np.int32,
)
_MIRROR_TOLERANCE_BOHR = 1.0e-3
_TEXT_ZERO_TOLERANCE = 1.0e-8


class AlamodeMirrorImageError(ValueError):
    """The physical MIC cannot be represented by ALAMODE's 27 images."""


def write_alamode(
    target: str | Path,
    force_constants: ForceConstants,
    *,
    orders: tuple[int, ...],
) -> None:
    """Write FC2--FC4 in ALAMODE's XML force-constant format."""
    if not orders:
        raise ValueError("at least one force-constant order is required")
    unsupported = tuple(order for order in orders if order not in {2, 3, 4})
    if unsupported:
        raise ValueError(
            f"ALAMODE XML output supports only orders 2, 3, and 4; requested {unsupported}"
        )

    geometry = _AlamodeGeometry.from_supercell(force_constants.supercell)
    root = Element("Data")
    # This is an ALAMODE schema/version field, not a producer string. The
    # standalone ALM Python writer uses ``None`` for externally assembled FCs.
    SubElement(root, "ALM_version").text = "None"
    _write_structure(root, geometry)
    _write_translations(root, geometry)
    force_constants_node = SubElement(root, "ForceConstants")
    for order in orders:
        entries = _force_constant_entries(force_constants, order)
        _write_order(force_constants_node, geometry, order, entries)

    indent(root, space="  ")
    ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)


class _AlamodeGeometry:
    def __init__(
        self,
        supercell: Atoms,
        primitive_index: np.ndarray,
        cell_translation: np.ndarray,
        translations: np.ndarray,
        atom_by_key: dict[tuple[int, int, int, int], int],
    ) -> None:
        self.supercell = supercell
        self.primitive_index = primitive_index
        self.cell_translation = cell_translation
        self.translations = translations
        self.atom_by_key = atom_by_key
        self.n_primitive = int(primitive_index.max()) + 1
        self._mirror_cache: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_supercell(cls, supercell: Atoms) -> _AlamodeGeometry:
        try:
            primitive_index = np.asarray(supercell.arrays["primitive_index"], dtype=np.int32)
            cell_translation = np.asarray(supercell.arrays["cell_translation"], dtype=np.int32)
        except KeyError as error:
            raise ValueError("supercell is missing MLFCS atom-order metadata") from error
        if primitive_index.shape != (len(supercell),):
            raise ValueError("primitive_index has an invalid shape")
        if cell_translation.shape != (len(supercell), 3):
            raise ValueError("cell_translation has an invalid shape")

        translations = np.unique(cell_translation, axis=0)
        zero = np.flatnonzero(np.all(translations == 0, axis=1))
        if len(zero) != 1:
            raise ValueError("MLFCS supercell must contain the zero cell translation")
        remaining = np.delete(translations, zero[0], axis=0)
        if len(remaining):
            remaining = remaining[np.lexsort(remaining.T[::-1])]
        # ALAMODE reserves translation number 1 for the identity operation;
        # the labels themselves may otherwise be negative or noncanonical.
        translations = np.vstack((np.zeros((1, 3), dtype=np.int32), remaining))
        n_primitive = int(primitive_index.max()) + 1
        if len(translations) * n_primitive != len(supercell):
            raise ValueError("primitive and translation metadata do not cover the supercell")
        atom_by_key = {
            (int(primitive), *(int(value) for value in translation)): atom
            for atom, (primitive, translation) in enumerate(
                zip(primitive_index, cell_translation, strict=True)
            )
        }
        if len(atom_by_key) != len(supercell):
            raise ValueError("MLFCS supercell atom mapping is not one-to-one")
        expected = {
            (primitive, *(int(value) for value in translation))
            for translation in translations
            for primitive in range(n_primitive)
        }
        if set(atom_by_key) != expected:
            raise ValueError("MLFCS supercell mapping is incomplete")
        return cls(
            supercell,
            primitive_index,
            cell_translation,
            translations,
            atom_by_key,
        )

    def central_atom(self, primitive: int) -> int:
        return self.atom_by_key[(primitive, 0, 0, 0)]

    def closest_mirror_images(self, first: int, second: int) -> np.ndarray:
        key = (first, second)
        if key in self._mirror_cache:
            return self._mirror_cache[key]
        cell = np.asarray(self.supercell.cell)
        delta = self.supercell.positions[second] - self.supercell.positions[first]
        candidates = delta[None, :] + _MIRROR_SHIFTS @ cell
        distances_bohr = np.linalg.norm(candidates, axis=1) / Bohr
        minimum = float(distances_bohr.min())
        images = np.flatnonzero(np.abs(distances_bohr - minimum) < _MIRROR_TOLERANCE_BOHR).astype(
            np.int32
        )

        _, general_distance = PeriodicGeometry(cell, self.supercell.pbc).mic(delta)
        if not np.isclose(minimum * Bohr, general_distance, atol=1.0e-8, rtol=1.0e-10):
            raise AlamodeMirrorImageError(
                "ALAMODE XML's 27-image convention cannot represent the minimum image "
                f"between supercell atoms {first} and {second}"
            )
        self._mirror_cache[key] = images
        return images


def _write_structure(root: Element, geometry: _AlamodeGeometry) -> None:
    supercell = geometry.supercell
    structure = SubElement(root, "Structure")
    SubElement(structure, "NumberOfAtoms").text = str(len(supercell))
    symbols = list(dict.fromkeys(supercell.get_chemical_symbols()))
    SubElement(structure, "NumberOfElements").text = str(len(symbols))
    elements = SubElement(structure, "AtomicElements")
    for number, symbol in enumerate(symbols, start=1):
        SubElement(elements, "element", number=str(number)).text = symbol

    lattice = SubElement(structure, "LatticeVector")
    for axis, vector in enumerate(np.asarray(supercell.cell) / Bohr, start=1):
        SubElement(lattice, f"a{axis}").text = _vector_text(vector)
    SubElement(structure, "Periodicity").text = " ".join(
        "1" if periodic else "0" for periodic in supercell.pbc
    )
    positions = SubElement(structure, "Position")
    for atom, (symbol, scaled) in enumerate(
        zip(supercell.get_chemical_symbols(), supercell.get_scaled_positions(), strict=True),
        start=1,
    ):
        SubElement(positions, "pos", index=str(atom), element=symbol).text = _vector_text(scaled)


def _write_translations(root: Element, geometry: _AlamodeGeometry) -> None:
    symmetry = SubElement(root, "Symmetry")
    SubElement(symmetry, "NumberOfTranslations").text = str(len(geometry.translations))
    mappings = SubElement(symmetry, "Translations")
    for translation_number, translation in enumerate(geometry.translations, start=1):
        for primitive in range(geometry.n_primitive):
            atom = geometry.atom_by_key[(primitive, *(int(value) for value in translation))]
            SubElement(
                mappings,
                "map",
                tran=str(translation_number),
                atom=str(primitive + 1),
            ).text = str(atom + 1)


def _force_constant_entries(
    force_constants: ForceConstants,
    order: int,
) -> dict[tuple[int, ...], float]:
    sparse = force_constants.sparse.get(order)
    if sparse is not None:
        return _sparse_entries(sparse, force_constants.supercell)
    values = np.asarray(force_constants.arrays[order])
    expected_atoms = (len(force_constants.supercell.arrays["primitive_index"]),) * (order - 1)
    n_primitive = int(force_constants.supercell.arrays["primitive_index"].max()) + 1
    expected = (n_primitive, *expected_atoms, *((3,) * order))
    if values.shape != expected:
        raise ValueError(f"order-{order} force constants must have shape {expected}")
    entries: dict[tuple[int, ...], float] = {}
    for location in zip(*np.nonzero(values), strict=True):
        atoms = location[:order]
        directions = location[order:]
        flat = tuple(3 * int(atom) + int(direction) for atom, direction in zip(atoms, directions))
        if _ascending_tail(flat):
            entries[flat] = float(values[location])
    return entries


def _sparse_entries(
    sparse: SparseOrderForceConstants, supercell: Atoms
) -> dict[tuple[int, ...], float]:
    """Convert reference-order clusters to ALAMODE's zero-cell atom view."""
    try:
        index = PeriodicIndex(
            np.asarray(supercell.arrays["primitive_index"]),
            np.asarray(supercell.arrays["cell_translation"]),
            np.asarray(supercell.info["mlfcs_supercell_matrix"]),
        )
    except KeyError as error:
        raise ValueError("supercell is missing MLFCS periodic mapping metadata") from error
    totals: dict[tuple[int, ...], float] = {}
    for cluster, tensor in zip(sparse.clusters, sparse.tensors, strict=True):
        sites = index.primitive[cluster]
        translations = index.translations[cluster]
        anchored = [index.representative(int(sites[0]))]
        anchored.extend(
            index.atom(int(site), translation - translations[0])
            for site, translation in zip(sites[1:], translations[1:], strict=True)
        )
        for directions in product(range(3), repeat=sparse.order):
            flat = tuple(
                3 * int(atom) + direction
                for atom, direction in zip(anchored, directions, strict=True)
            )
            if not _ascending_tail(flat):
                continue
            value = float(tensor[directions])
            totals[flat] = totals.get(flat, 0.0) + value
    return {key: value for key, value in totals.items() if value != 0.0}


def _write_order(
    parent: Element,
    geometry: _AlamodeGeometry,
    order: int,
    entries: dict[tuple[int, ...], float],
) -> None:
    container = SubElement(parent, "HARMONIC" if order == 2 else f"ANHARM{order}")
    conversion = Bohr**order / Rydberg
    for flat, value in sorted(entries.items()):
        value = zero_small_scalar(value, tolerance=_TEXT_ZERO_TOLERANCE)
        if value == 0.0:
            continue
        atoms = tuple(index // 3 for index in flat)
        directions = tuple(index % 3 + 1 for index in flat)
        first = atoms[0]
        first_primitive = int(geometry.primitive_index[first])
        if first != geometry.central_atom(first_primitive):
            raise ValueError(
                "the first sparse force-constant atom must be anchored in the "
                "MLFCS zero-translation primitive cell"
            )
        first_supercell = first
        unique_atoms = tuple(dict.fromkeys(atoms[1:]))
        images_by_atom = {
            atom: geometry.closest_mirror_images(first_supercell, atom) for atom in unique_atoms
        }
        combinations = tuple(product(*(images_by_atom[atom] for atom in unique_atoms)))
        if not combinations:
            raise ValueError("ALAMODE mirror-image expansion produced no entries")
        image_position = {atom: position for position, atom in enumerate(unique_atoms)}
        scaled = value * conversion / len(combinations)
        for combination in combinations:
            attributes = {"pair1": f"{first_primitive + 1} {directions[0]}"}
            for axis, (atom, direction) in enumerate(
                zip(atoms[1:], directions[1:], strict=True), start=2
            ):
                image = int(combination[image_position[atom]])
                attributes[f"pair{axis}"] = f"{atom + 1} {direction} {image + 1}"
            SubElement(container, f"FC{order}", attributes).text = f"{scaled:.15e}"


def _ascending_tail(flat: tuple[int, ...]) -> bool:
    return all(left <= right for left, right in pairwise(flat[1:]))


def _vector_text(vector: np.ndarray) -> str:
    return " " + " ".join(f"{float(value):.15e}" for value in vector)


__all__ = ["AlamodeMirrorImageError", "write_alamode"]
