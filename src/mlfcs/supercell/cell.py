"""An immutable supercell that provides atoms and an exact periodic quotient."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.core import LatticeSite, PrimitiveCell
from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log_error import get_logger

logger = get_logger(__name__)


def _determinant(rows: list[list[int]]) -> int:
    a, b, c = rows[0]
    d, e, f = rows[1]
    g, h, i = rows[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _adjugate(rows: list[list[int]]) -> list[list[int]]:
    a, b, c = rows[0]
    d, e, f = rows[1]
    g, h, i = rows[2]
    return [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Supercell:
    """An explicit supercell indexed by primitive sites and quotient labels."""

    primitive: PrimitiveCell
    matrix: np.ndarray
    cell: np.ndarray
    scaled_positions: np.ndarray
    numbers: np.ndarray
    sites: np.ndarray
    translations: np.ndarray
    quotients: np.ndarray
    determinant: int

    @classmethod
    def from_atoms(
        cls,
        primitive: PrimitiveCell,
        atoms: Atoms,
    ) -> Supercell:
        """Infer the integer basis change and map atoms at the primitive ``symprec``.

        Spglib determines the primitive's symmetry; its standardized-cell
        transformation is not the relation between these two supplied bases.
        Each supercell lattice vector is matched to one primitive lattice
        vector within ``symprec`` angstrom, in ASE's row convention.
        """
        if not isinstance(primitive, PrimitiveCell) or not isinstance(atoms, Atoms):
            raise TypeError("from_atoms requires a PrimitiveCell and ASE Atoms")
        actual_cell = np.asarray(atoms.cell, dtype=np.float64)
        if not np.all(np.isfinite(actual_cell)):
            raise ValueError("supercell lattice must be finite")
        geometry = PeriodicGeometry(primitive.cell)
        rows = []
        for axis, vector in enumerate(actual_cell):
            images, shifts = geometry.closest_images(vector, symprec=primitive.symprec)
            distances = np.linalg.norm(images, axis=1)
            matches = np.flatnonzero(distances < primitive.symprec)
            if len(matches) != 1:
                nearest = float(np.min(distances)) if distances.size else float("inf")
                raise ValueError(
                    f"supercell lattice residual {nearest:.10g} angstrom for vector {axis}: "
                    f"{len(matches)} integer matches below symprec "
                    f"{primitive.symprec:.10g} angstrom"
                )
            rows.append([-int(value) for value in shifts[int(matches[0])]])
        determinant = _determinant(rows)
        if determinant == 0:
            raise ValueError("inferred supercell matrix is singular")
        copies = abs(determinant)
        if len(atoms) != primitive.size * copies:
            raise ValueError(
                f"supercell has {len(atoms)} atoms, expected {primitive.size * copies} "
                f"from determinant {determinant}"
            )
        integer_matrix = np.asarray(rows, dtype=object)
        expected_cell = np.asarray(integer_matrix, dtype=np.float64) @ primitive.cell
        residual = float(np.max(np.linalg.norm(expected_cell - actual_cell, axis=1)))
        if residual >= primitive.symprec:
            raise ValueError(
                f"inferred supercell matrix {rows} leaves lattice residual "
                f"{residual:.10g} angstrom, not below symprec "
                f"{primitive.symprec:.10g} angstrom"
            )

        cartesian = atoms.get_positions()
        primitive_scaled = cartesian @ np.linalg.inv(primitive.cell)
        sites = np.empty(len(atoms), dtype=np.int32)
        translations = np.empty((len(atoms), 3), dtype=np.int64)
        numbers = np.asarray(atoms.numbers, dtype=np.int32)
        maximum_mapping_residual = 0.0
        for atom, (number, position) in enumerate(zip(numbers, primitive_scaled, strict=True)):
            candidates = np.flatnonzero(primitive.numbers == number)
            differences = position - primitive.scaled_positions[candidates]
            _, distances = geometry.minimum_image(differences @ primitive.cell)
            matches = np.flatnonzero(distances < primitive.symprec)
            if len(matches) != 1:
                nearest = float(np.min(distances)) if distances.size else float("inf")
                raise ValueError(
                    f"supercell atom {atom} has {len(matches)} primitive matches below "
                    f"symprec; nearest residual is {nearest:.10g} angstrom"
                )
            match = int(matches[0])
            maximum_mapping_residual = max(maximum_mapping_residual, float(distances[match]))
            _, image_shifts = geometry.closest_images(
                differences[match] @ primitive.cell, symprec=primitive.symprec
            )
            if len(image_shifts) != 1:
                raise ValueError(
                    f"supercell atom {atom} has {len(image_shifts)} primitive images below symprec"
                )
            sites[atom] = int(candidates[match])
            translations[atom] = -image_shifts[0]

        adjugate = _adjugate(rows)
        modulus = abs(determinant)
        quotients = np.empty((len(atoms), 3), dtype=np.int64)
        for atom, translation in enumerate(translations):
            quotients[atom] = [
                sum(int(translation[axis]) * adjugate[axis][column] for axis in range(3)) % modulus
                for column in range(3)
            ]
        keys = [
            (int(site), *(int(value) for value in quotient))
            for site, quotient in zip(sites, quotients, strict=True)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("supercell does not contain each primitive quotient site exactly once")

        logger.info(
            "Supercell mapped: %d atoms, matrix %s, determinant %d, cell residual %.6g Å, "
            "maximum site residual %.6g Å",
            len(atoms),
            rows,
            determinant,
            residual,
            maximum_mapping_residual,
        )

        return cls(
            primitive=primitive,
            matrix=_readonly(integer_matrix, object),
            cell=_readonly(actual_cell, np.float64),
            scaled_positions=_readonly(atoms.get_scaled_positions(wrap=True), np.float64),
            numbers=_readonly(numbers, np.int32),
            sites=_readonly(sites, np.int32),
            translations=_readonly(translations, np.int64),
            quotients=_readonly(quotients, np.int64),
            determinant=determinant,
        )

    def quotient(self, translation: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the exact quotient label of a primitive translation."""
        rows = [[int(value) for value in row] for row in self.matrix]
        adjugate = _adjugate(rows)
        modulus = abs(self.determinant)
        return tuple(
            sum(int(translation[axis]) * adjugate[axis][column] for axis in range(3)) % modulus
            for column in range(3)
        )

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        """Inherit periodicity from the certified primitive cell."""
        return self.primitive.pbc

    @property
    def cell_translations(self) -> tuple[tuple[int, int, int], ...]:
        """One primitive translation per quotient, in explicit atom order for site zero."""
        seen: set[tuple[int, int, int]] = set()
        result = []
        for site, translation, quotient in zip(
            self.sites, self.translations, self.quotients, strict=True
        ):
            if int(site) != 0:
                continue
            key = tuple(int(value) for value in quotient)
            if key not in seen:
                seen.add(key)
                result.append(tuple(int(value) for value in translation))
        if len(result) != abs(self.determinant):
            raise RuntimeError("supercell lacks one translation per periodic quotient")
        return tuple(result)

    def atom(self, site: LatticeSite) -> int:
        """Return the supercell atom addressed by one primitive lattice site."""
        quotient = self.quotient(site.translation)
        matches = np.flatnonzero(
            (self.sites == site.site) & np.all(self.quotients == quotient, axis=1)
        )
        if len(matches) != 1:
            raise RuntimeError(f"primitive site {site} has {len(matches)} supercell atoms")
        return int(matches[0])

    @property
    def fingerprint(self) -> str:
        payload = {
            "primitive": self.primitive.fingerprint,
            "matrix": self.matrix.tolist(),
            "cell": [[float(value).hex() for value in row] for row in self.cell],
            "scaled_positions": [
                [float(value).hex() for value in row] for row in self.scaled_positions
            ],
            "numbers": self.numbers.tolist(),
            "sites": self.sites.tolist(),
            "quotients": self.quotients.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["Supercell"]
