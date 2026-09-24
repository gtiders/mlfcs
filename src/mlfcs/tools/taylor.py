"""Optional fixed-cell ASE calculator for primitive Taylor force constants."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial
from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from numba import njit

from mlfcs.core import LatticeSite
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.lattice import expand
from mlfcs.supercell import ClusterMap


@dataclass(frozen=True, slots=True)
class _OrderTerms:
    order: int
    atoms: np.ndarray
    tensors: np.ndarray
    components: np.ndarray


def _kernel(terms: _OrderTerms):
    """Compile one material's fixed atom map and IFC tensor for one selected order."""
    atoms = terms.atoms.copy()
    tensors = terms.tensors.copy()
    components = terms.components.copy()
    factor = 1.0 / factorial(terms.order)
    order = terms.order

    # These arrays are closure constants to Numba, not runtime arguments. A new
    # calculator therefore owns compiled kernels for its material and order mask.
    @njit(nogil=True)
    def forces(displacement):
        result = np.zeros_like(displacement)
        for image in range(atoms.shape[0]):
            for translation in range(atoms.shape[1]):
                for component in range(components.shape[0]):
                    tensor_value = tensors[image, component]
                    for axis in range(order):
                        product = tensor_value
                        for other in range(order):
                            if other != axis:
                                product *= displacement[
                                    atoms[image, translation, other], components[component, other]
                                ]
                        result[atoms[image, translation, axis], components[component, axis]] -= (
                            factor * product
                        )
        return result

    return forces


def _compile(
    model: ForceConstants,
    mapping: ClusterMap,
    order: int,
    translations: tuple[tuple[int, int, int], ...],
) -> _OrderTerms:
    supercell = mapping.supercell
    expanded = expand(model, order)
    atoms = np.empty((len(expanded.sites), len(translations), order), dtype=np.int32)
    tensors = np.empty((len(expanded.sites), 3**order), dtype=np.float64)
    for image, (sites, offsets, tensor) in enumerate(
        zip(expanded.sites, expanded.translations, expanded.tensors, strict=True)
    ):
        labels = ((0, 0, 0), *offsets)
        for cell_index, cell_translation in enumerate(translations):
            for axis, (site, offset) in enumerate(zip(sites, labels, strict=True)):
                shifted = tuple(
                    int(value + shift)
                    for value, shift in zip(offset, cell_translation, strict=True)
                )
                atoms[image, cell_index, axis] = supercell.atom(LatticeSite(site, shifted))
        tensors[image] = np.asarray(tensor, dtype=np.float64).reshape(-1)
    return _OrderTerms(
        order=order,
        atoms=np.ascontiguousarray(atoms),
        tensors=np.ascontiguousarray(tensors),
        components=np.ascontiguousarray(tuple(np.ndindex((3,) * order)), dtype=np.int32),
    )


class TaylorCalculator(Calculator):
    """ASE energy and forces relative to one fixed supercell.

    Only the chosen IFC orders contribute. The undisplaced supercell has zero energy and
    zero force: this model contains neither an absolute energy nor FC1.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        force_constants: ForceConstants,
        mapping: ClusterMap,
        *,
        orders: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(force_constants, ForceConstants):
            raise TypeError("force_constants must be a ForceConstants object")
        if not isinstance(mapping, ClusterMap):
            raise TypeError("mapping must be a ClusterMap")
        if force_constants.space.fingerprint != mapping.space.fingerprint:
            raise ValueError("force constants and cluster map use different cluster spaces")
        selected = force_constants.orders if orders is None else tuple(orders)
        if not selected or any(not isinstance(order, (int, np.integer)) for order in selected):
            raise ValueError("orders must be a nonempty sequence of IFC orders")
        selected = tuple(int(order) for order in selected)
        if tuple(sorted(set(selected))) != selected:
            raise ValueError("orders must be unique and ascending")
        missing = tuple(order for order in selected if order not in force_constants.coefficients)
        if missing:
            raise ValueError(f"force constants do not contain orders {missing}")

        self.force_constants = force_constants
        self.mapping = mapping
        self._orders = selected
        supercell = mapping.supercell
        self._cell = np.array(supercell.cell, dtype=np.float64, copy=True)
        self._numbers = np.array(supercell.numbers, dtype=np.int32, copy=True)
        self._supercell_positions = np.asarray(supercell.scaled_positions) @ self._cell
        self._inverse_cell = np.linalg.inv(self._cell)
        translations = supercell.cell_translations
        self._terms = tuple(
            _compile(force_constants, mapping, order, translations) for order in selected
        )
        self._kernels = tuple(_kernel(terms) for terms in self._terms)

    @property
    def orders(self) -> tuple[int, ...]:
        """The immutable set of force-constant orders used by this calculator."""
        return self._orders

    @property
    def supercell(self) -> Atoms:
        """Return a fresh ASE copy of the fixed supercell."""
        return Atoms(
            numbers=self._numbers, positions=self._supercell_positions, cell=self._cell, pbc=True
        )

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        """Evaluate relative energy and forces; cell strain is unsupported."""
        if atoms is None or not isinstance(atoms, Atoms):
            raise TypeError("TaylorCalculator requires an ASE Atoms object")
        if not np.array_equal(atoms.numbers, self._numbers):
            raise ValueError("atoms have a different supercell atom sequence")
        if not bool(np.all(atoms.pbc)):
            raise ValueError("atoms must retain periodic boundary conditions")
        if not np.array_equal(np.asarray(atoms.cell), self._cell):
            raise ValueError("TaylorCalculator requires the fixed supercell; stress is unsupported")
        super().calculate(atoms, properties, system_changes)
        scaled = (atoms.positions - self._supercell_positions) @ self._inverse_cell
        scaled -= np.rint(scaled)
        displacement = np.ascontiguousarray(scaled @ self._cell)
        if not np.all(np.isfinite(displacement)):
            raise ValueError("atomic positions contain NaN or infinite values")

        forces = np.zeros_like(displacement)
        energy = 0.0
        for terms, kernel in zip(self._terms, self._kernels, strict=True):
            contribution = kernel(displacement)
            forces += contribution
            energy -= float(np.sum(displacement * contribution)) / terms.order
        self.results = {"energy": energy, "forces": forces}


__all__ = ["TaylorCalculator"]
