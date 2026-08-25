"""Reference-relative Taylor polynomial evaluation from sparse exact-R IFCs."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from ase import Atoms

from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.structure.relation import StructureRelation


@dataclass(frozen=True, slots=True)
class _TaylorTerms:
    order: int
    atoms: np.ndarray
    tensors: np.ndarray


class TaylorPotential:
    """A fixed-cell, reference-relative Taylor polynomial.

    This internal evaluator interprets canonical force constants as an energy
    expansion with zero constant and linear terms.  It deliberately has no
    dependency on fitting backends or JAX.
    """

    def __init__(
        self,
        force_constants: ForceConstants,
        *,
        reference: Atoms | None = None,
        maximum_displacement: float | None = None,
    ) -> None:
        if not isinstance(force_constants, ForceConstants):
            raise TypeError("force_constants must be a ForceConstants object")
        basis = force_constants.metadata.get("force_constants_basis")
        if basis is not None and str(basis).lower() != "taylor":
            raise ValueError("MLFCSCalculator requires Taylor force constants")
        if maximum_displacement is not None and maximum_displacement <= 0:
            raise ValueError("maximum_displacement must be positive")
        if not isinstance(force_constants.relation, StructureRelation):
            raise TypeError("force constants require an explicit primitive structure relation")

        target = force_constants.relation.reference if reference is None else reference
        realized = realize_force_constants(force_constants, target)
        relation = realized.relation
        if not isinstance(relation, StructureRelation):
            raise TypeError("target realization did not produce a structure relation")
        if any(order < 2 for order in realized.orders):
            raise ValueError("MLFCSCalculator supports Taylor force constants of order 2 or higher")

        self.force_constants = realized
        self.relation = relation
        self.maximum_displacement = maximum_displacement
        self._terms = self._prepare_terms()
        self._periodic_fc2 = (
            realized.periodic_fc2_completion.full_hessian(relation.reference)
            if realized.periodic_fc2_completion is not None
            else None
        )

    def _prepare_terms(self) -> tuple[_TaylorTerms, ...]:
        cells = self.relation.index.cell_representatives
        prepared: list[_TaylorTerms] = []
        for order, sparse in sorted(self.force_constants.sparse.items()):
            count = len(sparse.sites) * len(cells)
            atoms = np.empty((count, order), dtype=np.int32)
            tensors = np.empty((count,) + (3,) * order, dtype=float)
            row = 0
            zero = np.zeros((1, 3), dtype=np.int32)
            for sites, translations, tensor in zip(
                sparse.sites, sparse.translations, sparse.tensors, strict=True
            ):
                labels = np.vstack((zero, translations))
                for cell in cells:
                    atoms[row] = [
                        self.relation.index.atom(int(site), cell + translation)
                        for site, translation in zip(sites, labels, strict=True)
                    ]
                    tensors[row] = tensor
                    row += 1
            prepared.append(_TaylorTerms(order, atoms, tensors))
        return tuple(prepared)

    def displacement(self, atoms: Atoms) -> np.ndarray:
        """Return periodic Cartesian displacement from the fixed reference."""
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        if not np.array_equal(atoms.pbc, self.relation.reference.pbc):
            raise ValueError("structure PBC differs from the calculator reference")
        return self.relation.displacement(atoms)

    def evaluate_displacement(self, displacement: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate relative energy and forces for Cartesian displacements."""
        u = np.asarray(displacement, dtype=float)
        expected = (len(self.relation.reference), 3)
        if u.shape != expected:
            raise ValueError(f"displacement must have shape {expected}")
        energy = 0.0
        forces = np.zeros_like(u)
        if self._periodic_fc2 is not None:
            forces -= np.einsum("ijab,jb->ia", self._periodic_fc2, u, optimize=True)
            energy += 0.5 * float(
                np.einsum("ia,ijab,jb->", u, self._periodic_fc2, u, optimize=True)
            )
        for terms in self._terms:
            order = terms.order
            denominator = factorial(order)
            for atoms, tensor in zip(terms.atoms, terms.tensors, strict=True):
                operands: list[object] = [tensor, list(range(order))]
                for axis, atom in enumerate(atoms):
                    operands.extend((u[atom], [axis]))
                operands.append([])
                energy += float(np.einsum(*operands, optimize=True)) / denominator
                for axis, atom in enumerate(atoms):
                    derivative: list[object] = [tensor, list(range(order))]
                    for other_axis, other_atom in enumerate(atoms):
                        if other_axis != axis:
                            derivative.extend((u[other_atom], [other_axis]))
                    derivative.append([axis])
                    forces[atom] -= np.einsum(*derivative, optimize=True) / denominator
        return energy, forces

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Evaluate relative energy and forces for an ASE structure."""
        return self.evaluate_displacement(self.displacement(atoms))


__all__ = ["TaylorPotential"]
