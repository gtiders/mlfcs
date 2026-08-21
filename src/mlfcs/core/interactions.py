"""Shared symmetry-reduced interaction spaces for all calculation backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ase import Atoms

from mlfcs.core.geometry import (
    StructureRelation,
)
from mlfcs.core.orbits import OrbitSpace
from mlfcs.core.real_space import (
    PrimitiveInteractionSpace,
    PrimitiveSymmetryOperations,
    build_primitive_interaction_space,
    realize_orbit_space,
    resolve_primitive_cutoff,
    validate_realization_identifiability,
)
from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.ifc.model import RunConfig


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """Shared structure and symmetry mapping for one top-level calculation."""

    relation: StructureRelation
    primitive_symmetry: PrimitiveSymmetryOperations
    symmetry: SymmetryOperations

    @classmethod
    def from_atoms(
        cls, primitive: Atoms, reference: Atoms, *, symprec: float
    ) -> ReferenceFrame:
        relation = StructureRelation.from_atoms(primitive, reference, tolerance=symprec)
        primitive_symmetry = PrimitiveSymmetryOperations.from_atoms(
            relation.primitive, symprec=symprec
        )
        symmetry = SymmetryOperations.from_primitive_operations(
            primitive_symmetry, relation.index
        )
        return cls(relation, primitive_symmetry, symmetry)


class InteractionSpace:
    """Geometry, symmetry, cutoff, and irreducible clusters for one IFC order."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        order: int,
        reference: Atoms,
        cutoff: float,
        max_body_order: int | None = None,
        symprec: float = 1e-5,
        displacement: float = 0.01,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        frame = ReferenceFrame.from_atoms(atoms, reference, symprec=symprec)
        self._initialize(
            frame,
            order=order,
            cutoff=cutoff,
            max_body_order=max_body_order,
            symprec=symprec,
            displacement=displacement,
            reporter=reporter,
        )

    @classmethod
    def from_frame(
        cls,
        frame: ReferenceFrame,
        *,
        order: int,
        cutoff: float,
        max_body_order: int | None = None,
        symprec: float = 1e-5,
        displacement: float = 0.01,
        reporter: Callable[[str], None] | None = None,
    ) -> InteractionSpace:
        """Construct an order-specific space from a verified shared frame."""
        instance = cls.__new__(cls)
        instance._initialize(
            frame,
            order=order,
            cutoff=cutoff,
            max_body_order=max_body_order,
            symprec=symprec,
            displacement=displacement,
            reporter=reporter,
        )
        return instance

    def _initialize(
        self,
        frame: ReferenceFrame,
        *,
        order: int,
        cutoff: float,
        max_body_order: int | None,
        symprec: float,
        displacement: float,
        reporter: Callable[[str], None] | None,
    ) -> None:
        self.frame = frame
        self.relation = frame.relation
        matrix = self.relation.supercell_matrix
        self.config = RunConfig(
            order=order,
            supercell=tuple(tuple(int(value) for value in row) for row in matrix),
            cutoff=cutoff,
            max_body_order=max_body_order,
            displacement=displacement,
            symprec=symprec,
        )
        self.primitive = self.relation.primitive
        self._reporter = reporter
        self._report(f"Creating reference supercell with matrix {matrix.tolist()}")
        self.supercell = self.relation.reference
        self.index = self.relation.index
        self._report(
            f"- {len(self.primitive)} primitive atoms, {len(self.supercell)} supercell atoms"
        )
        self._report("Resolving the interaction cutoff")
        self.cutoff = resolve_primitive_cutoff(self.primitive, cutoff)
        self._report(f"- Cutoff radius: {self.cutoff:.10f} Å")
        self._report("Analyzing crystal symmetries")
        self.symmetry = frame.symmetry
        self._report(f"- Space group {self.symmetry.symbol}")
        self._report(f"- {self.symmetry.size} symmetry operations")
        self._orbit_space: OrbitSpace | None = None
        self._primitive_orbit_space: PrimitiveInteractionSpace | None = None

    @property
    def primitive_orbit_space(self) -> PrimitiveInteractionSpace:
        if self._primitive_orbit_space is None:
            self._primitive_orbit_space = build_primitive_interaction_space(
                self.primitive,
                order=self.config.order,
                cutoff=self.cutoff,
                max_body_order=self.config.max_body_order,
                symprec=self.config.symprec,
                symmetry=self.frame.primitive_symmetry,
            )
        return self._primitive_orbit_space

    @property
    def orbit_space(self) -> OrbitSpace:
        if self._orbit_space is None:
            self._report(
                f"Finding symmetry-inequivalent order-{self.config.order} interaction clusters"
            )
            self._orbit_space = realize_orbit_space(self.primitive_orbit_space, self.index)
            validate_realization_identifiability(
                self.primitive_orbit_space, self.index, realized=self._orbit_space
            )
            dimensions = sum(orbit.dimension for orbit in self._orbit_space.orbits)
            self._report(f"- {len(self._orbit_space.orbits)} cluster equivalence classes")
            self._report(f"- {dimensions} independent tensor parameters")
            if self.config.max_body_order is not None:
                self._report(f"- Maximum body order: {self.config.max_body_order}")
        return self._orbit_space

    def _report(self, message: str) -> None:
        if self._reporter is not None:
            self._reporter(message)


__all__ = ["InteractionSpace", "ReferenceFrame"]
