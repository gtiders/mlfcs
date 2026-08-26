"""Shared symmetry-reduced interaction spaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ase import Atoms

from mlfcs.interactions.models import PrimitiveInteractionSpace, RealizedInteractionSpace
from mlfcs.interactions.primitive.builder import build_primitive_interaction_space
from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
from mlfcs.interactions.realization import (
    realize_interaction_space,
)
from mlfcs.structure.relation import (
    StructureRelation,
)
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations, SymmetryOperations

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InteractionSettings:
    """Order-specific interaction construction settings in ASE units."""

    order: int
    supercell: object
    cutoff: float | int | None
    max_body_order: int | None = None
    displacement: float = 0.01
    symprec: float = 1e-5

    def __post_init__(self) -> None:
        if self.order < 2:
            raise ValueError("order must be at least 2")
        if self.cutoff == 0:
            raise ValueError("cutoff cannot be zero")
        if self.max_body_order is not None and not 1 <= self.max_body_order <= self.order:
            raise ValueError("max_body_order must be between 1 and order")
        if self.displacement <= 0:
            raise ValueError("displacement must be positive")


@dataclass(frozen=True, slots=True)
class ReferenceFrame:
    """Shared structure and symmetry mapping for one top-level calculation."""

    relation: StructureRelation
    primitive_symmetry: PrimitiveSymmetryOperations
    symmetry: SymmetryOperations

    @classmethod
    def from_atoms(cls, primitive: Atoms, reference: Atoms, *, symprec: float) -> ReferenceFrame:
        relation = StructureRelation.from_atoms(primitive, reference, tolerance=symprec)
        primitive_symmetry = PrimitiveSymmetryOperations.from_atoms(
            relation.primitive, symprec=symprec
        )
        symmetry = SymmetryOperations.from_primitive_operations(primitive_symmetry, relation.index)
        return cls(relation, primitive_symmetry, symmetry)


class InteractionSpace:
    """Geometry, symmetry, cutoff, and irreducible clusters for one IFC order."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        order: int,
        reference: Atoms,
        cutoff: float | None,
        max_body_order: int | None = None,
        symprec: float = 1e-5,
        displacement: float = 0.01,
    ) -> None:
        frame = ReferenceFrame.from_atoms(atoms, reference, symprec=symprec)
        self._initialize(
            frame,
            order=order,
            cutoff=cutoff,
            max_body_order=max_body_order,
            symprec=symprec,
            displacement=displacement,
        )

    @classmethod
    def from_frame(
        cls,
        frame: ReferenceFrame,
        *,
        order: int,
        cutoff: float | None,
        max_body_order: int | None = None,
        symprec: float = 1e-5,
        displacement: float = 0.01,
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
        )
        return instance

    def _initialize(
        self,
        frame: ReferenceFrame,
        *,
        order: int,
        cutoff: float | None,
        max_body_order: int | None,
        symprec: float,
        displacement: float,
    ) -> None:
        self.frame = frame
        self.relation = frame.relation
        matrix = self.relation.supercell_matrix
        self.config = InteractionSettings(
            order=order,
            supercell=tuple(tuple(int(value) for value in row) for row in matrix),
            cutoff=cutoff,
            max_body_order=max_body_order,
            displacement=displacement,
            symprec=symprec,
        )
        self.primitive = self.relation.primitive
        logger.info("Creating reference supercell with matrix %s", matrix.tolist())
        self.supercell = self.relation.reference
        self.index = self.relation.index
        logger.info(
            "%d primitive atoms, %d supercell atoms", len(self.primitive), len(self.supercell)
        )
        logger.info("Resolving the interaction cutoff")
        self.cutoff = resolve_primitive_cutoff(self.primitive, cutoff, reference=self.supercell)
        logger.info("Cutoff radius: %.10f Å", self.cutoff)
        logger.info("Analyzing crystal symmetries")
        self.symmetry = frame.symmetry
        logger.info("Space group %s", self.symmetry.symbol)
        logger.info("%d symmetry operations", self.symmetry.size)
        self._realized_orbit_space: RealizedInteractionSpace | None = None
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
    def realized_orbit_space(self) -> RealizedInteractionSpace:
        if self._realized_orbit_space is None:
            logger.info(
                "Finding symmetry-inequivalent order-%d interaction clusters", self.config.order
            )
            self._realized_orbit_space = realize_interaction_space(
                self.primitive_orbit_space, self.index
            )
            dimensions = sum(orbit.dimension for orbit in self._realized_orbit_space.orbits)
            logger.info("%d cluster equivalence classes", len(self._realized_orbit_space.orbits))
            logger.info("%d independent tensor parameters", dimensions)
            if self.config.max_body_order is not None:
                logger.info("Maximum body order: %d", self.config.max_body_order)
        return self._realized_orbit_space


__all__ = ["InteractionSettings", "InteractionSpace", "ReferenceFrame"]
