"""Test-only helpers for constructing explicit reference frames."""

from ase import Atoms

from mlfcs.core.geometry import StructureRelation
from mlfcs import build_supercell


def make_supercell(primitive: Atoms, matrix: object):
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference)
    return relation.reference, relation.index
