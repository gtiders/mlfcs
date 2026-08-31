"""Test-only helpers for constructing explicit reference frames."""

from ase import Atoms

from mlfcs import build_supercell
from mlfcs.structure.relation import StructureRelation


def monoatomic_periodic(symbol: str = "Si", cell_length: float = 4.0) -> Atoms:
    """Return the minimal periodic one-atom test structure."""
    return Atoms(
        symbol,
        scaled_positions=[[0, 0, 0]],
        cell=[[cell_length, 0, 0], [0, cell_length, 0], [0, 0, cell_length]],
        pbc=True,
    )


def make_supercell(primitive: Atoms, matrix: object):
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference)
    return relation.reference, relation.index
