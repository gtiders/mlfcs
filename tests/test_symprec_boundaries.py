"""Boundary behaviour of the single ``symprec`` contract.

The relation accepts a reference supercell when its lattice and its atom mapping are inside
``symprec``, and the comparisons are strict.  These tests walk each threshold from below, from
the boundary itself and from above, on cells where the dimensionless matrix difference and the
angstrom residual disagree -- the case a floating-point ``allclose`` on the matrix would get
wrong in both directions.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.structure.relation import StructureRelation
from mlfcs.tools.supercell import build_supercell

SYMPREC = 1e-5


def _cubic() -> Atoms:
    return bulk("Ar", "fcc", a=5.26)


def _scaled(atoms: Atoms, factor: float) -> Atoms:
    """Return the same structure with every lattice vector scaled."""
    scaled = atoms.copy()
    scaled.set_cell(np.asarray(atoms.cell) * factor, scale_atoms=False)
    scaled.wrap()
    return scaled


def test_a_large_repeat_is_accepted() -> None:
    """A coefficient of 31 is a legal supercell, and its residual is per coefficient."""
    primitive = _cubic()
    reference = build_supercell(primitive, (31, 1, 1))
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    assert relation.supercell_matrix.tolist() == [[31, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert relation.cell_residual == pytest.approx(0.0, abs=1e-12)
    assert relation.position_residual == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("matrix", (
    np.diag((2, 3, 4)),
    np.asarray([[2, 1, 0], [0, 2, 1], [0, 0, 2]]),
    np.asarray([[2, -1, 0], [0, 2, 1], [0, 0, 2]]),
    np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    np.asarray([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
    np.diag((2, 2, 2)) @ np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
))
def test_every_candidate_matrix_is_recovered_exactly(matrix: np.ndarray) -> None:
    """Diagonal, non-diagonal, sheared and large-determinant matrices round-trip."""
    primitive = _cubic()
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    np.testing.assert_array_equal(relation.supercell_matrix, matrix)
    assert relation.cell_residual < SYMPREC
    assert relation.position_residual < SYMPREC


def test_a_tiny_dimensionless_error_that_is_a_large_angstrom_error_is_refused() -> None:
    """The matrix difference is what an allclose would look at; the angstrom residual decides.

    A 1x1x1 cell whose lattice is off by 1e-4 angstrom has a *relative* difference of 2e-5, which
    a loose matrix comparison would wave through, while the physical error is ten times the
    declared precision.
    """
    primitive = _cubic()
    reference = _scaled(primitive, 1.0 + 2e-5)
    with pytest.raises(ValueError) as failure:
        StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    message = str(failure.value)
    assert "angstrom" in message and "residual" in message and "candidate matrix" in message


def test_a_large_dimensionless_error_that_is_inside_symprec_is_accepted() -> None:
    """A 1x1x1 cell with a 1e-6 angstrom error is legal even though the ratio is 2e-7."""
    primitive = _cubic()
    reference = _scaled(primitive, 1.0 + 2e-7)
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    assert 0.0 < relation.cell_residual < SYMPREC


@pytest.mark.parametrize(("offset", "accepted"), (
    (0.0, True),
    (0.5e-5, True),
    (0.99e-5, True),
    (1.01e-5, False),
    (2e-5, False),
))
def test_the_cell_threshold_is_strict(offset: float, accepted: bool) -> None:
    """Half of symprec, just below, just above: the comparison is a strict ``<``."""
    primitive = _cubic()
    length = float(np.linalg.norm(np.asarray(primitive.cell)[0]))
    reference = _scaled(primitive, 1.0 + offset / length)
    if accepted:
        relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
        assert relation.cell_residual < SYMPREC
    else:
        with pytest.raises(ValueError):
            StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)


def test_the_cell_threshold_boundary_itself_is_refused() -> None:
    """Exactly ``symprec`` is not inside ``symprec``."""
    primitive = _cubic()
    length = float(np.linalg.norm(np.asarray(primitive.cell)[0]))
    reference = _scaled(primitive, 1.0 + SYMPREC / length)
    with pytest.raises(ValueError):
        StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)


@pytest.mark.parametrize(("offset", "accepted"), (
    (0.0, True),
    (0.5e-5, True),
    (1.01e-5, False),
    (5e-5, False),
))
def test_the_atom_threshold_is_strict(offset: float, accepted: bool) -> None:
    """The mapping residual has the same strict boundary as the lattice residual."""
    primitive = _cubic()
    reference = build_supercell(primitive, (2, 1, 1))
    reference.positions[1] += np.asarray([offset, 0.0, 0.0])
    if accepted:
        relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
        assert relation.position_residual < SYMPREC
    else:
        with pytest.raises(ValueError) as failure:
            StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
        assert "angstrom" in str(failure.value)


def test_a_mirrored_supercell_is_a_legal_lattice_relation() -> None:
    """A negative determinant is a nonzero determinant; only the size has to match."""
    primitive = _cubic()
    mirrored = primitive.copy()
    mirrored.set_cell(-np.asarray(primitive.cell), scale_atoms=False)
    mirrored.wrap()
    reference = build_supercell(mirrored, (2, 2, 2))
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    np.testing.assert_array_equal(relation.supercell_matrix, -2 * np.eye(3, dtype=int))
    assert relation.position_residual < SYMPREC


def test_a_reference_with_the_wrong_atom_count_is_refused() -> None:
    """The discrete consistency check reports the counts and the matrix."""
    primitive = _cubic()
    reference = build_supercell(primitive, (2, 1, 1))
    reference = reference[:-1]
    with pytest.raises((ValueError, Exception)) as failure:
        StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    assert "matrix" in str(failure.value) or "count" in str(failure.value)


def test_a_non_orthogonal_cell_uses_the_minimum_image() -> None:
    """A tilted cell maps atoms by minimum image, not by plain fractional wrapping."""
    primitive = bulk("Mg", "hcp", a=3.2, c=5.2)
    reference = build_supercell(primitive, (2, 2, 2))
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    assert relation.position_residual < SYMPREC
    assert len(relation.primitive_index) == len(reference)


def test_the_identity_relation_records_the_precision() -> None:
    """The canonical exact-R case needs no float mapping but still declares its precision."""
    primitive = _cubic()
    relation = StructureRelation.identity(primitive, symprec=SYMPREC)
    assert relation.symprec == SYMPREC
    assert relation.supercell_matrix.tolist() == np.eye(3, dtype=int).tolist()
    assert relation.cell_residual == 0.0 and relation.position_residual == 0.0


def test_a_training_frame_uses_the_declared_precision() -> None:
    """The frame check reads ``self.symprec``: displacements stay physical, cells do not."""
    primitive = _cubic()
    reference = build_supercell(primitive, (2, 2, 2))
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)

    frame = reference.copy()
    frame.positions[0] += np.asarray([0.2, 0.0, 0.0])
    displacement = relation.displacement(frame)
    assert float(np.max(np.abs(displacement))) == pytest.approx(0.2, rel=1e-6)

    stretched = reference.copy()
    stretched.set_cell(np.asarray(reference.cell) * (1.0 + 5e-5), scale_atoms=False)
    with pytest.raises(ValueError) as failure:
        relation.displacement(stretched)
    assert "angstrom" in str(failure.value)

    slightly = reference.copy()
    slightly.set_cell(np.asarray(reference.cell) * (1.0 + 1e-6), scale_atoms=False)
    assert float(np.max(np.abs(relation.displacement(slightly)))) < 1e-5


def test_a_reordered_training_frame_is_refused() -> None:
    """Element order is part of the frame identity, not something to silently repair."""
    primitive = bulk("NaCl", "rocksalt", a=5.64)
    reference = build_supercell(primitive, (2, 1, 1))
    relation = StructureRelation.from_atoms(primitive, reference, symprec=SYMPREC)
    reordered = reference[[1, 0]]
    with pytest.raises(ValueError):
        relation.displacement(reordered)
