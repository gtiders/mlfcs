"""Contract tests for the exact integer lattice frame of the orbit algebra.

The frame is the only place where the algebra fixes a lattice basis, so these
tests pin what it must expose: row-vector cells, integer translations that are
never reduced, spglib rotations acting on fractional column vectors, anchored
``(site, translation)`` labels, and a canonical basis that depends on the crystal
alone -- not on the unimodular basis the caller happens to pass in.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.structure.lattice_frame import LatticeFrame, reduction_is_unimodular

MATERIAL_NAMES = ("cubic", "fcc", "hexagonal", "rhombohedral", "diamond")

# A strongly sheared source basis: the lattice is unchanged, the basis is not.
_STRONG_SHEAR = np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)

SOURCE_VARIANTS = {"plain": np.eye(3, dtype=np.int64), "sheared": _STRONG_SHEAR}

# Unimodular source bases, including one of determinant -1.
UNIMODULAR_SOURCES = {
    "shear": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
    "mixed": np.array([[2, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64),
    "mixed-long": np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], dtype=np.int64),
    "left-handed": np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=np.int64),
}

# Integer matrices of genuine rotations of the corresponding lattices.
ROTATIONS = {
    "hexagonal-3-fold": ("hexagonal", np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])),
    "cubic-4-fold": ("cubic", np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
    "fcc-3-fold": ("fcc", np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
}


def _graphene() -> Atoms:
    """Return a two-atom honeycomb cell (the graphene structure motif)."""
    a = 2.46
    cell = np.array(
        [[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, 6.7]],
        dtype=np.float64,
    )
    scaled = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]])
    return Atoms("C2", scaled_positions=scaled, cell=cell, pbc=True)


def _rhombohedral(a: float = 4.0, alpha: float = np.deg2rad(75.0)) -> Atoms:
    """Return a one-atom rhombohedral cell whose vectors pairwise subtend ``alpha``."""
    cosine, sine = np.cos(alpha), np.sin(alpha)
    third = (cosine - cosine**2) / sine
    cell = a * np.array(
        [
            [1.0, 0.0, 0.0],
            [cosine, sine, 0.0],
            [cosine, third, np.sqrt(1.0 - cosine**2 - third**2)],
        ],
        dtype=np.float64,
    )
    return Atoms("Na", scaled_positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)


def _material(name: str) -> Atoms:
    """Return the primitive cell of one test material."""
    if name == "cubic":
        return bulk("Cu", "sc", a=3.6)
    if name == "fcc":
        return bulk("Cu", "fcc", a=3.6)
    if name == "hexagonal":
        return _graphene()
    if name == "rhombohedral":
        return _rhombohedral()
    if name == "diamond":
        return bulk("Si", "diamond", a=5.43)
    raise AssertionError(f"unknown material {name}")


def _integer_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the integer inverse of a unimodular matrix."""
    inverse = np.linalg.inv(matrix.astype(np.float64))
    rounded = np.rint(inverse).astype(np.int64)
    np.testing.assert_allclose(inverse, rounded, atol=1e-10, rtol=0.0)
    return rounded


def _rescaled(atoms: Atoms, transform: np.ndarray, *, wrap: bool = True) -> Atoms:
    """Return the same physical crystal expressed in the basis ``transform @ cell``."""
    scaled = atoms.get_scaled_positions(wrap=False) @ _integer_inverse(transform)
    if wrap:
        scaled = np.mod(scaled, 1.0)
    cell = transform @ np.asarray(atoms.cell, dtype=np.float64)
    return Atoms(numbers=atoms.numbers, scaled_positions=scaled, cell=cell, pbc=True)


def _cartesian_rotation(cell: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return the Cartesian rotation of a spglib (column-convention) rotation."""
    return np.linalg.inv(cell) @ rotation.T @ cell


def _cartesian_contract(tensor: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Contract ``matrix`` into every axis of ``tensor``, as the tensor frame must."""
    order = tensor.ndim
    lower = "abcdef"[:order]
    upper = lower.upper()
    terms = ",".join(f"{low}{up}" for low, up in zip(lower, upper, strict=True))
    return np.einsum(f"{terms},{upper}->{lower}", *(matrix,) * order, tensor)


def _integer_grid(limit: int = 5) -> np.ndarray:
    """Return every integer translation with components in ``[-limit, limit]``."""
    axis = np.arange(-limit, limit + 1, dtype=np.int64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _fractional_samples() -> np.ndarray:
    """Return fractional sample points, including a rational grid and generic values."""
    grid = _integer_grid(2) * 0.25 + 0.031
    generic = np.random.default_rng(20260921).uniform(-2.0, 2.0, size=(16, 3))
    return np.concatenate([grid.astype(np.float64), generic])


def _label_grid(sites: int, limit: int = 3) -> np.ndarray:
    """Return every ``(site, translation)`` label with ``|t| <= limit`` for every site."""
    translations = _integer_grid(limit)
    blocks = [
        np.concatenate(
            [np.full((translations.shape[0], 1), site, dtype=np.int64), translations], axis=1
        )
        for site in range(sites)
    ]
    return np.concatenate(blocks).astype(np.int64)


def _label_points(labels: np.ndarray, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return the Cartesian points of ``(site, translation)`` label rows."""
    return (positions[labels[:, 0]] + labels[:, 1:].astype(np.float64)) @ cell


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_fixture_cells_match_their_documented_lattices(name: str) -> None:
    """Keep the fixtures honest: a wrong fixture would make invariance vacuous."""
    atoms = _material(name)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    assert atoms.pbc.all()
    assert np.linalg.det(cell) != 0.0
    if name == "fcc":
        lengths = np.linalg.norm(cell, axis=1)
        cosines = cell @ cell.T / np.outer(lengths, lengths)
        expected = 0.5 * (np.ones((3, 3)) - np.eye(3)) + np.eye(3)
        np.testing.assert_allclose(cosines, expected, atol=1e-12)
    if name == "rhombohedral":
        lengths = np.linalg.norm(cell, axis=1)
        np.testing.assert_allclose(lengths, lengths[0], atol=1e-12)
        cosines = cell @ cell.T / np.outer(lengths, lengths)
        expected = np.cos(np.deg2rad(75.0)) * (np.ones((3, 3)) - np.eye(3)) + np.eye(3)
        np.testing.assert_allclose(cosines, expected, atol=1e-12)
    for variant in SOURCE_VARIANTS.values():
        assert abs(round(float(np.linalg.det(variant.astype(np.float64))))) == 1


@pytest.mark.parametrize("variant", sorted(SOURCE_VARIANTS))
@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_translation_maps_round_trip_exactly(name: str, variant: str) -> None:
    """Integer translations must survive the frame change exactly and stay unreduced."""
    atoms = _rescaled(_material(name), SOURCE_VARIANTS[variant])
    frame = LatticeFrame.from_atoms(atoms)
    translations = _integer_grid()
    mapped = frame.source_to_algebra_translations(translations)
    assert mapped.dtype == np.int64
    assert np.array_equal(frame.algebra_to_source_translations(mapped), translations)
    # An unreduced integer vector must still describe the same Cartesian offset.
    np.testing.assert_allclose(
        mapped @ frame.algebra_cell,
        translations @ np.asarray(atoms.cell, dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("variant", sorted(SOURCE_VARIANTS))
@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_position_maps_round_trip_modulo_lattice_vectors(name: str, variant: str) -> None:
    """Fractional coordinates must round trip up to lattice translations."""
    atoms = _rescaled(_material(name), SOURCE_VARIANTS[variant])
    frame = LatticeFrame.from_atoms(atoms)
    points = _fractional_samples()
    algebra = frame.source_to_algebra_positions(points)
    round_tripped = frame.algebra_to_source_positions(algebra)
    delta = round_tripped - points
    np.testing.assert_allclose(delta - np.rint(delta), 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(round_tripped % 1.0, points % 1.0, atol=1e-12, rtol=0.0)
    # The remapped coordinates must describe the same Cartesian points.
    np.testing.assert_allclose(
        algebra @ frame.algebra_cell,
        points @ np.asarray(atoms.cell, dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    )


def test_maps_accept_arbitrary_leading_batch_shapes() -> None:
    """Every map acts on the trailing axis of any batch shape."""
    frame = LatticeFrame.from_atoms(_material("diamond"))
    translations = np.random.default_rng(3).integers(-4, 5, size=(2, 3, 4, 3))
    mapped = frame.source_to_algebra_translations(translations)
    assert mapped.shape == (2, 3, 4, 3)
    assert np.array_equal(frame.algebra_to_source_translations(mapped), translations)
    points = np.random.default_rng(4).uniform(-1.0, 1.0, size=(2, 5, 3))
    algebra = frame.source_to_algebra_positions(points)
    assert algebra.shape == (2, 5, 3)
    delta = frame.algebra_to_source_positions(algebra) - points
    np.testing.assert_allclose(delta - np.rint(delta), 0.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("rotation", sorted(ROTATIONS))
def test_rotation_conjugation_matches_the_cartesian_rotation(rotation: str) -> None:
    """Mapping a rotation must keep the physical Cartesian rotation invariant."""
    name, matrix = ROTATIONS[rotation]
    frame = LatticeFrame.from_atoms(_material(name))
    mapped = frame.source_to_algebra_rotation(matrix)
    assert np.issubdtype(mapped.dtype, np.integer)
    cartesian = _cartesian_rotation(frame.source_cell, matrix)
    np.testing.assert_allclose(
        _cartesian_rotation(frame.algebra_cell, mapped), cartesian, atol=1e-10, rtol=0.0
    )
    # The fixtures must actually be lattice rotations, not arbitrary integers.
    np.testing.assert_allclose(cartesian @ cartesian.T, np.eye(3), atol=1e-10, rtol=0.0)
    assert np.array_equal(frame.algebra_to_source_rotation(mapped), matrix)


@pytest.mark.parametrize("order", (2, 3, 4))
@pytest.mark.parametrize("name", ("diamond", "rhombohedral"))
def test_tensor_frame_is_the_kronecker_power_of_the_transposed_cell(name: str, order: int) -> None:
    """The tensor frame must be ``kron(cell.T, order times)`` in that exact axis order."""
    frame = LatticeFrame.from_atoms(_material(name))
    transposed = frame.algebra_cell.T
    expected = transposed
    for _ in range(order - 1):
        expected = np.kron(expected, transposed)
    assert frame.tensor_frame(order).shape == (3**order, 3**order)
    np.testing.assert_allclose(frame.tensor_frame(order), expected, atol=1e-12, rtol=0.0)
    tensor = np.random.default_rng(7).standard_normal((3,) * order)
    flat = frame.tensor_frame(order) @ np.ravel(tensor)
    np.testing.assert_allclose(
        flat.reshape((3,) * order),
        _cartesian_contract(tensor, transposed),
        atol=1e-10,
        rtol=0.0,
    )


def test_tensor_frame_order_one_matches_row_vector_positions() -> None:
    """Pin the row-vector convention: ``cell.T @ t`` is the Cartesian image of ``t @ cell``."""
    frame = LatticeFrame.from_atoms(_material("rhombohedral"))
    translation = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(
        frame.tensor_frame(1) @ translation,
        translation @ frame.algebra_cell,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(frame.tensor_frame(0), np.eye(1), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("source", sorted(UNIMODULAR_SOURCES))
@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_canonical_frame_is_invariant_under_unimodular_source_bases(name: str, source: str) -> None:
    """Re-expressing the same crystal in another unimodular basis must change nothing."""
    transform = UNIMODULAR_SOURCES[source]
    reference = LatticeFrame.from_atoms(_material(name))
    sheared_atoms = _rescaled(_material(name), transform)
    sheared = LatticeFrame.from_atoms(sheared_atoms)
    np.testing.assert_allclose(sheared.algebra_cell, reference.algebra_cell, atol=1e-10, rtol=0.0)
    np.testing.assert_array_equal(
        sheared.source_to_algebra @ transform, reference.source_to_algebra
    )
    np.testing.assert_allclose(sheared.positions, reference.positions, atol=1e-10, rtol=0.0)
    np.testing.assert_array_equal(sheared.numbers, reference.numbers)
    np.testing.assert_array_equal(sheared.atom_map, reference.atom_map)
    np.testing.assert_allclose(
        sheared.source_to_algebra @ np.asarray(sheared_atoms.cell, dtype=np.float64),
        reference.algebra_cell,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        sheared.positions @ sheared.algebra_cell,
        reference.positions @ reference.algebra_cell,
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("source", sorted(UNIMODULAR_SOURCES))
def test_rotation_map_is_invariant_under_unimodular_source_bases(source: str) -> None:
    """One physical rotation has one image, whichever unimodular basis describes it."""
    transform = UNIMODULAR_SOURCES[source]
    _, rotation = ROTATIONS["hexagonal-3-fold"]
    reference = LatticeFrame.from_atoms(_material("hexagonal"))
    sheared = LatticeFrame.from_atoms(_rescaled(_material("hexagonal"), transform))
    rescaled_rotation = _integer_inverse(transform).T @ rotation @ transform.T
    np.testing.assert_array_equal(
        sheared.source_to_algebra_rotation(rescaled_rotation),
        reference.source_to_algebra_rotation(rotation),
    )


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_wrapped_and_unwrapped_source_motifs_give_the_same_frame(name: str) -> None:
    """Wrapping the source motif must not change the frame or the atom correspondence."""
    atoms = _material(name)
    unwrapped = Atoms(
        numbers=atoms.numbers,
        scaled_positions=atoms.get_scaled_positions(wrap=False) - 2.0,
        cell=np.asarray(atoms.cell, dtype=np.float64),
        pbc=True,
    )
    reference = LatticeFrame.from_atoms(atoms)
    shifted = LatticeFrame.from_atoms(unwrapped)
    np.testing.assert_allclose(shifted.algebra_cell, reference.algebra_cell, atol=0.0, rtol=0.0)
    assert np.array_equal(shifted.source_to_algebra, reference.source_to_algebra)
    np.testing.assert_allclose(shifted.positions, reference.positions, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        shifted.source_positions, reference.source_positions, atol=1e-12, rtol=0.0
    )
    assert np.array_equal(shifted.atom_map, reference.atom_map)


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_frame_reports_an_exactly_unimodular_reduction(name: str) -> None:
    """Every constructed frame must be an exact unimodular change of basis."""
    atoms = _material(name)
    frame = LatticeFrame.from_atoms(atoms)
    assert reduction_is_unimodular(frame)
    assert frame.source_to_algebra.dtype == np.int64
    assert frame.algebra_to_source.dtype == np.int64
    assert frame.numbers.dtype == np.int32
    assert frame.atom_map.dtype == np.int32
    assert abs(round(float(np.linalg.det(frame.source_to_algebra)))) == 1
    np.testing.assert_array_equal(
        frame.source_to_algebra @ frame.algebra_to_source, np.eye(3, dtype=np.int64)
    )
    np.testing.assert_allclose(
        frame.source_to_algebra @ np.asarray(atoms.cell, dtype=np.float64),
        frame.algebra_cell,
        atol=1e-10,
        rtol=0.0,
    )
    assert np.linalg.det(frame.algebra_cell) > 0.0
    assert not frame.positions.flags.writeable
    assert not frame.source_positions.flags.writeable


def test_non_unimodular_frame_is_not_reported() -> None:
    """The diagnostic must reject a frame whose stored map is not unimodular."""
    frame = LatticeFrame.from_atoms(_material("cubic"))
    stretched = replace(
        frame,
        source_to_algebra=np.diag([2, 1, 1]).astype(np.int64),
        algebra_to_source=np.eye(3, dtype=np.int64),
    )
    assert not reduction_is_unimodular(stretched)
    inconsistent = replace(frame, algebra_to_source=np.eye(3, dtype=np.int64))
    assert not reduction_is_unimodular(inconsistent)


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_algebra_atoms_reproduces_the_source_structure(name: str) -> None:
    """Canonical atoms must describe the same physical crystal as the source cell."""
    atoms = _material(name)
    frame = LatticeFrame.from_atoms(atoms)
    canonical = frame.algebra_atoms()
    np.testing.assert_allclose(canonical.cell, frame.algebra_cell, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        canonical.get_scaled_positions(wrap=False), frame.positions, atol=1e-12, rtol=0.0
    )
    np.testing.assert_array_equal(canonical.numbers, frame.numbers)
    assert canonical.pbc.all()
    # Each canonical atom sits on its source atom, up to source lattice vectors.
    delta = canonical.positions - atoms.positions[frame.atom_map]
    fractional = delta @ np.linalg.inv(np.asarray(atoms.cell, dtype=np.float64))
    np.testing.assert_allclose(fractional, np.rint(fractional), atol=1e-10, rtol=0.0)
    # ``positions[i]`` and ``source_positions[i]`` are the same atom.
    np.testing.assert_allclose(
        frame.source_positions,
        atoms.get_scaled_positions(wrap=True)[frame.atom_map],
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("variant", sorted(SOURCE_VARIANTS))
@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_source_labels_describe_the_same_cartesian_atoms(name: str, variant: str) -> None:
    """Mapped labels must be exact integers describing the same physical atoms."""
    atoms = _rescaled(_material(name), SOURCE_VARIANTS[variant])
    frame = LatticeFrame.from_atoms(atoms)
    labels = _label_grid(frame.positions.shape[0])
    mapped = frame.source_labels(labels)
    assert mapped.shape == labels.shape
    assert mapped.dtype == np.int64
    source_sites = mapped[:, 0]
    np.testing.assert_array_equal(atoms.numbers[source_sites], frame.numbers[labels[:, 0]])
    # The translation must be exact: both rows describe the same Cartesian point.
    np.testing.assert_allclose(
        _label_points(mapped, frame.source_positions, np.asarray(atoms.cell, dtype=np.float64)),
        _label_points(labels, frame.positions, frame.algebra_cell),
        atol=1e-10,
        rtol=0.0,
    )
    # ... and it must not have been wrapped or reduced: |t| <= 3 maps beyond 3.
    assert np.abs(mapped[:, 1:]).max() >= 3


def test_source_labels_accept_batch_shapes_and_validate_inputs() -> None:
    """Labels are ``(..., 4)`` rows and reject non-integral or out-of-range input."""
    frame = LatticeFrame.from_atoms(_material("diamond"))
    labels = _label_grid(2, limit=1).reshape(3, 18, 4)
    mapped = frame.source_labels(labels)
    assert mapped.shape == (3, 18, 4)
    assert np.array_equal(
        mapped.reshape(-1, 4),
        frame.source_labels(labels.reshape(-1, 4)),
    )
    with pytest.raises(ValueError, match="length 4"):
        frame.source_labels(np.zeros((2, 5), dtype=np.int64))
    with pytest.raises(ValueError, match="integers"):
        frame.source_labels(np.array([[0.5, 0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="range"):
        frame.source_labels(np.array([[frame.positions.shape[0], 0, 0, 0]], dtype=np.int64))


@pytest.mark.parametrize("source", sorted(UNIMODULAR_SOURCES))
@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_source_labels_compose_with_a_sheared_source_cell(name: str, source: str) -> None:
    """Labels of two frames for the same crystal agree once the shear is composed."""
    transform = UNIMODULAR_SOURCES[source]
    reference_atoms = _material(name)
    sheared_atoms = _rescaled(_material(name), transform, wrap=True)
    reference = LatticeFrame.from_atoms(reference_atoms)
    sheared = LatticeFrame.from_atoms(sheared_atoms)
    labels = _label_grid(reference.positions.shape[0], limit=1)
    reference_mapped = reference.source_labels(labels)
    sheared_mapped = sheared.source_labels(labels)
    np.testing.assert_array_equal(reference_mapped[:, 0], sheared_mapped[:, 0])
    expected = (
        reference.source_positions[reference_mapped[:, 0]] + reference_mapped[:, 1:]
    ) @ _integer_inverse(transform)
    actual = sheared.source_positions[sheared_mapped[:, 0]] + sheared_mapped[:, 1:]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        actual @ np.asarray(sheared_atoms.cell, dtype=np.float64),
        expected @ np.asarray(sheared_atoms.cell, dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        _label_points(sheared_mapped, sheared.source_positions, sheared.source_cell),
        _label_points(reference_mapped, reference.source_positions, reference.source_cell),
        atol=1e-10,
        rtol=0.0,
    )


def test_source_labels_keep_atoms_near_the_cell_boundary_exact() -> None:
    """An atom wrapped onto 1 - 1e-7 must still get the exact integer translation."""
    atoms = bulk("Si", "diamond", a=5.43)
    scaled = atoms.get_scaled_positions(wrap=True)
    scaled[0] = np.mod(scaled[0] + np.array([0.0, 0.0, 1.0 - 1e-7]), 1.0)
    boundary = Atoms(
        numbers=atoms.numbers,
        scaled_positions=scaled,
        cell=np.asarray(atoms.cell, dtype=np.float64),
        pbc=True,
    )
    frame = LatticeFrame.from_atoms(boundary)
    boundary_atom = int(np.flatnonzero(frame.source_positions[:, 2] > 0.999)[0])
    assert frame.source_positions[boundary_atom, 2] == pytest.approx(1.0 - 1e-7, abs=1e-12)
    labels = _label_grid(frame.positions.shape[0], limit=2)
    mapped = frame.source_labels(labels)
    np.testing.assert_allclose(
        _label_points(mapped, frame.source_positions, frame.source_cell),
        _label_points(labels, frame.positions, frame.algebra_cell),
        atol=1e-10,
        rtol=0.0,
    )
    # The near-boundary atom must not pick up an off-by-one on its own label.
    own = np.array([[boundary_atom, 0, 0, 0]], dtype=np.int64)
    np.testing.assert_allclose(
        _label_points(frame.source_labels(own), frame.source_positions, frame.source_cell),
        _label_points(own, frame.positions, frame.algebra_cell),
        atol=1e-10,
        rtol=0.0,
    )


def test_source_labels_reject_inconsistent_frames() -> None:
    """A non-integral residual means the two stored frames describe different lattices."""
    frame = LatticeFrame.from_atoms(_material("cubic"))
    broken = replace(frame, source_to_algebra=np.eye(3) * 0.5, algebra_to_source=np.eye(3) * 2.0)
    labels = np.array([[0, 1, 0, 0]], dtype=np.int64)
    with pytest.raises(RuntimeError, match="not reachable"):
        broken.source_labels(labels)


def test_non_periodic_and_singular_cells_are_rejected() -> None:
    """The frame is only defined for fully periodic, non-singular primitive cells."""
    with pytest.raises(ValueError, match="periodic"):
        LatticeFrame.from_atoms(Atoms("H", cell=np.eye(3) * 3.0, pbc=(True, True, False)))
    with pytest.raises(ValueError, match="non-singular"):
        LatticeFrame.from_atoms(Atoms("H", cell=np.diag([3.0, 3.0, 0.0]), pbc=True))


def test_colliding_atoms_are_rejected() -> None:
    """Two atoms of one species on the same site leave the canonical order undefined."""
    cell = np.eye(3) * 3.0
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], cell=cell, pbc=True)
    with pytest.raises(RuntimeError, match="collide"):
        LatticeFrame.from_atoms(atoms)


def test_maps_validate_their_inputs() -> None:
    """Integrality and trailing shape are part of the contract of every map."""
    frame = LatticeFrame.from_atoms(_material("cubic"))
    with pytest.raises(ValueError, match="integers"):
        frame.source_to_algebra_rotation(np.eye(3) * 0.5)
    with pytest.raises(ValueError, match="shape"):
        frame.source_to_algebra_rotation(np.eye(2))
    with pytest.raises(ValueError, match="integers"):
        frame.source_to_algebra_translations(np.array([[0.5, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="length 3"):
        frame.algebra_to_source_translations(np.array([[0, 0, 0, 0]]))
    with pytest.raises(ValueError, match="length 3"):
        frame.algebra_to_source_positions(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        frame.tensor_frame(-1)
    # Integral floating point input is accepted and mapped exactly.
    assert np.array_equal(
        frame.source_to_algebra_rotation(np.eye(3)),
        frame.source_to_algebra_rotation(np.eye(3, dtype=np.int64)),
    )
    assert np.array_equal(
        frame.source_to_algebra_translations(np.array([[2.0, -3.0, 0.0]])),
        frame.source_to_algebra_translations(np.array([[2, -3, 0]])),
    )
