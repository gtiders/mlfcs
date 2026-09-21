"""Canonical, serializable identity for finite-difference displacement plans.

A finite-difference workflow is split in two halves that can be separated by hours or by
days: :meth:`~mlfcs.finite_difference.calculation.FiniteDifferenceCalculation.sow`
produces displaced structures, an external code (VASP, Quantum ESPRESSO, a NEP
potential, a batch queue) returns forces much later, and only then are force constants
reconstructed.  Between those halves the Python object that owned the plan is usually
gone and the forces come back as files on disk.

That split is exactly where a silently wrong answer becomes possible.  Two
symmetry-reduced plans of the same crystal, order, cutoff and displacement can require
the same *number* of displacements while measuring *different* component rows: the
number of configurations says nothing about which observation rows the plan observed.
The previous API wrote only a zero-based ``mlfcs_configuration_id`` into each displaced
structure, so forces collected from an older plan were accepted positionally by a newer
plan and interpreted as belonging to it; the reconstruction then expanded a wrong set of
parameters and nothing in the result said so.

This module closes that hole.  A plan describes itself with a
:class:`DisplacementManifest`, whose :attr:`DisplacementManifest.fingerprint` is a
SHA-256 hash of the complete plan description.  Forces are accepted only together with
the fingerprint of the plan that produced the displacements (:class:`ForceBatch`); a
batch that cannot prove its provenance is rejected instead of guessed at.

How the fingerprint is computed
===============================

:attr:`DisplacementManifest.fingerprint` is
``hashlib.sha256(text.encode("utf-8")).hexdigest()`` where ``text`` is one canonical
JSON document:

1. The document holds exactly one entry per :class:`DisplacementManifest` field except
   ``fingerprint`` itself::

       schema_version, primitive_fingerprint, reference_fingerprint, order,
       cutoff_angstrom, max_body_order, symprec, displacement_angstrom,
       derivative_backend, stencil_signs, extrapolation, orbits, displacement_keys,
       configurations

   ``json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)``
   serializes it: the documented key *set* above appears in sorted lexical order, with
   no insignificant whitespace and no locale-dependent escaping.  The hash is therefore
   a function of the documented fields only; nothing about object layout, dictionary
   insertion order, ``id()`` values, memory layout or ``repr()`` forms enters it.
2. Floats are encoded with :meth:`float.hex`, which is exact and independent of
   platform, endianness and the float-to-decimal algorithm.  The raw bytes of a
   ``np.float64`` are never hashed, because they depend on the endianness of the machine
   that wrote them.  Integers are decimal, strings and enums are hashed as written, and
   ``None`` (for ``max_body_order`` and for a central backend's ``extrapolation``) is
   JSON ``null``.
3. Structure identity is :func:`structure_fingerprint`: SHA-256 over one document of
   the same canonical form, holding the atom numbers, the cell entries as hex, the
   *wrapped* scaled positions as hex and the periodic boundary flags (documented here as
   the key set; the byte order is the sorted one from rule 1).  The identity is exact: a
   reference whose coordinates differ in the last bit is a different plan, because the
   orbit algebra and the reconstruction solve depend on exactly those bits.  Wrapping
   into the cell is part of the definition, so two structures differing only by a lattice
   translation are the same reference.
4. :meth:`DisplacementManifest.from_json` recomputes the fingerprint from the decoded
   fields and raises :class:`ValueError` when the stored value disagrees, so a
   hand-edited or corrupted manifest cannot be loaded.  ``fingerprint`` is a *derived*
   field: a constructor may leave it empty, and ``__post_init__`` always recomputes it,
   so an in-memory manifest can never disagree with its own document.

Plan layout versus plan identity
================================

:attr:`DisplacementManifest.displacement_keys` and
:attr:`DisplacementManifest.configurations` describe the layout the forces must follow;
:attr:`DisplacementManifest.orbits` describes what the plan *measures*.  Both enter the
fingerprint, so a plan that reorders or reselects observation rows of the same crystal
with the same configuration count is a different plan, and its force batch is refused by
the first plan.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from hashlib import sha256
from os import PathLike
from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs.finite_difference.sampling import DisplacementConfiguration, DisplacementPlan
from mlfcs.interactions.algebra.actions import TensorAction
from mlfcs.interactions.models import (
    PrimitiveInteractionOrbit,
    RealizedInteractionOrbit,
    RealizedInteractionSpace,
    RealizedOrbitImage,
)

#: Version of the fingerprint document and of the manifest layout.  It is hashed with
#: every manifest, so a force batch collected under one schema can never be replayed
#: against a plan of another schema.  Bump it whenever the document below changes.
PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OrbitIdentity:
    """The observation model of one symmetry-inequivalent interaction orbit.

    ``exact_lattice_basis`` holds the canonical *integer* columns of the primitive orbit
    the realized orbit was built from (the realized orbit keeps the primitive Cartesian
    basis and observation rows verbatim, so the two orbit spaces pair index by index).
    ``images`` is one ``(cluster, permutation)`` pair per symmetry image: the concrete
    supercell atom indices and the IFC-axis permutation of its tensor action.
    """

    representative: tuple[int, ...]
    dimension: int
    observation_rows: tuple[int, ...]
    exact_lattice_basis: tuple[tuple[int, ...], ...]
    images: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class ConfigurationIdentity:
    """One required force calculation and the stencil signs that built it.

    ``atoms`` and ``directions`` repeat the displacement key in key order, and
    ``step_angstrom`` is the exact displacement magnitude of this configuration
    (``max(abs(displacement))``), which is the plan step for a central plan and the
    subplan step for one step of an extrapolation grid.
    """

    configuration_id: int
    key: tuple[tuple[int, int], ...]
    atoms: tuple[int, ...]
    directions: tuple[int, ...]
    signs: tuple[int, ...]
    step_angstrom: float


@dataclass(frozen=True, slots=True)
class DisplacementManifest:
    """The complete, serializable description and identity of one displacement plan.

    Every field except ``fingerprint`` is an independent input; ``fingerprint`` is
    derived from all of them, so a constructor may leave it empty and ``__post_init__``
    recomputes it.  An in-memory manifest can therefore never disagree with its own
    document, and :meth:`from_json` compares the stored value against the recomputed one.
    The module docstring documents the exact document that is hashed.
    """

    schema_version: int
    primitive_fingerprint: str
    reference_fingerprint: str
    order: int
    cutoff_angstrom: float
    max_body_order: int | None
    symprec: float
    displacement_angstrom: float
    derivative_backend: str
    stencil_signs: tuple[tuple[int, ...], ...]
    extrapolation: dict | None
    orbits: tuple[OrbitIdentity, ...]
    displacement_keys: tuple[tuple[tuple[int, int], ...], ...]
    configurations: tuple[ConfigurationIdentity, ...]
    fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "extrapolation", _normalized_extrapolation(self.extrapolation))
        object.__setattr__(self, "fingerprint", _fingerprint(_payload(self)))

    def to_json(self) -> str:
        """str : the canonical single-line JSON document including the fingerprint."""
        return _canonical_json({**_payload(self), "fingerprint": self.fingerprint})

    @classmethod
    def from_json(cls, text: str) -> DisplacementManifest:
        """Rebuild a manifest from :meth:`to_json`, verifying its fingerprint.

        Raises :class:`ValueError` when the text is not valid JSON, when the document
        does not match the manifest schema, or when its stored fingerprint disagrees with
        the recomputed one (an edited or corrupted manifest).
        """
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"displacement manifest is not valid JSON: {error}") from error
        try:
            stored = _stored_fingerprint(document)
            manifest = cls(**_decode_fields(document))
        except TypeError as error:
            raise ValueError(f"the displacement manifest document is malformed: {error}") from error
        if manifest.fingerprint != stored:
            raise ValueError(
                f"displacement manifest fingerprint mismatch: stored {stored}, "
                f"recomputed {manifest.fingerprint}; the manifest was edited or corrupted"
            )
        return manifest

    def save(self, path: str | PathLike[str]) -> None:
        """Write :meth:`to_json` to ``path`` as one newline-terminated UTF-8 line."""
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | PathLike[str]) -> DisplacementManifest:
        """Read a manifest written by :meth:`save`, verifying its fingerprint."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class DisplacementBatch:
    """The displaced structures of one plan together with the plan identity.

    Construction is the provenance check: every structure must carry the manifest
    fingerprint and its own positional configuration id in ``info``.
    """

    manifest: DisplacementManifest
    structures: tuple[Atoms, ...]

    def __post_init__(self) -> None:
        expected = len(self.manifest.configurations)
        if len(self.structures) != expected:
            raise ValueError(
                f"the plan has {expected} configurations but the batch carries "
                f"{len(self.structures)} structures"
            )
        for index, atoms in enumerate(self.structures):
            identity = atoms.info.get("mlfcs_configuration_id")
            if identity != index:
                raise ValueError(
                    f"structure {index} carries mlfcs_configuration_id {identity}; the plan "
                    "order is positional and every structure must carry its own index"
                )
            fingerprint = atoms.info.get("mlfcs_plan_fingerprint")
            if fingerprint != self.manifest.fingerprint:
                raise ValueError(
                    f"structure {index} carries plan fingerprint {fingerprint}, not the "
                    f"plan fingerprint {self.manifest.fingerprint}"
                )

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self) -> Iterator[Atoms]:
        return iter(self.structures)


@dataclass(frozen=True, slots=True)
class ForceBatch:
    """Forces returned by an external code, bound to the plan that asked for them.

    ``configuration_ids`` names the configuration each force row belongs to, in any
    order and in any id set: the receiving calculation validates the ids and reorders the
    rows before differentiating.  ``schema_version`` is the manifest schema the forces
    were collected under.
    """

    fingerprint: str
    configuration_ids: tuple[int, ...]
    forces: np.ndarray
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = np.asarray(self.forces, dtype=float)
        object.__setattr__(self, "forces", values)
        object.__setattr__(self, "configuration_ids", _as_configuration_ids(self.configuration_ids))
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError(
                f"forces must have shape (n_configurations, n_atoms, 3), got {values.shape}"
            )
        if len(self.configuration_ids) != len(values):
            raise ValueError(
                f"the force batch carries {len(self.configuration_ids)} configuration ids "
                f"but {len(values)} force rows"
            )


def _as_configuration_ids(values: Sequence[object]) -> tuple[int, ...]:
    """Return configuration ids as plain integers, refusing anything a float could hide.

    Truncating ``0.9`` to ``0`` would silently reorder forces, which is exactly the
    misattribution this module exists to prevent.
    """
    identifiers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"configuration ids must be integers, got {type(value).__name__}")
        identifiers.append(int(value))
    return tuple(identifiers)


def structure_fingerprint(atoms: Atoms) -> str:
    """Return the exact SHA-256 identity of one structure.

    The hashed document is ``numbers``, ``cell`` (row-major hex), the *wrapped* scaled
    positions (row-major hex) and ``pbc``, canonicalized as described in the module
    docstring.
    """
    document = {
        "numbers": [int(number) for number in atoms.numbers],
        "cell": [[_hex_float(value) for value in row] for row in np.asarray(atoms.cell, float)],
        "scaled_positions": [
            [_hex_float(value) for value in row]
            for row in np.asarray(atoms.get_scaled_positions(wrap=True), float)
        ],
        "pbc": [bool(value) for value in atoms.pbc],
    }
    return sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def build_manifest(
    plan: DisplacementPlan,
    orbit_space: RealizedInteractionSpace,
    *,
    primitive: Atoms,
    reference: Atoms,
    order: int,
    cutoff: float,
    max_body_order: int | None,
    symprec: float,
    displacement: float,
    primitive_orbits: Sequence[PrimitiveInteractionOrbit],
    derivative_backend: str = "central",
    extrapolation: Mapping[str, object] | None = None,
) -> DisplacementManifest:
    """Describe and fingerprint one displacement plan.

    ``plan`` supplies the supercell, the configurations and the stencil signs;
    ``orbit_space`` with ``primitive_orbits`` supplies what the plan measures.  For an
    extrapolated workflow ``plan`` is the top-level plan whose configurations are the
    concatenation of every subplan, and ``extrapolation`` describes the grid.
    """
    if derivative_backend not in ("central", "extrapolate"):
        raise ValueError("derivative_backend must be 'central' or 'extrapolate'")
    if (derivative_backend == "extrapolate") != (extrapolation is not None):
        raise ValueError(
            "derivative_backend='extrapolate' requires an extrapolation description, and "
            "derivative_backend='central' must not carry one"
        )
    if len(primitive_orbits) != len(orbit_space.orbits):
        raise ValueError(
            f"{len(orbit_space.orbits)} realized orbits do not pair with "
            f"{len(primitive_orbits)} primitive orbits"
        )
    return DisplacementManifest(
        schema_version=PLAN_SCHEMA_VERSION,
        primitive_fingerprint=structure_fingerprint(primitive),
        reference_fingerprint=structure_fingerprint(reference),
        order=int(order),
        cutoff_angstrom=float(cutoff),
        max_body_order=None if max_body_order is None else int(max_body_order),
        symprec=float(symprec),
        displacement_angstrom=float(displacement),
        derivative_backend=derivative_backend,
        stencil_signs=tuple(tuple(int(sign) for sign in row) for row in plan.stencil.signs),
        extrapolation=None if extrapolation is None else dict(extrapolation),
        orbits=tuple(
            _orbit_identity(realized, primitive_orbit)
            for realized, primitive_orbit in zip(orbit_space.orbits, primitive_orbits, strict=True)
        ),
        displacement_keys=tuple(orbit_space.displacement_keys),
        configurations=tuple(
            _configuration_identity(index, configuration)
            for index, configuration in enumerate(plan.configurations)
        ),
    )


def _orbit_identity(
    realized: RealizedInteractionOrbit, primitive_orbit: PrimitiveInteractionOrbit
) -> OrbitIdentity:
    return OrbitIdentity(
        representative=tuple(int(atom) for atom in realized.representative),
        dimension=int(realized.dimension),
        observation_rows=tuple(
            int(row) for row in np.asarray(realized.observation_rows, dtype=np.int64).reshape(-1)
        ),
        exact_lattice_basis=tuple(
            tuple(int(value) for value in row)
            for row in np.asarray(primitive_orbit.exact_lattice_basis)
        ),
        images=tuple(_orbit_image(image) for image in realized.images),
    )


def _orbit_image(image: RealizedOrbitImage) -> tuple[tuple[int, ...], tuple[int, ...]]:
    action = image.action
    if not isinstance(action, TensorAction):
        raise TypeError(
            f"a realized orbit image must carry a tensor action, got {type(action).__name__}"
        )
    return (
        tuple(int(atom) for atom in image.cluster),
        tuple(int(axis) for axis in action.permutation),
    )


def _configuration_identity(
    configuration_id: int, configuration: DisplacementConfiguration
) -> ConfigurationIdentity:
    displacement = np.asarray(configuration.displacement, dtype=float)
    return ConfigurationIdentity(
        configuration_id=configuration_id,
        key=tuple((int(atom), int(direction)) for atom, direction in configuration.key),
        atoms=tuple(int(atom) for atom, _ in configuration.key),
        directions=tuple(int(direction) for _, direction in configuration.key),
        signs=tuple(int(sign) for sign in np.asarray(configuration.signs).reshape(-1)),
        step_angstrom=float(np.abs(displacement).max()),
    )


def _payload(manifest: DisplacementManifest) -> dict[str, object]:
    """Return the canonical document every field except the fingerprint contributes to."""
    return {
        "schema_version": int(manifest.schema_version),
        "primitive_fingerprint": str(manifest.primitive_fingerprint),
        "reference_fingerprint": str(manifest.reference_fingerprint),
        "order": int(manifest.order),
        "cutoff_angstrom": _hex_float(manifest.cutoff_angstrom),
        "max_body_order": None if manifest.max_body_order is None else int(manifest.max_body_order),
        "symprec": _hex_float(manifest.symprec),
        "displacement_angstrom": _hex_float(manifest.displacement_angstrom),
        "derivative_backend": str(manifest.derivative_backend),
        "stencil_signs": [[int(sign) for sign in row] for row in manifest.stencil_signs],
        "extrapolation": _extrapolation_payload(manifest.extrapolation),
        "orbits": [_orbit_payload(orbit) for orbit in manifest.orbits],
        "displacement_keys": [
            [[int(atom), int(direction)] for atom, direction in key]
            for key in manifest.displacement_keys
        ],
        "configurations": [
            {
                "configuration_id": int(configuration.configuration_id),
                "key": [[int(atom), int(direction)] for atom, direction in configuration.key],
                "atoms": [int(atom) for atom in configuration.atoms],
                "directions": [int(direction) for direction in configuration.directions],
                "signs": [int(sign) for sign in configuration.signs],
                "step_angstrom": _hex_float(configuration.step_angstrom),
            }
            for configuration in manifest.configurations
        ],
    }


def _orbit_payload(orbit: OrbitIdentity) -> dict[str, object]:
    return {
        "representative": [int(atom) for atom in orbit.representative],
        "dimension": int(orbit.dimension),
        "observation_rows": [int(row) for row in orbit.observation_rows],
        "exact_lattice_basis": [[int(value) for value in row] for row in orbit.exact_lattice_basis],
        "images": [
            [[int(atom) for atom in cluster], [int(axis) for axis in permutation]]
            for cluster, permutation in orbit.images
        ],
    }


def _extrapolation_payload(extrapolation: dict | None) -> dict[str, object] | None:
    if extrapolation is None:
        return None
    return {
        "degree": int(extrapolation["degree"]),
        "grid_angstrom": [_hex_float(step) for step in extrapolation["grid_angstrom"]],
        "side_steps": int(extrapolation["side_steps"]),
    }


def _canonical_json(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(payload: Mapping[str, object]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hex_float(value: float) -> str:
    """Encode one in-memory float exactly, with ``float.hex`` semantics."""
    return float(value).hex()


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, str):
        try:
            return float.fromhex(value)
        except ValueError as error:
            raise ValueError(
                f"{name} must be a decimal or hexadecimal float, got {value}"
            ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a float, got {type(value).__name__}")
    return float(value)


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value)


def _as_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _as_list(value: object, *, name: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array, got {type(value).__name__}")
    return value


def _int_rows(value: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_as_int(item, name=name) for item in _as_list(row, name=name))
        for row in _as_list(value, name=name)
    )


def _int_pairs(value: object, *, name: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for row in _int_rows(value, name=name):
        if len(row) != 2:
            raise ValueError(
                f"{name} entries must be (atom, direction) pairs, got {len(row)} values"
            )
        pairs.append((row[0], row[1]))
    return tuple(pairs)


def _normalized_extrapolation(extrapolation: object) -> dict | None:
    """Return the fixed in-memory description of an extrapolation grid, or ``None``."""
    if extrapolation is None:
        return None
    if not isinstance(extrapolation, Mapping):
        raise TypeError(
            f"extrapolation must be null or an object, got {type(extrapolation).__name__}"
        )
    expected = {"degree", "grid_angstrom", "side_steps"}
    keys = set(extrapolation)
    if keys != expected:
        raise ValueError(
            f"an extrapolation description must carry exactly {sorted(expected)}, "
            f"got {sorted(keys)}"
        )
    grid = extrapolation["grid_angstrom"]
    if isinstance(grid, (str, bytes)) or not isinstance(grid, (list, tuple)):
        raise TypeError(
            f"extrapolation grid_angstrom must be a sequence of steps, got {type(grid).__name__}"
        )
    description = {
        "degree": _as_int(extrapolation["degree"], name="extrapolation degree"),
        "grid_angstrom": [_as_float(step, name="extrapolation grid step") for step in grid],
        "side_steps": _as_int(extrapolation["side_steps"], name="extrapolation side_steps"),
    }
    if len(description["grid_angstrom"]) != 2 * description["side_steps"] + 1:
        raise ValueError(
            "extrapolation grid_angstrom must hold 2 * side_steps + 1 steps, got "
            f"{len(description['grid_angstrom'])} for {description['side_steps']} side steps"
        )
    return description


def _decode_fields(document: Mapping[str, object]) -> dict[str, object]:
    extrapolation = document["extrapolation"]
    if extrapolation is not None:
        extrapolation = _decode_mapping(extrapolation, name="extrapolation")
    return {
        "schema_version": _as_int(document["schema_version"], name="schema_version"),
        "primitive_fingerprint": _as_string(
            document["primitive_fingerprint"], name="primitive_fingerprint"
        ),
        "reference_fingerprint": _as_string(
            document["reference_fingerprint"], name="reference_fingerprint"
        ),
        "order": _as_int(document["order"], name="order"),
        "cutoff_angstrom": _as_float(document["cutoff_angstrom"], name="cutoff_angstrom"),
        "max_body_order": (
            None
            if document["max_body_order"] is None
            else _as_int(document["max_body_order"], name="max_body_order")
        ),
        "symprec": _as_float(document["symprec"], name="symprec"),
        "displacement_angstrom": _as_float(
            document["displacement_angstrom"], name="displacement_angstrom"
        ),
        "derivative_backend": _as_string(document["derivative_backend"], name="derivative_backend"),
        "stencil_signs": _int_rows(document["stencil_signs"], name="stencil_signs"),
        "extrapolation": extrapolation,
        "orbits": tuple(
            _decode_orbit(item) for item in _as_list(document["orbits"], name="orbits")
        ),
        "displacement_keys": tuple(
            _int_pairs(key, name="displacement_keys")
            for key in _as_list(document["displacement_keys"], name="displacement_keys")
        ),
        "configurations": tuple(
            _decode_configuration(item)
            for item in _as_list(document["configurations"], name="configurations")
        ),
    }


def _decode_orbit(value: object) -> OrbitIdentity:
    document = _decode_mapping(value, name="orbits")
    return OrbitIdentity(
        representative=tuple(
            _as_int(item, name="orbit representative")
            for item in _as_list(document["representative"], name="orbit representative")
        ),
        dimension=_as_int(document["dimension"], name="orbit dimension"),
        observation_rows=tuple(
            _as_int(item, name="orbit observation_rows")
            for item in _as_list(document["observation_rows"], name="orbit observation_rows")
        ),
        exact_lattice_basis=_int_rows(
            document["exact_lattice_basis"], name="orbit exact_lattice_basis"
        ),
        images=tuple(
            _decode_image(image) for image in _as_list(document["images"], name="orbit images")
        ),
    )


def _decode_image(value: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    document = _as_list(value, name="orbit images")
    if len(document) != 2:
        raise ValueError(f"an orbit image must be a (cluster, permutation) pair, got {document}")
    cluster = tuple(
        _as_int(item, name="image cluster") for item in _as_list(document[0], name="image cluster")
    )
    permutation = tuple(
        _as_int(item, name="image permutation")
        for item in _as_list(document[1], name="image permutation")
    )
    return (cluster, permutation)


def _decode_configuration(value: object) -> ConfigurationIdentity:
    document = _decode_mapping(value, name="configurations")
    return ConfigurationIdentity(
        configuration_id=_as_int(document["configuration_id"], name="configuration_id"),
        key=_int_pairs(document["key"], name="configuration key"),
        atoms=tuple(
            _as_int(item, name="configuration atoms")
            for item in _as_list(document["atoms"], name="configuration atoms")
        ),
        directions=tuple(
            _as_int(item, name="configuration directions")
            for item in _as_list(document["directions"], name="configuration directions")
        ),
        signs=tuple(
            _as_int(item, name="configuration signs")
            for item in _as_list(document["signs"], name="configuration signs")
        ),
        step_angstrom=_as_float(document["step_angstrom"], name="configuration step_angstrom"),
    )


def _decode_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def _stored_fingerprint(document: object) -> str:
    """Return the fingerprint stored in one decoded manifest document."""
    if not isinstance(document, dict):
        raise TypeError(
            f"a displacement manifest must be a JSON object, got {type(document).__name__}"
        )
    expected = {field.name for field in fields(DisplacementManifest)}
    missing = sorted(expected - set(document))
    unknown = sorted(set(document) - expected)
    if missing or unknown:
        raise ValueError(
            f"displacement manifest fields do not match the schema: "
            f"missing={missing}, unknown={unknown}"
        )
    return _as_string(document["fingerprint"], name="fingerprint")


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "ConfigurationIdentity",
    "DisplacementBatch",
    "DisplacementManifest",
    "ForceBatch",
    "OrbitIdentity",
    "build_manifest",
    "structure_fingerprint",
]
