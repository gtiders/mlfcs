"""Forces may only be reaped by the plan that asked for them.

The observable contract of ``sow``/``evaluate``/``reap`` is that a
:class:`ForceBatch` proves its provenance: the plan fingerprint, the schema version and
the configuration ids are verified before any differentiation, so a batch of forces from
an older or differently measured plan is refused instead of silently reinterpreted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
from mlfcs.finite_difference.extrapolation import ExtrapolationBackend
from mlfcs.finite_difference.plan_identity import (
    PLAN_SCHEMA_VERSION,
    DisplacementBatch,
    DisplacementManifest,
    ForceBatch,
    build_manifest,
)
from mlfcs.finite_difference.sampling import DisplacementPlan
from mlfcs.finite_difference.stencil import CentralDifferenceStencil

#: A fresh interpreter that rebuilds the same plan and reports its identity.
_REBUILD_SCRIPT = """
import json
import logging

from ase.build import bulk

from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation

logging.disable(logging.INFO)
primitive = bulk("Si", "diamond", a=5.43)
job = FiniteDifferenceCalculation(
    primitive, order=2, reference=primitive.repeat((2, 1, 1)), cutoff=3.0
)
print(json.dumps({"fingerprint": job.manifest.fingerprint, "manifest": job.manifest.to_json()}))
"""


def _calculation(*, order: int = 2, cutoff: float = 3.0, displacement: float = 0.01):
    primitive = bulk("Si", "diamond", a=5.43)
    return FiniteDifferenceCalculation(
        primitive,
        order=order,
        reference=primitive.repeat((2, 1, 1)),
        cutoff=cutoff,
        displacement=displacement,
    )


def _batch(
    job: FiniteDifferenceCalculation,
    forces: np.ndarray,
    *,
    ids: tuple[int, ...] | None = None,
    fingerprint: str | None = None,
    schema_version: int = PLAN_SCHEMA_VERSION,
) -> ForceBatch:
    return ForceBatch(
        fingerprint=job.manifest.fingerprint if fingerprint is None else fingerprint,
        configuration_ids=tuple(range(len(forces))) if ids is None else ids,
        forces=forces,
        schema_version=schema_version,
    )


def _aluminium_calculation():
    """Four displacements of fcc aluminium, cheap enough for a real calculator."""
    primitive = bulk("Al", "fcc", a=4.05)
    return FiniteDifferenceCalculation(
        primitive, order=2, reference=primitive.repeat((2, 2, 1)), cutoff=-1
    )


def _external_forces(job: FiniteDifferenceCalculation) -> np.ndarray:
    """Return forces structure by structure, the way an external code would.

    One calculator instance walks the whole sweep, exactly as a queue submission does.
    """
    calculator = EMT()
    forces = []
    for atoms in job.sow():
        atoms.calc = calculator
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
    return np.asarray(forces)


def _zero_forces(job: FiniteDifferenceCalculation) -> np.ndarray:
    """Zero forces; the batches that use them are rejected before any differentiation."""
    return np.zeros((len(job.plan), len(job.supercell), 3))


def _oracle_manifest(
    job: FiniteDifferenceCalculation, plan=None, **options
) -> DisplacementManifest:
    """Build a manifest through the public entry point with this calculation's inputs."""
    return build_manifest(
        job.plan if plan is None else plan,
        job.realized_orbit_space,
        primitive=job.primitive,
        reference=job.supercell,
        order=job.config.order,
        cutoff=job.cutoff,
        max_body_order=job.config.max_body_order,
        symprec=job.config.symprec,
        displacement=job.config.displacement,
        primitive_orbits=job.interaction_space.primitive_orbit_space.orbits,
        **options,
    )


def test_sow_forces_reap_round_trip_matches_the_calculator_path():
    job = _aluminium_calculation()
    external = _external_forces(job)
    evaluated = job.evaluate(EMT())

    # Both halves see the same displacements, and the evaluated batch is bound to the plan.
    np.testing.assert_allclose(external, evaluated.forces, rtol=1e-12, atol=1e-14)
    assert evaluated.fingerprint == job.manifest.fingerprint
    assert evaluated.configuration_ids == tuple(range(len(job.plan)))
    assert np.abs(external).max() > 0

    reaped = job.reap(_batch(job, external))
    direct = job.run(EMT())

    assert reaped.metadata["plan_fingerprint"] == job.manifest.fingerprint
    assert reaped.metadata["plan_schema_version"] == PLAN_SCHEMA_VERSION
    assert np.abs(reaped.materialize(2)).max() > 0
    # The two force sets agree to the last bits of a real calculator, and the force
    # constants agree to the same level; a mis-ordered sweep would differ by order one.
    np.testing.assert_allclose(reaped.materialize(2), direct.materialize(2), rtol=1e-10, atol=1e-12)


def test_shuffled_force_batch_reconstructs_identically():
    job = _aluminium_calculation()
    forces = _external_forces(job)
    identifiers = (2, 0, 3, 1)
    assert len(forces) == len(identifiers)

    straight = job.reap(_batch(job, forces)).materialize(2)
    shuffled = job.reap(_batch(job, forces[list(identifiers)], ids=identifiers)).materialize(2)

    assert not np.array_equal(forces, forces[list(identifiers)])
    assert np.abs(straight).max() > 0
    np.testing.assert_array_equal(straight, shuffled)


@pytest.mark.parametrize("shape", ["ndarray", "list", "mapping"])
def test_forces_without_plan_identity_are_rejected(shape):
    job = _calculation()
    forces = _zero_forces(job)
    inputs = {
        "ndarray": forces,
        "list": list(forces),
        "mapping": {index: row for index, row in enumerate(forces)},
    }

    with pytest.raises(ValueError) as error:
        job.reap(inputs[shape])

    message = str(error.value)
    assert "ForceBatch" in message
    assert "Regenerate the displacements" in message
    assert "sow()" in message
    if shape == "mapping":
        assert "mapping keyed by configuration id" in message


def test_a_batch_from_another_plan_is_rejected():
    job = _calculation()
    other = _calculation(displacement=0.02)
    forces = _zero_forces(job)
    assert other.manifest.fingerprint != job.manifest.fingerprint

    with pytest.raises(ValueError, match="fingerprint") as error:
        job.reap(_batch(job, forces, fingerprint=other.manifest.fingerprint))

    message = str(error.value)
    assert other.manifest.fingerprint in message
    assert job.manifest.fingerprint in message


def test_a_batch_from_another_schema_version_is_rejected():
    job = _calculation()
    stale = PLAN_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema version") as error:
        job.reap(_batch(job, _zero_forces(job), schema_version=stale))

    message = str(error.value)
    assert f"schema version {stale}" in message
    assert f"schema version {PLAN_SCHEMA_VERSION}" in message


def test_missing_extra_and_duplicated_configuration_ids_are_rejected():
    job = _calculation()
    forces = _zero_forces(job)

    with pytest.raises(ValueError, match="configuration ids") as error:
        job.reap(_batch(job, forces[:1], ids=(0,)))
    assert "missing=[1]" in str(error.value)

    with pytest.raises(ValueError, match="configuration ids") as error:
        job.reap(_batch(job, np.concatenate([forces, forces[:1]]), ids=(0, 1, 2)))
    assert "unknown=[2]" in str(error.value)

    with pytest.raises(ValueError, match="configuration ids") as error:
        job.reap(_batch(job, forces, ids=(0, 0)))
    assert "duplicates=[0]" in str(error.value)


def test_a_wrong_atom_count_is_rejected():
    job = _calculation()
    forces = _zero_forces(job)

    with pytest.raises(ValueError, match="shape") as error:
        job.reap(_batch(job, forces[:, :-1]))

    assert f"({len(forces)}, {len(job.supercell) - 1}, 3)" in str(error.value)
    assert f"({len(forces)}, {len(job.supercell)}, 3)" in str(error.value)


def test_a_force_batch_rejects_malformed_ids_and_forces():
    job = _calculation()
    forces = _zero_forces(job)

    with pytest.raises(TypeError, match="configuration ids must be integers"):
        ForceBatch(job.manifest.fingerprint, (0.5, 1), forces)
    with pytest.raises(ValueError, match="n_configurations, n_atoms, 3"):
        ForceBatch(job.manifest.fingerprint, (0, 1), forces[:, :, 0])
    with pytest.raises(ValueError, match="configuration ids"):
        ForceBatch(job.manifest.fingerprint, (0,), forces)


def test_reselected_observation_rows_change_the_identity_without_changing_the_plan(monkeypatch):
    reference = _calculation()
    reselected = _calculation()
    space = reselected.realized_orbit_space
    orbit = space.orbits[1]
    rows = np.asarray(orbit.observation_rows)
    assert len(rows) == 2

    # The plan is built only after this replacement, exactly as the review demands: the
    # same crystal with the same configuration count but a different observation row set.
    reselected_space = replace(
        space, orbits=(space.orbits[0], replace(orbit, observation_rows=rows[::-1].copy()))
    )
    monkeypatch.setattr(reselected.interaction_space, "_realized_orbit_space", reselected_space)

    assert [
        (configuration.key, tuple(configuration.signs))
        for configuration in reselected.plan.configurations
    ] == [
        (configuration.key, tuple(configuration.signs))
        for configuration in reference.plan.configurations
    ]
    assert len(reselected.plan) == len(reference.plan)
    assert reselected.manifest.fingerprint != reference.manifest.fingerprint

    forces = _zero_forces(reference)
    with pytest.raises(ValueError, match="fingerprint") as error:
        reselected.reap(_batch(reference, forces))

    message = str(error.value)
    assert reference.manifest.fingerprint in message
    assert reselected.manifest.fingerprint in message


@pytest.mark.parametrize(
    "variant",
    ["primitive", "reference", "cutoff", "displacement", "stencil"],
)
def test_plan_settings_change_the_fingerprint(variant):
    job = _calculation()
    baseline = job.manifest.fingerprint
    assert _oracle_manifest(job) == job.manifest

    if variant == "primitive":
        changed = bulk("Si", "diamond", a=5.44)
        other = FiniteDifferenceCalculation(
            changed, order=2, reference=changed.repeat((2, 1, 1)), cutoff=3.0
        ).manifest
    elif variant == "reference":
        primitive = bulk("Si", "diamond", a=5.43)
        other = FiniteDifferenceCalculation(
            primitive, order=2, reference=primitive.repeat((1, 2, 1)), cutoff=3.0
        ).manifest
    elif variant == "cutoff":
        other = _calculation(cutoff=3.5).manifest
    elif variant == "displacement":
        other = _calculation(displacement=0.02).manifest
    else:
        coarser = replace(
            job.plan,
            stencil=CentralDifferenceStencil(
                job.plan.stencil.derivative_order + 1, job.config.displacement
            ),
        )
        other = _oracle_manifest(job, coarser)

    assert other.fingerprint != baseline


def test_every_documented_field_changes_the_fingerprint():
    manifest = _calculation().manifest
    changes = {
        "schema_version": PLAN_SCHEMA_VERSION + 1,
        "primitive_fingerprint": "0" * 64,
        "reference_fingerprint": "1" * 64,
        "order": manifest.order + 1,
        "cutoff_angstrom": manifest.cutoff_angstrom + 0.5,
        "max_body_order": 3,
        "symprec": manifest.symprec * 2.0,
        "displacement_angstrom": manifest.displacement_angstrom * 2.0,
        "derivative_backend": "extrapolate",
        "stencil_signs": ((0, 1), (0, -1)),
        "extrapolation": {"degree": 1, "grid_angstrom": [0.005, 0.01, 0.015], "side_steps": 1},
        "orbits": tuple(reversed(manifest.orbits)),
        "displacement_keys": manifest.displacement_keys + (((7, 2),),),
        "configurations": tuple(reversed(manifest.configurations)),
    }
    # The identity document must cover every documented field; a new field that nobody
    # fingerprints would silently leave two different plans sharing one identity.
    assert set(changes) == {field.name for field in fields(manifest)} - {"fingerprint"}

    for name, value in changes.items():
        assert replace(manifest, **{name: value}).fingerprint != manifest.fingerprint, name


def test_manifest_json_round_trip_is_exact(tmp_path):
    job = _calculation(order=3)
    manifest = job.manifest

    restored = DisplacementManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.fingerprint == manifest.fingerprint
    assert restored.to_json() == manifest.to_json()
    assert json.loads(manifest.to_json())["fingerprint"] == manifest.fingerprint

    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert DisplacementManifest.load(path) == manifest


@pytest.mark.parametrize("edit", ["fingerprint", "cutoff_angstrom", "orbits"])
def test_an_edited_manifest_is_rejected(edit):
    manifest = _calculation().manifest
    document = json.loads(manifest.to_json())
    if edit == "fingerprint":
        document["fingerprint"] = "0" * 64
    elif edit == "cutoff_angstrom":
        document["cutoff_angstrom"] = "0x1.0000000000000p+2"
    else:
        document["orbits"][0]["dimension"] += 1

    with pytest.raises(ValueError) as error:
        DisplacementManifest.from_json(json.dumps(document))

    assert "fingerprint" in str(error.value)


def test_displacement_batch_verifies_plan_provenance():
    job = _calculation()
    manifest = job.manifest
    structures = tuple(job.plan)

    with pytest.raises(ValueError, match="plan fingerprint"):
        DisplacementBatch(manifest, structures)
    with pytest.raises(ValueError, match="configurations"):
        DisplacementBatch(manifest, tuple(job.sow())[:1])
    with pytest.raises(ValueError, match="mlfcs_configuration_id"):
        DisplacementBatch(manifest, tuple(reversed(list(job.sow()))))
    assert len(job.sow()) == len(manifest.configurations)


def test_fingerprint_is_stable_across_constructions_and_processes():
    first = _calculation().manifest
    second = _calculation().manifest

    assert first.fingerprint == second.fingerprint
    assert first.to_json() == second.to_json()

    completed = subprocess.run(
        [sys.executable, "-c", _REBUILD_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    rebuilt = json.loads(completed.stdout.strip().splitlines()[-1])

    assert rebuilt["fingerprint"] == first.fingerprint
    assert rebuilt["manifest"] == first.to_json()


def test_the_extrapolated_backend_fingerprints_its_top_level_plan():
    job = _aluminium_calculation()
    backend = ExtrapolationBackend(job.config.displacement, 0.005, side_steps=1, degree=1)

    result = job.run(
        EMT(),
        derivative_backend="extrapolate",
        extrapolation_spacing=0.005,
        acoustic_sum_rule=False,
    )

    plans = backend.plans(job.supercell, job.realized_orbit_space)
    grid_plan = DisplacementPlan(
        job.supercell.copy(),
        tuple(configuration for plan in plans for configuration in plan.configurations),
        job.plan.stencil,
    )
    manifest = _oracle_manifest(
        job,
        grid_plan,
        derivative_backend="extrapolate",
        extrapolation={
            "grid_angstrom": backend.grid.tolist(),
            "side_steps": 1,
            "degree": 1,
        },
    )

    assert result.metadata["configurations"] == sum(len(plan) for plan in plans)
    assert len(manifest.configurations) == result.metadata["configurations"]
    assert result.metadata["plan_fingerprint"] == manifest.fingerprint
    assert result.metadata["plan_schema_version"] == PLAN_SCHEMA_VERSION
    assert result.metadata["plan_fingerprint"] != job.manifest.fingerprint
    assert np.isfinite(result.materialize(2)).all()
