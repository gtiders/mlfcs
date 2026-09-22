from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping
from math import ceil
from typing import Literal, NoReturn

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlfcs.finite_difference.extrapolation import ExtrapolationBackend
from mlfcs.finite_difference.plan_identity import (
    DisplacementBatch,
    DisplacementManifest,
    ForceBatch,
    build_manifest,
)
from mlfcs.finite_difference.reconstruction import reconstruct_sparse
from mlfcs.finite_difference.sampling import DisplacementPlan, build_displacement_plan
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.interactions.space import InteractionSpace

Progress = Callable[[int, int], None]
logger = logging.getLogger(__name__)


def _raise_unverified_forces(forces: object) -> NoReturn:
    """Reject a force input that cannot prove which plan produced the displacements.

    A bare array, a positional sequence and a mapping keyed by configuration id all lack
    the plan fingerprint, so accepting any of them would silently interpret forces from
    an older plan as belonging to the current one.
    """
    if isinstance(forces, Mapping):
        detail = "a mapping keyed by configuration id carries no plan fingerprint"
    elif isinstance(forces, np.ndarray):
        detail = "a bare force array carries no plan fingerprint"
    elif isinstance(forces, (list, tuple)):
        detail = "a positional force sequence carries no plan fingerprint"
    else:
        detail = f"{type(forces).__name__} carries no plan fingerprint"
    raise ValueError(
        f"reap() accepts only a ForceBatch as returned by evaluate(), got "
        f"{type(forces).__name__}: {detail}. Regenerate the displacements with sow() and "
        "re-evaluate them so the forces are bound to the plan fingerprint."
    )


class FiniteDifferenceCalculation:
    """Symmetry-reduced finite-difference calculation defined by ASE objects."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        order: int,
        reference: Atoms,
        cutoff: float = -5,
        max_body_order: int | None = None,
        displacement: float = 0.01,
        symprec: float = 1e-5,
    ):
        logger.info("Preparing order-%d finite-difference calculation", order)
        self.interaction_space = InteractionSpace(
            atoms,
            order=order,
            reference=reference,
            cutoff=cutoff,
            max_body_order=max_body_order,
            symprec=symprec,
            displacement=displacement,
        )
        self.primitive = self.interaction_space.primitive
        self.config = self.interaction_space.config
        self.supercell = self.interaction_space.supercell
        self.index = self.interaction_space.index
        self.cutoff = self.interaction_space.cutoff
        self.symmetry = self.interaction_space.symmetry
        self._plan: DisplacementPlan | None = None
        self._manifest: DisplacementManifest | None = None

    @property
    def realized_orbit_space(self):
        return self.interaction_space.realized_orbit_space

    @property
    def plan(self) -> DisplacementPlan:
        if self._plan is None:
            orbit_space = self.realized_orbit_space
            logger.info("Building the central-difference displacement plan")
            self._plan = build_displacement_plan(
                self.supercell,
                orbit_space,
                displacement=self.config.displacement,
            )
            displacement_keys = len(self._plan) // len(self._plan.stencil.signs)
            logger.info("%d displacement keys", displacement_keys)
            logger.info("%d force calculations required", len(self._plan))
        return self._plan

    @property
    def manifest(self) -> DisplacementManifest:
        """Canonical identity of the central-difference plan, built once.

        The manifest fingerprints everything the plan measures and requires, so a force
        batch can prove it belongs to exactly this plan.  The extrapolated workflow
        fingerprints its own top-level plan when :meth:`run` drives the backend.
        """
        if self._manifest is None:
            self._manifest = self._build_manifest(self.plan)
        return self._manifest

    def _build_manifest(
        self,
        plan: DisplacementPlan,
        *,
        derivative_backend: str = "central",
        extrapolation: Mapping[str, object] | None = None,
    ) -> DisplacementManifest:
        logger.info("Fingerprinting the %s displacement plan", derivative_backend)
        return build_manifest(
            plan,
            self.realized_orbit_space,
            primitive=self.primitive,
            reference=self.supercell,
            order=self.config.order,
            cutoff=self.cutoff,
            max_body_order=self.config.max_body_order,
            symprec=self.config.symprec,
            displacement=self.config.displacement,
            primitive_orbits=self.interaction_space.primitive_orbit_space.orbits,
            derivative_backend=derivative_backend,
            extrapolation=extrapolation,
        )

    def sow(self) -> DisplacementBatch:
        """Return the displaced structures of this plan together with its identity.

        Configuration ``i`` of the batch is the ``i``-th structure, and every structure
        carries its zero-based ``mlfcs_configuration_id`` and the plan fingerprint in
        ``info``.
        """
        manifest = self.manifest
        structures = []
        for atoms in self.plan:
            atoms.info["mlfcs_plan_fingerprint"] = manifest.fingerprint
            structures.append(atoms)
        logger.info("Sowing %d displaced structures in reference atom order", len(structures))
        return DisplacementBatch(manifest, tuple(structures))

    def reap(
        self,
        forces: ForceBatch,
        *,
        acoustic_sum_rule: bool = True,
        asr_tolerance: float = 1e-10,
    ) -> ForceConstants:
        """Reconstruct force constants from forces verified against this plan.

        ``forces`` must be a :class:`ForceBatch`: either the result of :meth:`evaluate`
        or a batch built from structures returned by :meth:`sow`.  Its fingerprint,
        schema version, configuration ids and shape must match this plan, and its rows
        may arrive in any configuration-id order.
        """
        logger.info("Reaping forces for order-%d force constants", self.config.order)
        manifest = self.manifest
        values = self._validated_forces(forces, manifest)
        logger.info("Validated %d force configurations", len(values))
        derivatives = self.plan.contract_forces(values)
        logger.info("Contracted %d finite-difference derivatives", len(derivatives))
        return self._reconstruct(
            derivatives,
            acoustic_sum_rule=acoustic_sum_rule,
            asr_tolerance=asr_tolerance,
            metadata={
                "derivative_backend": "central",
                "configurations": len(self.plan),
                "plan_schema_version": manifest.schema_version,
                "plan_fingerprint": manifest.fingerprint,
            },
        )

    def _validated_forces(self, forces: ForceBatch, manifest: DisplacementManifest) -> np.ndarray:
        """Return one batch's forces in plan order, or reject the batch.

        Everything is checked before any differentiation: the batch type, the schema
        version, the plan fingerprint, the configuration ids, the array shape and the
        finiteness of the values.
        """
        if not isinstance(forces, ForceBatch):
            _raise_unverified_forces(forces)
        if forces.schema_version != manifest.schema_version:
            raise ValueError(
                f"force batch schema version {forces.schema_version} does not match the plan "
                f"schema version {manifest.schema_version} (batch fingerprint "
                f"{forces.fingerprint}, plan fingerprint {manifest.fingerprint}); regenerate "
                "the displacements with sow()"
            )
        if forces.fingerprint != manifest.fingerprint:
            raise ValueError(
                f"force batch fingerprint {forces.fingerprint} does not match the plan "
                f"fingerprint {manifest.fingerprint}: the forces were collected from a "
                "different displacement plan. Regenerate the displacements with sow() and "
                "re-evaluate them."
            )
        identifiers = forces.configuration_ids
        counts = Counter(identifiers)
        expected_ids = set(range(len(manifest.configurations)))
        duplicates = sorted(value for value, count in counts.items() if count > 1)
        unknown = sorted(value for value in counts if value not in expected_ids)
        missing = sorted(expected_ids - set(identifiers))
        if duplicates or unknown or missing:
            raise ValueError(
                f"force batch configuration ids do not match the plan: duplicates={duplicates}, "
                f"unknown={unknown}, missing={missing}"
            )
        expected_shape = (len(manifest.configurations), len(self.supercell), 3)
        if forces.forces.shape != expected_shape:
            raise ValueError(
                f"force batch shape {forces.forces.shape} does not match the plan: expected "
                f"{expected_shape}, with {len(self.supercell)} supercell atoms and three "
                "Cartesian components per configuration"
            )
        if not np.isfinite(forces.forces).all():
            raise ValueError("force batch contains NaN or infinite forces")
        permutation = np.argsort(np.asarray(identifiers, dtype=np.int64), kind="stable")
        return np.ascontiguousarray(forces.forces[permutation], dtype=float)

    def _reconstruct(
        self,
        derivatives,
        *,
        acoustic_sum_rule: bool,
        asr_tolerance: float,
        metadata: dict[str, object],
    ) -> ForceConstants:
        logger.info(
            "Reconstructing symmetry-expanded force constants "
            f"(ASR {'enabled' if acoustic_sum_rule else 'disabled'})"
        )
        sparse, diagnostics = reconstruct_sparse(
            self.realized_orbit_space,
            self.index,
            derivatives,
            enforce_asr=acoustic_sum_rule,
            asr_tolerance=asr_tolerance,
            report=logger.info,
            primitive_interaction_space=self.interaction_space.primitive_orbit_space,
            return_diagnostics=True,
        )
        logger.info("Reconstructed %d sparse cluster tensors", len(sparse.tensors))
        return ForceConstants(
            {},
            self.supercell.copy(),
            metadata={
                "order": self.config.order,
                "cutoff_angstrom": self.cutoff,
                "displacement_angstrom": self.config.displacement,
                "spacegroup": self.symmetry.symbol,
                "acoustic_sum_rule": acoustic_sum_rule,
                "asr_projection_tolerance": asr_tolerance if acoustic_sum_rule else None,
                "maximum_asr_residual_before": diagnostics.initial_residual,
                "maximum_asr_residual_after": diagnostics.final_residual,
                "asr_parameter_correction": diagnostics.correction_norm,
                "asr_relative_parameter_correction": diagnostics.relative_correction,
                "asr_projection_iterations": diagnostics.iterations,
                **metadata,
            },
            sparse={self.config.order: sparse},
            relation=self.interaction_space.relation,
        )

    def run(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        acoustic_sum_rule: bool = True,
        asr_tolerance: float = 1e-10,
        derivative_backend: Literal["central", "extrapolate"] = "central",
        extrapolation_spacing: float | None = None,
        extrapolation_side_steps: int = 1,
        extrapolation_degree: int = 1,
    ) -> ForceConstants:
        """Evaluate force constants serially with a user-owned ASE Calculator."""
        if derivative_backend == "extrapolate":
            if extrapolation_spacing is None:
                raise ValueError(
                    "extrapolation_spacing is required for derivative_backend='extrapolate'"
                )
            return self._run_extrapolation(
                calculator,
                spacing=extrapolation_spacing,
                side_steps=extrapolation_side_steps,
                degree=extrapolation_degree,
                progress=progress,
                acoustic_sum_rule=acoustic_sum_rule,
                asr_tolerance=asr_tolerance,
            )
        if derivative_backend != "central":
            raise ValueError("derivative_backend must be 'central' or 'extrapolate'")
        if (
            extrapolation_spacing is not None
            or extrapolation_side_steps != 1
            or extrapolation_degree != 1
        ):
            raise ValueError("extrapolation options require derivative_backend='extrapolate'")
        forces = self.evaluate(calculator, progress=progress)
        return self.reap(
            forces,
            acoustic_sum_rule=acoustic_sum_rule,
            asr_tolerance=asr_tolerance,
        )

    def _run_extrapolation(
        self,
        calculator: Calculator,
        *,
        spacing: float,
        side_steps: int,
        degree: int,
        progress: Progress | None,
        acoustic_sum_rule: bool,
        asr_tolerance: float,
    ) -> ForceConstants:
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        backend = ExtrapolationBackend(
            self.config.displacement,
            spacing,
            side_steps,
            degree,
        )
        plans = backend.plans(self.supercell, self.realized_orbit_space)
        total = sum(len(plan) for plan in plans)
        grid_text = ", ".join(f"{step:.10f}" for step in backend.grid)
        logger.info("Derivative backend: zero-step extrapolation")
        logger.info("Displacement grid: %s Å", grid_text)
        logger.info("Polynomial degree in h^2: %d", degree)
        logger.info("%d central-difference subplans", len(plans))
        logger.info("%d force calculations required", total)

        # The top-level plan of this backend is the concatenation of every subplan; its
        # stencil signs are the central ones, because every subplan varies only the step.
        grid_plan = DisplacementPlan(
            self.supercell.copy(),
            tuple(configuration for plan in plans for configuration in plan.configurations),
            self.plan.stencil,
        )
        manifest = self._build_manifest(
            grid_plan,
            derivative_backend="extrapolate",
            extrapolation={
                "grid_angstrom": backend.grid.tolist(),
                "side_steps": side_steps,
                "degree": degree,
            },
        )
        forces = np.empty((total, len(self.supercell), 3), dtype=float)
        completed = 0
        for step, plan in zip(backend.grid, plans, strict=True):
            logger.info("Evaluating displacement step %.10f Å", step)
            forces[completed : completed + len(plan)] = self._evaluate_plan(
                plan,
                calculator,
                progress=progress,
                completed_offset=completed,
                total=total,
            )
            completed += len(plan)
        values = self._validated_forces(
            ForceBatch(
                fingerprint=manifest.fingerprint,
                configuration_ids=tuple(range(total)),
                forces=forces,
                schema_version=manifest.schema_version,
            ),
            manifest,
        )
        derivative_sets = []
        cursor = 0
        for plan in plans:
            derivative_sets.append(plan.contract_forces(values[cursor : cursor + len(plan)]))
            cursor += len(plan)
        derivatives, metrics = backend.extrapolate(derivative_sets)
        unit = f"eV/angstrom^{self.config.order}"
        logger.info("Zero-step derivative extrapolation")
        logger.info(
            f"- Maximum correction from central displacement: "
            f"{metrics.maximum_correction:.10e} {unit}"
        )
        logger.info("Relative L2 correction: %.10e", metrics.relative_l2_correction)
        logger.info(
            f"- Maximum polynomial fit residual: {metrics.maximum_fit_residual:.10e} {unit}"
        )
        return self._reconstruct(
            derivatives,
            acoustic_sum_rule=acoustic_sum_rule,
            asr_tolerance=asr_tolerance,
            metadata={
                "derivative_backend": "extrapolate",
                "configurations": total,
                "extrapolation_grid_angstrom": backend.grid.tolist(),
                "extrapolation_degree": degree,
                "plan_schema_version": manifest.schema_version,
                "plan_fingerprint": manifest.fingerprint,
                "extrapolation_maximum_correction": metrics.maximum_correction,
                "extrapolation_relative_l2_correction": metrics.relative_l2_correction,
                "extrapolation_maximum_fit_residual": metrics.maximum_fit_residual,
            },
        )

    def evaluate(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
    ) -> ForceBatch:
        """Evaluate every configuration and bind the forces to this plan."""
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        manifest = self.manifest
        logger.info(
            "Evaluating %d configurations with %s", len(self.plan), type(calculator).__name__
        )
        values = self._evaluate_plan(
            self.plan,
            calculator,
            progress=progress,
            completed_offset=0,
            total=len(self.plan),
        )
        return ForceBatch(
            fingerprint=manifest.fingerprint,
            configuration_ids=tuple(range(len(self.plan))),
            forces=values,
            schema_version=manifest.schema_version,
        )

    def _evaluate_plan(
        self,
        plan: DisplacementPlan,
        calculator: Calculator,
        *,
        progress: Progress | None,
        completed_offset: int,
        total: int,
    ) -> np.ndarray:
        forces = np.empty((len(plan), len(self.supercell), 3), dtype=float)
        reporting_interval = max(1, ceil(total / 10))
        for configuration_id, atoms in enumerate(plan):
            atoms.calc = calculator
            values = np.asarray(atoms.get_forces(), dtype=float)
            expected = (len(self.supercell), 3)
            if values.shape != expected:
                raise ValueError(
                    f"calculator forces for configuration {configuration_id} must have "
                    f"shape {expected}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"calculator forces for configuration {configuration_id} "
                    "contain NaN or infinite values"
                )
            forces[configuration_id] = values
            completed = completed_offset + configuration_id + 1
            if progress is not None:
                progress(completed, total)
            elif completed == 1 or completed == total or completed % reporting_interval == 0:
                percentage = 100.0 * completed / total
                logger.info("Forces: %d/%d (%.0f%%)", completed, total, percentage)
        return forces
