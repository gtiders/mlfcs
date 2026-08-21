from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlfcs.constraints.solver import reconstruct_sparse
from mlfcs.core.interactions import InteractionSpace
from mlfcs.finite_difference.extrapolation import ExtrapolationBackend
from mlfcs.finite_difference.sampling import DisplacementPlan, build_displacement_plan
from mlfcs.ifc.model import ForceConstants

Progress = Callable[[int, int], None]
ForceInput = np.ndarray | Sequence[np.ndarray] | Mapping[int, np.ndarray]


class ForceConstantCalculation:
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
        verbose: bool = True,
    ):
        self.verbose = bool(verbose)
        self._report(f"Preparing order-{order} force-constant calculation")
        self.interaction_space = InteractionSpace(
            atoms,
            order=order,
            reference=reference,
            cutoff=cutoff,
            max_body_order=max_body_order,
            symprec=symprec,
            displacement=displacement,
            reporter=self._report if self.verbose else None,
        )
        self.primitive = self.interaction_space.primitive
        self.config = self.interaction_space.config
        self.supercell = self.interaction_space.supercell
        self.index = self.interaction_space.index
        self.cutoff = self.interaction_space.cutoff
        self.symmetry = self.interaction_space.symmetry
        self._plan: DisplacementPlan | None = None

    @property
    def orbit_space(self):
        return self.interaction_space.orbit_space

    @property
    def plan(self) -> DisplacementPlan:
        if self._plan is None:
            orbit_space = self.orbit_space
            self._report("Building the central-difference displacement plan")
            self._plan = build_displacement_plan(
                self.supercell,
                orbit_space,
                displacement=self.config.displacement,
            )
            displacement_keys = len(self._plan) // len(self._plan.stencil.signs)
            self._report(f"- {displacement_keys} displacement keys")
            self._report(f"- {len(self._plan)} force calculations required")
        return self._plan

    def sow(self) -> list[Atoms]:
        """Return displaced structures in the exact positional reap order.

        Configuration ``i`` must be returned to positional :meth:`reap` at
        index ``i``. Each structure also carries its zero-based stable ID.
        """
        structures = list(self.plan)
        self._report(f"Sowing {len(structures)} displaced structures in reference atom order")
        return structures

    def reap(
        self,
        forces: ForceInput,
        *,
        acoustic_sum_rule: bool = True,
    ) -> ForceConstants:
        """Reconstruct force constants from forces supplied by the user.

        A sequence is positional and must follow :meth:`sow` exactly. A
        mapping is keyed by ``mlfcs_configuration_id`` and may arrive in any
        insertion order.
        """
        self._report(f"Reaping forces for order-{self.config.order} force constants")
        values = self._normalize_forces(forces)
        self._report(f"- Validated {len(values)} force configurations")
        derivatives = self.plan.contract_forces(values)
        self._report(f"- Contracted {len(derivatives)} finite-difference derivatives")
        return self._reconstruct(
            derivatives,
            acoustic_sum_rule=acoustic_sum_rule,
            metadata={
                "derivative_backend": "central",
                "configurations": len(self.plan),
            },
        )

    def _reconstruct(
        self,
        derivatives,
        *,
        acoustic_sum_rule: bool,
        metadata: dict[str, object],
    ) -> ForceConstants:
        self._report(
            "Reconstructing symmetry-expanded force constants "
            f"(ASR {'enabled' if acoustic_sum_rule else 'disabled'})"
        )
        sparse = reconstruct_sparse(
            self.orbit_space,
            self.index,
            derivatives,
            enforce_asr=acoustic_sum_rule,
            report=self._report,
            primitive_interaction_space=self.interaction_space.primitive_orbit_space,
        )
        self._report(f"- Reconstructed {len(sparse.tensors)} sparse cluster tensors")
        return ForceConstants(
            {},
            self.supercell.copy(),
            metadata={
                "order": self.config.order,
                "cutoff_angstrom": self.cutoff,
                "displacement_angstrom": self.config.displacement,
                "spacegroup": self.symmetry.symbol,
                "acoustic_sum_rule": acoustic_sum_rule,
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
    ) -> ForceConstants:
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        backend = ExtrapolationBackend(
            self.config.displacement,
            spacing,
            side_steps,
            degree,
        )
        plans = backend.plans(self.supercell, self.orbit_space)
        total = sum(len(plan) for plan in plans)
        grid_text = ", ".join(f"{step:.10f}" for step in backend.grid)
        self._report("Derivative backend: zero-step extrapolation")
        self._report(f"- Displacement grid: {grid_text} Å")
        self._report(f"- Polynomial degree in h^2: {degree}")
        self._report(f"- {len(plans)} central-difference subplans")
        self._report(f"- {total} force calculations required")

        derivative_sets = []
        completed = 0
        for step, plan in zip(backend.grid, plans, strict=True):
            self._report(f"Evaluating displacement step {step:.10f} Å")
            forces = self._evaluate_plan(
                plan,
                calculator,
                progress=progress,
                completed_offset=completed,
                total=total,
            )
            derivative_sets.append(plan.contract_forces(forces))
            completed += len(plan)
        derivatives, metrics = backend.extrapolate(derivative_sets)
        unit = f"eV/angstrom^{self.config.order}"
        self._report("Zero-step derivative extrapolation")
        self._report(
            f"- Maximum correction from central displacement: "
            f"{metrics.maximum_correction:.10e} {unit}"
        )
        self._report(f"- Relative L2 correction: {metrics.relative_l2_correction:.10e}")
        self._report(
            f"- Maximum polynomial fit residual: {metrics.maximum_fit_residual:.10e} {unit}"
        )
        return self._reconstruct(
            derivatives,
            acoustic_sum_rule=acoustic_sum_rule,
            metadata={
                "derivative_backend": "extrapolate",
                "configurations": total,
                "extrapolation_grid_angstrom": backend.grid.tolist(),
                "extrapolation_degree": degree,
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
    ) -> np.ndarray:
        """Evaluate and return forces in the exact positional reap order."""
        if not isinstance(calculator, Calculator):
            raise TypeError("calculator must be an ASE Calculator")
        self._report(f"Evaluating {len(self.plan)} configurations with {type(calculator).__name__}")
        return self._evaluate_plan(
            self.plan,
            calculator,
            progress=progress,
            completed_offset=0,
            total=len(self.plan),
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
            elif self.verbose and (
                completed == 1 or completed == total or completed % reporting_interval == 0
            ):
                percentage = 100.0 * completed / total
                self._report(f"- Forces: {completed}/{total} ({percentage:.0f}%)")
        return forces

    def _report(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _normalize_forces(self, forces: ForceInput) -> np.ndarray:
        if isinstance(forces, Mapping):
            expected_ids = set(range(len(self.plan)))
            received_ids = set(forces)
            if received_ids != expected_ids:
                missing = sorted(expected_ids - received_ids)
                extra = sorted(received_ids - expected_ids)
                raise ValueError(
                    f"force IDs do not match sow order: missing={missing}, extra={extra}"
                )
            values = np.asarray([forces[index] for index in range(len(self.plan))], dtype=float)
        else:
            values = np.asarray(forces, dtype=float)
        expected_shape = (len(self.plan), len(self.supercell), 3)
        if values.shape != expected_shape:
            raise ValueError(f"forces must have shape {expected_shape}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("forces contain NaN or infinite values")
        return values


# Short alias for interactive use.
Calculation = ForceConstantCalculation
