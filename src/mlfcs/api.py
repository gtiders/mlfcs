from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlfcs.core.geometry import make_supercell, resolve_cutoff
from mlfcs.core.orbits import OrbitSpace, build_orbit_space
from mlfcs.core.symmetry import SymmetryOperations
from mlfcs.finite_difference.sampling import DisplacementPlan, build_displacement_plan
from mlfcs.model import ForceConstants, RunConfig
from mlfcs.reconstruction.solver import reconstruct_sparse
from mlfcs.runtime import JaxPlatform, configure_jax

Progress = Callable[[int, int], None]
ForceInput = np.ndarray | Sequence[np.ndarray] | Mapping[int, np.ndarray]


class ForceConstantCalculation:
    """Symmetry-reduced finite-difference calculation defined by ASE objects."""

    def __init__(
        self,
        atoms: Atoms,
        *,
        order: int,
        supercell: tuple[int, int, int] = (2, 2, 2),
        cutoff: float = -5,
        displacement: float = 0.01,
        symprec: float = 1e-5,
        jax_platform: JaxPlatform = "auto",
        report_cutoff: bool = True,
        verbose: bool = True,
    ):
        configure_jax(jax_platform)
        config = RunConfig(
            order=order,
            supercell=supercell,
            cutoff=cutoff,
            displacement=displacement,
            symprec=symprec,
        )
        self.primitive = atoms.copy()
        self.config = config
        self.jax_platform = jax_platform
        self.verbose = bool(verbose)
        self._report(f"Preparing order-{config.order} force-constant calculation")
        self._report(
            f"Creating {config.supercell[0]}x{config.supercell[1]}x{config.supercell[2]} supercell"
        )
        self.supercell, self.index = make_supercell(self.primitive, config.supercell)
        self._report(
            f"- {len(self.primitive)} primitive atoms, {len(self.supercell)} supercell atoms"
        )
        self._report("Resolving the interaction cutoff")
        self.cutoff = resolve_cutoff(
            self.supercell,
            self.index,
            config.cutoff,
            report=report_cutoff and self.verbose,
        )
        if config.cutoff >= 0 or not report_cutoff:
            self._report(f"- Cutoff radius: {self.cutoff:.10f} Å")
        self._report("Analyzing crystal symmetries")
        self.symmetry = SymmetryOperations.from_atoms(
            self.primitive,
            self.supercell,
            symprec=config.symprec,
        )
        self._report(f"- Space group {self.symmetry.symbol}")
        self._report(f"- {self.symmetry.size} symmetry operations")
        self._orbit_space: OrbitSpace | None = None
        self._plan: DisplacementPlan | None = None

    @property
    def orbit_space(self) -> OrbitSpace:
        if self._orbit_space is None:
            self._report(
                f"Finding symmetry-inequivalent order-{self.config.order} interaction clusters"
            )
            self._orbit_space = build_orbit_space(
                self.supercell,
                self.index,
                self.symmetry,
                order=self.config.order,
                cutoff=self.cutoff,
            )
            dimensions = sum(orbit.dimension for orbit in self._orbit_space.orbits)
            self._report(f"- {len(self._orbit_space.orbits)} cluster equivalence classes")
            self._report(f"- {dimensions} independent tensor parameters")
        return self._orbit_space

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

    def sow(self, *, atom_order: str = "internal") -> list[Atoms]:
        """Return displaced structures in the exact positional reap order.

        Configuration ``i`` must be returned to positional :meth:`reap` at
        index ``i``. Each structure also carries its zero-based stable ID.
        """
        structures = list(self.plan)
        self._report(f"Sowing {len(structures)} displaced structures in {atom_order} atom order")
        self._report(f"- Plan hash: {self.plan.hash}")
        if atom_order == "internal":
            return structures
        if atom_order == "grouped":
            grouped: list[Atoms] = []
            for atoms in structures:
                reordered = self.index.group_atoms(atoms)
                reordered.info.update(atoms.info)
                reordered.info["mlfcs_atom_order"] = "grouped"
                grouped.append(reordered)
            return grouped
        raise ValueError("atom_order must be 'internal' or 'grouped'")

    def reap(
        self,
        forces: ForceInput,
        *,
        atom_order: str = "internal",
        plan_hash: str | None = None,
        acoustic_sum_rule: bool = True,
    ) -> ForceConstants:
        """Reconstruct force constants from forces supplied by the user.

        A sequence is positional and must follow :meth:`sow` exactly. A
        mapping is keyed by ``mlfcs_configuration_id`` and may arrive in any
        insertion order.
        """
        self._report(f"Reaping forces for order-{self.config.order} force constants")
        if plan_hash is not None and plan_hash != self.plan.hash:
            raise ValueError("force dataset plan hash does not match this calculation")
        values = self._normalize_forces(forces)
        self._report(f"- Validated {len(values)} force configurations")
        if atom_order == "grouped":
            values = values[:, self.index.internal_from_grouped, :]
        elif atom_order != "internal":
            raise ValueError("atom_order must be 'internal' or 'grouped'")
        derivatives = self.plan.contract_forces(values)
        self._report(f"- Contracted {len(derivatives)} finite-difference derivatives")
        self._report(
            "Reconstructing symmetry-expanded force constants "
            f"(ASR {'enabled' if acoustic_sum_rule else 'disabled'})"
        )
        sparse = reconstruct_sparse(
            self.orbit_space,
            self.index,
            derivatives,
            enforce_asr=acoustic_sum_rule,
        )
        self._report(f"- Reconstructed {len(sparse.clusters)} sparse cluster tensors")
        return ForceConstants(
            {},
            self.supercell.copy(),
            metadata={
                "order": self.config.order,
                "cutoff_angstrom": self.cutoff,
                "displacement_angstrom": self.config.displacement,
                "spacegroup": self.symmetry.symbol,
                "configurations": len(self.plan),
                "plan_hash": self.plan.hash,
                "acoustic_sum_rule": acoustic_sum_rule,
                "jax_platform": self.jax_platform,
            },
            sparse={self.config.order: sparse},
        )

    def run(
        self,
        calculator: Calculator,
        *,
        progress: Progress | None = None,
        acoustic_sum_rule: bool = True,
    ) -> ForceConstants:
        """Evaluate the sow list serially with a user-owned ASE Calculator."""
        forces = self.evaluate(calculator, progress=progress)
        return self.reap(
            forces,
            plan_hash=self.plan.hash,
            acoustic_sum_rule=acoustic_sum_rule,
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
        forces = np.empty((len(self.plan), len(self.supercell), 3), dtype=float)
        reporting_interval = max(1, ceil(len(self.plan) / 10))
        for configuration_id, atoms in enumerate(self.sow()):
            atoms.calc = calculator
            forces[configuration_id] = atoms.get_forces()
            if progress is not None:
                progress(configuration_id + 1, len(self.plan))
            elif self.verbose and (
                configuration_id == 0
                or configuration_id + 1 == len(self.plan)
                or (configuration_id + 1) % reporting_interval == 0
            ):
                completed = configuration_id + 1
                percentage = 100.0 * completed / len(self.plan)
                self._report(f"- Forces: {completed}/{len(self.plan)} ({percentage:.0f}%)")
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
