from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy import sparse

from mlfcs.constraints.translational import project_parameters
from mlfcs.fitting.constraints import build_joint_constraints
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.gram import GramBuilder, GramStatistics
from mlfcs.fitting.jax_runtime import JaxPlatform, resolve_jax_device
from mlfcs.fitting.linear_solvers import (
    explicit_constraint_null_space,
    solve_scaled_group_lasso,
)
from mlfcs.fitting.parameterization import pack_order
from mlfcs.fitting.taylor.model import TaylorModel
from mlfcs.force_constants.expansion import expand_fitted_orders
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.interactions.space import InteractionSpace, ReferenceFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FittingResult:
    force_constants: ForceConstants
    fitting_parameters: np.ndarray
    parameter_scale: np.ndarray
    gram_statistics: GramStatistics
    iterations: int
    training_force_rmse: float
    training_relative_force_error: float
    order_force_rms: dict[int, float]
    stop_code: int
    residual_norm: float
    normal_equation_residual: float
    maximum_constraint_residual: float
    regularization: str = "none"
    effective_noise_scale: float = 0.0
    active_orbits: int = 0
    admm_primal_residual: float = 0.0
    admm_dual_residual: float = 0.0


class ForceConstantFitter:
    """Jointly fit consecutive symmetry-reduced IFC orders from ASE force snapshots."""

    def __init__(
        self,
        primitive: Atoms,
        reference: Atoms,
        *,
        orders: tuple[int, ...] = (2, 3),
        cutoffs: dict[int, float | int | None] | None = None,
        max_body_orders: dict[int, int | None] | None = None,
        symprec: float = 1e-5,
        jax_platform: JaxPlatform = "auto",
    ):
        self.jax_device = resolve_jax_device(jax_platform)
        frame = ReferenceFrame.from_atoms(primitive, reference, symprec=symprec)
        self.geometry = frame.relation
        self.primitive = self.geometry.primitive
        self.reference = self.geometry.reference
        self.supercell = self.geometry.supercell_matrix
        self.orders = tuple(sorted(set(orders)))
        if not self.orders or self.orders[0] < 2:
            raise ValueError("orders must contain integers greater than or equal to 2")
        if self.orders != tuple(range(self.orders[0], self.orders[-1] + 1)):
            raise ValueError(
                "orders must be consecutive so adjacent-order effects are identifiable"
            )
        self.cutoffs = dict(cutoffs or {})
        missing_cutoffs = tuple(order for order in self.orders if order not in self.cutoffs)
        if missing_cutoffs:
            raise ValueError(
                "a cutoff entry is required "
                f"for every fitted order; missing FC orders: {missing_cutoffs}"
            )
        self.max_body_orders = dict(max_body_orders or {})
        self.symprec = symprec
        self.jax_platform = jax_platform
        self._taylor = TaylorModel()
        order_text = "+".join(f"FC{order}" for order in self.orders)
        logger.info(f"Preparing independent {order_text} fitting parameterization")
        self.calculations = tuple(
            InteractionSpace.from_frame(
                frame,
                order=order,
                cutoff=self.cutoffs.get(order),
                max_body_order=self.max_body_orders.get(order),
                symprec=symprec,
            )
            for order in self.orders
        )
        offset = 0
        tensors = []
        for calculation in self.calculations:
            tensor, offset = pack_order(calculation, offset)
            tensors.append(tensor)
            logger.info(
                f"- FC{tensor.order}: {len(calculation.realized_orbit_space.orbits)} orbits, "
                f"{np.count_nonzero(tensor.parameter_mask)} parameters"
            )
        self.order_tensors = tuple(tensors)
        self.n_parameters = offset
        self.index = self.calculations[0].index
        self.canonical_supercell = self.calculations[0].supercell
        logger.info(f"- Joint parameter count: {self.n_parameters}")

    def fit(
        self,
        gram: GramStatistics,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        acoustic_sum_rule: bool = True,
        precondition: bool = True,
        allow_unconverged: bool = False,
        regularization: str | None = None,
    ) -> FittingResult:
        if not isinstance(gram, GramStatistics):
            raise TypeError("fit expects a GramStatistics object")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        normalized_regularization = "none" if regularization is None else regularization.casefold()
        if normalized_regularization not in {"none", "scaled_group_lasso"}:
            raise ValueError("regularization must be None or 'scaled_group_lasso'")
        constraints = build_joint_constraints(
            self.calculations,
            acoustic=acoustic_sum_rule,
        )
        logger.info(
            f"Constraint system: {constraints.matrix.shape[0]} rows after duplicate removal "
            f"({constraints.translational_rows} ASR before compression)"
        )
        parameter_map = gram.metadata.get("parameter_map")
        if parameter_map is None and normalized_regularization == "none" and constraints.matrix.shape[0]:
            parameter_map = explicit_constraint_null_space(
                constraints.matrix,
                tolerance=1e-11,
            )
        gram_system = gram
        if precondition:
            parameter_scale = gram_system.exact_column_scale()
            if parameter_map is None:
                self._report_parameter_scale(parameter_scale)
            else:
                active_scale = parameter_scale[parameter_scale > 0]
                logger.info("Column-norm preconditioning in constrained coordinates")
                if len(active_scale):
                    logger.info(
                        f"- Inverse column scale: {np.min(active_scale):.6e} to "
                        f"{np.max(active_scale):.6e}"
                    )
                else:
                    logger.info("- Inverse column scale: no active columns")
        else:
            parameter_scale = np.ones(gram_system.gram.shape[0])
            logger.info("- Parameter preconditioning disabled")
        if normalized_regularization == "none":
            logger.info("Solving the force-only least-squares problem with streamed Gram")
        else:
            logger.info("Solving the force-only problem with constrained scaled orbit-group LASSO")
        logger.info(f"- Equations: {gram.n_equations}, unknowns: {gram_system.gram.shape[0]}")
        solve_constraint_matrix = (
            sparse.csr_matrix((0, gram_system.gram.shape[0]))
            if parameter_map is not None
            else constraints.matrix
        )
        scaled_constraints = solve_constraint_matrix @ sparse.diags(parameter_scale)
        solve_constraints = self._normalize_constraint_rows(scaled_constraints)
        effective_noise_scale = 0.0
        active_orbits = sum(
            len(calculation.realized_orbit_space.orbits) for calculation in self.calculations
        )
        admm_primal = 0.0
        admm_dual = 0.0
        if normalized_regularization == "none":
            solution = gram_system.solve(
                parameter_scale,
                solve_constraints,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            scaled_parameters, stop_code, iterations, residual_norm, normal_residual = solution
        else:
            groups = self._orbit_parameter_groups()
            solution = solve_scaled_group_lasso(
                gram_system.gram,
                gram_system.rhs,
                gram_system.target_norm,
                parameter_scale,
                solve_constraints,
                groups,
                n_equations=gram.n_equations,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            (
                scaled_parameters,
                stop_code,
                iterations,
                residual_norm,
                normal_residual,
                effective_noise_scale,
                active_orbits,
                _cg_iterations,
                admm_primal,
                admm_dual,
            ) = solution
        if stop_code != 0 and not allow_unconverged:
            residual_label = (
                "ADMM residual"
                if normalized_regularization == "scaled_group_lasso"
                else "projected normal residual"
            )
            raise RuntimeError(
                "force-constant fitting did not converge: "
                f"stop_code={stop_code}, iterations={iterations}, "
                f"{residual_label}={normal_residual:.6e}; "
                "set allow_unconverged=True only to inspect the incomplete solution"
            )
        if solve_constraints.shape[0]:
            # Krylov stopping criteria control the full KKT residual and can
            # leave a visible equality-constraint tail.  Finish in null(C)
            # before converting back to physical FC parameters.
            projection_tolerance = tolerance / max(float(np.linalg.norm(scaled_parameters)), 1.0)
            scaled_parameters = project_parameters(
                solve_constraints,
                np.asarray(scaled_parameters),
                tolerance=projection_tolerance,
            )
        reduced_parameters = np.asarray(scaled_parameters) * parameter_scale
        parameters_numpy = (
            np.asarray(parameter_map @ reduced_parameters)
            if parameter_map is not None
            else reduced_parameters
        )
        constraint_residual = self._constraint_drift(
            parameters_numpy[: self.n_parameters], constraints
        )
        training_metrics = gram_system.force_metrics(reduced_parameters)
        counts = [
            sum(orbit.dimension for orbit in calculation.realized_orbit_space.orbits)
            for calculation in self.calculations
        ]
        if parameter_map is None:
            order_force_rms = gram_system.order_force_rms(
                parameters_numpy, self.orders, counts, gram.n_equations
            )
        else:
            # A reduced Gram does not contain cross-order physical blocks when
            # the constraint map mixes orders.  Do not recreate an operator or
            # silently perform a second feature pass for this diagnostic.
            order_force_rms = {}
        logger.info("Force fitting summary")
        logger.info(f"- Training relative error: {100 * training_metrics[1]:.6f} %")
        logger.info(f"- Training force RMSE: {training_metrics[0]:.10e} eV/Å")
        for order, rms in order_force_rms.items():
            logger.info(f"- FC{order} force contribution RMS: {rms:.10e} eV/Å")
        logger.info(
            "- JAX execution guard: 1 prepared program, "
            "independent Gram statistics"
        )
        logger.info(f"- Solver iterations={iterations}, stop_code={stop_code}")
        if stop_code != 0:
            logger.warning(
                "Returning unconverged fitting solution: stop_code=%d, iterations=%d, residual=%.6e",
                stop_code,
                iterations,
                normal_residual,
            )
        lowering = self._taylor.lower(None, parameters_numpy[: self.n_parameters])
        expansion_started = perf_counter()
        logger.info("Expanding fitted Taylor parameters into sparse physical IFCs")
        taylor_parameters = lowering.taylor_parameters
        residual_sparse = expand_fitted_orders(taylor_parameters, self.calculations)
        sparse_values = dict(residual_sparse)
        logger.info(
            f"- Expanded {sum(len(value.tensors) for value in sparse_values.values())} "
            f"sparse tensors in {perf_counter() - expansion_started:.2f} s"
        )
        force_constants = ForceConstants(
            {},
            self.canonical_supercell.copy(),
            metadata={
                "method": "joint_force_fit",
                "solver": (
                    "gram"
                    if normalized_regularization == "none"
                    else "gram_scaled_group_lasso_admm"
                ),
                "regularization": normalized_regularization,
                "fitted_with": "taylor",
                "force_constants_basis": "taylor",
                "cutoff_angstrom": self.calculations[-1].cutoff,
                "cutoff_angstrom_by_order": {
                    calculation.config.order: calculation.cutoff
                    for calculation in self.calculations
                },
                "acoustic_sum_rule": acoustic_sum_rule,
                "training_equations": gram.n_equations,
                "jax_platform": self.jax_platform,
            },
            sparse=sparse_values,
            relation=self.geometry,
        )
        result = FittingResult(
            force_constants=force_constants,
            fitting_parameters=parameters_numpy,
            parameter_scale=parameter_scale,
            gram_statistics=gram,
            iterations=int(iterations),
            training_force_rmse=training_metrics[0],
            training_relative_force_error=training_metrics[1],
            order_force_rms=order_force_rms,
            stop_code=int(stop_code),
            residual_norm=float(residual_norm),
            normal_equation_residual=float(normal_residual),
            maximum_constraint_residual=constraint_residual,
            regularization=normalized_regularization,
            effective_noise_scale=effective_noise_scale,
            active_orbits=active_orbits,
            admm_primal_residual=admm_primal,
            admm_dual_residual=admm_dual,
        )
        return result

    def evaluate_force_error(
        self,
        result: FittingResult,
        structures: list[Atoms] | tuple[Atoms, ...],
    ) -> tuple[float, float]:
        """Evaluate RMSE and relative force error for any supplied structures.

        The structures are evaluated through the public Taylor calculator;
        this method does not use or mutate the Gram cache and makes no
        distinction between training and test data.
        """
        if not isinstance(result, FittingResult):
            raise TypeError("result must be a FittingResult")
        dataset = FitDataset.from_atoms(self.geometry, structures)
        from mlfcs.calculators.ase import MLFCSCalculator

        calculator = MLFCSCalculator(result.force_constants, reference=self.reference)
        predicted = calculator.force_design_batch(dataset.displacements)
        residual = predicted - dataset.forces
        squared = float(np.sum(residual**2))
        count = residual.size
        target_squared = float(np.sum(dataset.forces**2))
        rmse = float(np.sqrt(squared / count)) if count else 0.0
        relative = float(np.sqrt(squared / target_squared)) if target_squared else (0.0 if squared == 0 else float("inf"))
        return rmse, relative

    def prepare_gram(
        self,
        structures: list[Atoms] | tuple[Atoms, ...],
        *,
        batch_size: int = 1,
        acoustic_sum_rule: bool = True,
    ) -> GramStatistics:
        """Build independent training statistics for a user-owned dataset."""
        if batch_size < 1 or batch_size > 4:
            raise ValueError("batch_size must be between 1 and 4")
        dataset = FitDataset.from_atoms(self.geometry, structures)
        constraints = build_joint_constraints(self.calculations, acoustic=acoustic_sum_rule)
        parameter_map = None
        if constraints.matrix.shape[0]:
            parameter_map = explicit_constraint_null_space(constraints.matrix)
        prepared = self._taylor.prepare(
            calculations=self.calculations,
            training_displacements=dataset.displacements,
            parameterizations=self.order_tensors,
            n_parameters=self.n_parameters,
            batch_size=batch_size,
            parameter_map=parameter_map,
            device=self.jax_device,
        )
        return GramBuilder.from_operator(
            prepared.operator,
            dataset.forces.reshape(-1),
            batch_size=batch_size,
        )

    def _orbit_parameter_groups(self):
        groups = []
        offset = 0
        for calculation in self.calculations:
            for orbit in calculation.realized_orbit_space.orbits:
                groups.append(slice(offset, offset + orbit.dimension))
                offset += orbit.dimension
        return tuple(groups)

    @staticmethod
    def _normalize_constraint_rows(constraints):
        if constraints.shape[0] == 0:
            return constraints
        norms = np.sqrt(np.asarray(constraints.multiply(constraints).sum(axis=1)).reshape(-1))
        scale = np.ones_like(norms)
        active = norms > np.finfo(float).tiny
        scale[active] = 1.0 / norms[active]
        return sparse.diags(scale) @ constraints

    def _constraint_drift(self, parameters, constraints):
        residual = constraints.matrix @ parameters
        maximum = float(np.max(np.abs(residual))) if len(residual) else 0.0
        logger.info(f"- Maximum joint constraint residual: {maximum:.6e}")
        return maximum

    def _report_parameter_scale(self, parameter_scale):
        logger.info("Column-norm preconditioning (exact from streamed Gram matrix)")
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.realized_orbit_space.orbits)
            values = parameter_scale[offset : offset + count]
            active = values[values > 0]
            if len(active):
                logger.info(
                    f"- FC{calculation.config.order} inverse column scale: "
                    f"{np.min(active):.6e} to {np.max(active):.6e}"
                )
            else:
                logger.info(
                    f"- FC{calculation.config.order} inverse column scale: no active columns"
                )
            offset += count
