from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy import sparse

from mlfcs.constraints.translational import project_parameters
from mlfcs.fitting.backends.factory import create_fitting_backend
from mlfcs.fitting.constraints import build_joint_constraints
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.gram_system import (
    _force_metrics,
    _normalize_constraint_rows,
    _orbit_parameter_groups,
    _order_force_rms_from_reduced_gram,
    _StreamingGramSystem,
)
from mlfcs.fitting.jax_runtime import JaxPlatform, resolve_jax_device
from mlfcs.fitting.linear_solvers import (
    explicit_constraint_null_space,
    solve_scaled_group_lasso,
)
from mlfcs.fitting.parameterization import pack_order
from mlfcs.force_constants.expansion import expand_fitted_orders
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.interactions.space import InteractionSpace, ReferenceFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FittingResult:
    force_constants: ForceConstants
    fitting_parameters: np.ndarray
    fitting_basis: str
    parameter_scale: np.ndarray
    cache_directory: Path | None
    iterations: int
    training_force_rmse: float
    validation_force_rmse: float
    training_relative_force_error: float
    validation_relative_force_error: float
    order_force_rms: dict[int, float]
    stop_code: int
    residual_norm: float
    normal_equation_residual: float
    maximum_constraint_residual: float
    maximum_reference_force: float
    maximum_snapshot_net_force: float
    maximum_center_of_mass_displacement: float
    lowered_fc1_maximum: float | None
    lowered_fc1_net: float | None
    lowering_force_maximum: float | None = None
    lowering_force_relative: float | None = None
    regularization: str = "none"
    effective_noise_scale: float = 0.0
    active_orbits: int = 0
    admm_primal_residual: float = 0.0
    admm_dual_residual: float = 0.0
    design_kernel_signatures: int = 0
    design_tiles: int = 0
    static_device_bytes: int = 0
    gram_feature_passes: int = 0
    prediction_feature_passes: int = 0


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
        fitting_basis: str = "taylor",
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
        self._backend = create_fitting_backend(fitting_basis)
        self.fitting_basis = self._backend.name
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
                f"- FC{tensor.order}: {len(calculation.orbit_space.orbits)} orbits, "
                f"{np.count_nonzero(tensor.parameter_mask)} parameters"
            )
        self.order_tensors = tuple(tensors)
        self.n_parameters = offset
        self.index = self.calculations[0].index
        self.canonical_supercell = self.calculations[0].supercell
        logger.info(f"- Joint parameter count: {self.n_parameters}")

    def fit(
        self,
        structures: list[Atoms] | tuple[Atoms, ...],
        *,
        batch_size: int = 1,
        validation_split: float = 0.1,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        seed: int = 0,
        acoustic_sum_rule: bool = True,
        precondition: bool = True,
        allow_unconverged: bool = False,
        regularization: str | None = None,
        cache_directory: str | Path | None = None,
    ) -> FittingResult:
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if batch_size < 1 or batch_size > 4:
            raise ValueError("batch_size must be between 1 and 4")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        normalized_regularization = "none" if regularization is None else regularization.casefold()
        if normalized_regularization not in {"none", "scaled_group_lasso"}:
            raise ValueError("regularization must be None or 'scaled_group_lasso'")
        dataset = FitDataset.from_atoms(self.geometry, structures)
        maximum_reference_force = float(np.max(np.linalg.norm(dataset.reference_forces, axis=1)))
        maximum_snapshot_net_force = float(np.max(np.linalg.norm(dataset.net_forces, axis=1)))
        maximum_center_of_mass_displacement = float(
            np.max(np.linalg.norm(dataset.center_of_mass_displacements, axis=1))
        )
        logger.info("Training-data diagnostics (inputs are not recentered)")
        logger.info(f"- Maximum reference force: {maximum_reference_force:.10e} eV/Å")
        logger.info(f"- Maximum snapshot net force: {maximum_snapshot_net_force:.10e} eV/Å")
        logger.info(
            f"- Maximum center-of-mass displacement: {maximum_center_of_mass_displacement:.10e} Å"
        )
        # The reference frame is the public atom order.  Reordering here used
        # to hide a cell-major calculation frame and made fitted IFCs depend
        # on the incidental order of the input structure.
        displacements = dataset.displacements
        forces = dataset.forces
        residual_forces = forces
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(structures))
        n_validation = round(len(indices) * validation_split)
        validation = indices[:n_validation]
        training = indices[n_validation:]
        if not len(training):
            raise ValueError("validation split leaves no training structures")
        constraints = build_joint_constraints(
            self.calculations,
            acoustic=acoustic_sum_rule,
        )
        logger.info(
            f"Constraint system: {constraints.matrix.shape[0]} rows after duplicate removal "
            f"({constraints.translational_rows} ASR before compression)"
        )
        parameter_map = None
        if normalized_regularization == "none" and constraints.matrix.shape[0]:
            parameter_map = explicit_constraint_null_space(
                constraints.matrix,
                tolerance=1e-11,
            )
        prepared_basis = self._backend.prepare(
            calculations=self.calculations,
            training_displacements=displacements[training],
            parameterizations=self.order_tensors,
            n_parameters=self.n_parameters,
            batch_size=batch_size,
            parameter_map=parameter_map,
            device=self.jax_device,
        )
        operator = prepared_basis.operator
        target = residual_forces[training].reshape(-1)
        gram_system = _StreamingGramSystem.from_operator(
            operator,
            target,
            cache_directory=cache_directory,
        )
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
        logger.info(f"- Equations: {len(target)}, unknowns: {gram_system.gram.shape[0]}")
        solve_constraint_matrix = (
            sparse.csr_matrix((0, gram_system.gram.shape[0]))
            if parameter_map is not None
            else constraints.matrix
        )
        scaled_constraints = solve_constraint_matrix @ sparse.diags(parameter_scale)
        solve_constraints = _normalize_constraint_rows(scaled_constraints)
        effective_noise_scale = 0.0
        active_orbits = sum(
            len(calculation.orbit_space.orbits) for calculation in self.calculations
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
            groups = _orbit_parameter_groups(self.calculations)
            solution = solve_scaled_group_lasso(
                gram_system.gram,
                gram_system.rhs,
                gram_system.target_norm,
                parameter_scale,
                solve_constraints,
                groups,
                n_equations=len(target),
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
        constraint_residual = self._constraint_drift(parameters_numpy, constraints)
        training_metrics = gram_system.force_metrics(reduced_parameters, target)
        if n_validation:
            validation_operator = self._backend.build_operator(
                prepared_basis, displacements[validation]
            )
            validation_metrics = _force_metrics(
                validation_operator.matvec(parameters_numpy),
                residual_forces[validation].reshape(-1),
            )
        else:
            validation_metrics = training_metrics
        counts = [
            sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
            for calculation in self.calculations
        ]
        if parameter_map is None:
            order_force_rms = gram_system.order_force_rms(
                parameters_numpy, self.orders, counts, len(target)
            )
        else:
            order_force_rms = _order_force_rms_from_reduced_gram(
                gram_system,
                reduced_parameters,
                parameter_map,
                operator,
                parameters_numpy,
                self.orders,
                counts,
                len(target),
            )
        logger.info("Force fitting summary")
        logger.info(f"- Training relative error: {100 * training_metrics[1]:.6f} %")
        logger.info(f"- Validation relative error: {100 * validation_metrics[1]:.6f} %")
        logger.info(f"- Training force RMSE: {training_metrics[0]:.10e} eV/Å")
        logger.info(f"- Validation force RMSE: {validation_metrics[0]:.10e} eV/Å")
        for order, rms in order_force_rms.items():
            logger.info(f"- FC{order} force contribution RMS: {rms:.10e} eV/Å")
        logger.info(
            "- JAX execution guard: 1 prepared program, "
            f"{len(operator.program.groups)} signatures, {operator.program.tile_count} tiles, "
            f"Gram passes={operator.program.gram_feature_passes}, "
            f"prediction passes={operator.program.prediction_feature_passes}"
        )
        logger.info(f"- Solver iterations={iterations}, stop_code={stop_code}")
        if stop_code != 0:
            logger.warning(
                "Returning unconverged fitting solution: stop_code=%d, iterations=%d, residual=%.6e",
                stop_code,
                iterations,
                normal_residual,
            )
        lowering = self._backend.lower(prepared_basis, parameters_numpy)
        fc1 = lowering.reference_fc1
        fc1_maximum = float(np.max(np.abs(fc1))) if fc1 is not None and fc1.size else None
        fc1_net = float(np.linalg.norm(np.sum(fc1, axis=0))) if fc1 is not None else None
        if fc1 is not None:
            logger.info(
                "- Lowered FC1 diagnostic: "
                f"maximum={fc1_maximum:.10e} eV/Å, net={fc1_net:.10e} eV/Å"
            )
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
                "fitted_with": self.fitting_basis,
                "force_constants_basis": "taylor",
                "cutoff_angstrom": self.calculations[-1].cutoff,
                "cutoff_angstrom_by_order": {
                    calculation.config.order: calculation.cutoff
                    for calculation in self.calculations
                },
                "acoustic_sum_rule": acoustic_sum_rule,
                "training_structures": len(structures),
                "jax_platform": self.jax_platform,
            },
            sparse=sparse_values,
            relation=self.geometry,
        )
        result = FittingResult(
            force_constants=force_constants,
            fitting_parameters=parameters_numpy,
            fitting_basis=self.fitting_basis,
            parameter_scale=parameter_scale,
            cache_directory=gram_system.cache_directory,
            iterations=int(iterations),
            training_force_rmse=training_metrics[0],
            validation_force_rmse=validation_metrics[0],
            training_relative_force_error=training_metrics[1],
            validation_relative_force_error=validation_metrics[1],
            order_force_rms=order_force_rms,
            stop_code=int(stop_code),
            residual_norm=float(residual_norm),
            normal_equation_residual=float(normal_residual),
            maximum_constraint_residual=constraint_residual,
            maximum_reference_force=maximum_reference_force,
            maximum_snapshot_net_force=maximum_snapshot_net_force,
            maximum_center_of_mass_displacement=maximum_center_of_mass_displacement,
            lowered_fc1_maximum=fc1_maximum,
            lowered_fc1_net=fc1_net,
            lowering_force_maximum=lowering.lowering_force_maximum,
            lowering_force_relative=lowering.lowering_force_relative,
            regularization=normalized_regularization,
            effective_noise_scale=effective_noise_scale,
            active_orbits=active_orbits,
            admm_primal_residual=admm_primal,
            admm_dual_residual=admm_dual,
            design_kernel_signatures=len(operator.program.groups),
            design_tiles=operator.program.tile_count,
            static_device_bytes=operator.program.static_device_bytes,
            gram_feature_passes=operator.program.gram_feature_passes,
            prediction_feature_passes=operator.program.prediction_feature_passes,
        )
        return result

    def _constraint_drift(self, parameters, constraints):
        residual = constraints.matrix @ parameters
        maximum = float(np.max(np.abs(residual))) if len(residual) else 0.0
        logger.info(f"- Maximum joint constraint residual: {maximum:.6e}")
        return maximum

    def _report_parameter_scale(self, parameter_scale):
        logger.info("Column-norm preconditioning (exact from streamed Gram matrix)")
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
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
