from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from ase import Atoms
from scipy import sparse
from scipy.linalg.blas import dsyrk

from mlfcs.core.constraints import project_parameters
from mlfcs.core.geometry import StructureRelation
from mlfcs.core.interactions import InteractionSpace
from mlfcs.fitting.basis import symmetrized_covariance as _symmetrized_covariance
from mlfcs.fitting.constraints import (
    build_joint_constraints,
    build_wick_to_taylor_transform,
    omitted_taylor_fc1,
)
from mlfcs.fitting.data import FitDataset
from mlfcs.fitting.design import ForceDesignOperator as _BatchedForceOperator
from mlfcs.fitting.design import accumulate_physical_design
from mlfcs.fitting.design import prepare_design_kernel_groups as _prepare_physical_design_builders
from mlfcs.fitting.parameterization import expand_sparse as _expand_sparse
from mlfcs.fitting.parameterization import pack_order as _pack_order
from mlfcs.fitting.solver import (
    explicit_constraint_null_space,
    solve_gram_system,
    solve_scaled_group_lasso,
)
from mlfcs.model import ForceConstants
from mlfcs.runtime import JaxPlatform, resolve_jax_device


@dataclass(frozen=True, slots=True)
class FittingDiagnostics:
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
    omitted_taylor_fc1_maximum: float
    omitted_taylor_fc1_net: float
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


@dataclass(slots=True)
class FittingResult:
    force_constants: ForceConstants
    parameters: np.ndarray
    parameter_scale: np.ndarray
    covariance: np.ndarray
    diagnostics: FittingDiagnostics
    cache_directory: Path | None = None


@partial(jax.jit, donate_argnums=(0, 1))
def _update_device_statistics(current_gram, current_rhs, design, force):
    """Accumulate one device-resident reduced design batch into Gram statistics."""
    return current_gram + design.T @ design, current_rhs + design.T @ force


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
        verbose: bool = True,
    ):
        self.jax_device = resolve_jax_device(jax_platform)
        self.geometry = StructureRelation.from_atoms(primitive, reference, tolerance=symprec)
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
        self.max_body_orders = dict(max_body_orders or {})
        self.symprec = symprec
        self.jax_platform = jax_platform
        self.verbose = verbose
        order_text = "+".join(f"FC{order}" for order in self.orders)
        self._report(f"Preparing independent {order_text} fitting parameterization")
        self.calculations = tuple(
            InteractionSpace(
                self.primitive,
                order=order,
                reference=self.reference,
                cutoff=self.cutoffs.get(order),
                max_body_order=self.max_body_orders.get(order),
                symprec=symprec,
                reporter=self._report if verbose else None,
            )
            for order in self.orders
        )
        offset = 0
        tensors = []
        for calculation in self.calculations:
            tensor, offset = _pack_order(calculation, offset)
            tensors.append(tensor)
            self._report(
                f"- FC{tensor.order}: {len(calculation.orbit_space.orbits)} orbits, "
                f"{np.count_nonzero(tensor.parameter_mask)} parameters"
            )
        self.order_tensors = tuple(tensors)
        self.n_parameters = offset
        self.index = self.calculations[0].index
        self.canonical_supercell = self.calculations[0].supercell
        self._report(f"- Joint parameter count: {self.n_parameters}")

    def fit(
        self,
        structures: list[Atoms] | tuple[Atoms, ...],
        *,
        batch_size: int = 1,
        validation_split: float = 0.1,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        damping: float = 0.0,
        seed: int = 0,
        acoustic_sum_rule: bool = True,
        rotational_invariance: int = 0,
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
        if tolerance <= 0 or damping < 0:
            raise ValueError("tolerance must be positive and damping must be non-negative")
        normalized_regularization = "none" if regularization is None else regularization.casefold()
        if normalized_regularization not in {"none", "scaled_group_lasso"}:
            raise ValueError("regularization must be None or 'scaled_group_lasso'")
        if normalized_regularization != "none" and damping:
            raise ValueError("damping and scaled group LASSO cannot be enabled together")
        dataset = FitDataset.from_atoms(self.geometry, structures)
        maximum_reference_force = float(np.max(np.linalg.norm(dataset.reference_forces, axis=1)))
        maximum_snapshot_net_force = float(np.max(np.linalg.norm(dataset.net_forces, axis=1)))
        maximum_center_of_mass_displacement = float(
            np.max(np.linalg.norm(dataset.center_of_mass_displacements, axis=1))
        )
        self._report("Training-data diagnostics (inputs are not recentered)")
        self._report(f"- Maximum reference force: {maximum_reference_force:.10e} eV/Å")
        self._report(f"- Maximum snapshot net force: {maximum_snapshot_net_force:.10e} eV/Å")
        self._report(
            f"- Maximum center-of-mass displacement: {maximum_center_of_mass_displacement:.10e} Å"
        )
        # The reference frame is the public atom order.  Reordering here used
        # to hide a cell-major calculation frame and made fitted IFCs depend
        # on the incidental order of the input structure.
        displacements = dataset.displacements
        forces = dataset.forces
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(structures))
        n_validation = round(len(indices) * validation_split)
        validation = indices[:n_validation]
        training = indices[n_validation:]
        if not len(training):
            raise ValueError("validation split leaves no training structures")
        covariance = _symmetrized_covariance(displacements[training], self.calculations[0])
        constraints = build_joint_constraints(
            self.calculations,
            acoustic=acoustic_sum_rule,
            rotational_mode=rotational_invariance,
            covariance=covariance if rotational_invariance else None,
        )
        self._report(
            f"Constraint system: {constraints.matrix.shape[0]} rows after duplicate removal "
            f"({constraints.translational_rows} ASR, "
            f"{constraints.rotational_rows} rotational before compression)"
        )
        parameter_map = None
        if normalized_regularization == "none" and damping == 0.0 and constraints.matrix.shape[0]:
            parameter_map = explicit_constraint_null_space(
                constraints.matrix,
                tolerance=1e-11,
                reporter=self._report if self.verbose else None,
            )
        operator = _BatchedForceOperator(
            displacements[training],
            covariance,
            self.order_tensors,
            self.n_parameters,
            batch_size,
            parameter_map=parameter_map,
            reporter=self._report if self.verbose else None,
            device=self.jax_device,
        )
        target = forces[training].reshape(-1)
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
                self._report("Column-norm preconditioning in constrained coordinates")
                if len(active_scale):
                    self._report(
                        f"- Inverse column scale: {np.min(active_scale):.6e} to "
                        f"{np.max(active_scale):.6e}"
                    )
                else:
                    self._report("- Inverse column scale: no active columns")
        else:
            parameter_scale = np.ones(gram_system.gram.shape[0])
            self._report("- Parameter preconditioning disabled")
        if normalized_regularization == "none":
            self._report("Solving the force-only least-squares problem with streamed Gram")
        else:
            self._report("Solving the force-only problem with constrained scaled orbit-group LASSO")
        self._report(f"- Equations: {len(target)}, unknowns: {gram_system.gram.shape[0]}")
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
                damping=damping,
                verbose=self.verbose,
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
                verbose=self.verbose,
                reporter=self._report if self.verbose else None,
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
            validation_operator = operator.with_displacements(
                displacements[validation],
                reporter=self._report if self.verbose else None,
            )
            validation_metrics = _force_metrics(
                validation_operator.matvec(parameters_numpy), forces[validation].reshape(-1)
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
        self._report("Force fitting summary")
        self._report(f"- Training relative error: {100 * training_metrics[1]:.6f} %")
        self._report(f"- Validation relative error: {100 * validation_metrics[1]:.6f} %")
        self._report(f"- Training force RMSE: {training_metrics[0]:.10e} eV/Å")
        self._report(f"- Validation force RMSE: {validation_metrics[0]:.10e} eV/Å")
        for order, rms in order_force_rms.items():
            self._report(f"- FC{order} force contribution RMS: {rms:.10e} eV/Å")
        self._report(
            "- JAX execution guard: 1 prepared program, "
            f"{len(operator.program.groups)} signatures, {operator.program.tile_count} tiles, "
            f"Gram passes={operator.program.gram_feature_passes}, "
            f"prediction passes={operator.program.prediction_feature_passes}"
        )
        self._report(f"- Solver iterations={iterations}, stop_code={stop_code}")
        if stop_code != 0:
            self._report("- WARNING: returning an explicitly allowed unconverged solution")
        taylor_transform = build_wick_to_taylor_transform(self.calculations, covariance)
        fc1 = omitted_taylor_fc1(self.calculations, parameters_numpy, covariance)
        fc1_maximum = float(np.max(np.abs(fc1))) if fc1.size else 0.0
        fc1_net = float(np.linalg.norm(np.sum(fc1, axis=0)))
        self._report(
            f"- Omitted Taylor FC1: maximum={fc1_maximum:.10e} eV/Å, net={fc1_net:.10e} eV/Å"
        )
        expansion_started = perf_counter()
        self._report("Expanding fitted Taylor parameters into sparse physical IFCs")
        taylor_parameters = np.asarray(taylor_transform @ parameters_numpy)
        sparse_values = _expand_sparse(
            taylor_parameters,
            self.calculations,
            self.index.n_primitive,
            len(self.canonical_supercell),
        )
        self._report(
            f"- Expanded {sum(len(value.clusters) for value in sparse_values.values())} "
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
                "fitting_basis": "wick",
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
        diagnostics = FittingDiagnostics(
            int(iterations),
            training_metrics[0],
            validation_metrics[0],
            training_metrics[1],
            validation_metrics[1],
            order_force_rms,
            int(stop_code),
            float(residual_norm),
            float(normal_residual),
            constraint_residual,
            maximum_reference_force,
            maximum_snapshot_net_force,
            maximum_center_of_mass_displacement,
            fc1_maximum,
            fc1_net,
            normalized_regularization,
            effective_noise_scale,
            active_orbits,
            admm_primal,
            admm_dual,
            len(operator.program.groups),
            operator.program.tile_count,
            operator.program.static_device_bytes,
            operator.program.gram_feature_passes,
            operator.program.prediction_feature_passes,
        )
        result = FittingResult(
            force_constants,
            parameters_numpy,
            parameter_scale,
            covariance,
            diagnostics,
            gram_system.cache_directory,
        )
        return result

    def _constraint_drift(self, parameters, constraints):
        residual = constraints.matrix @ parameters
        maximum = float(np.max(np.abs(residual))) if len(residual) else 0.0
        self._report(f"- Maximum joint constraint residual: {maximum:.6e}")
        return maximum

    def _report_parameter_scale(self, parameter_scale):
        self._report("Column-norm preconditioning (exact from streamed Gram matrix)")
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
            values = parameter_scale[offset : offset + count]
            active = values[values > 0]
            if len(active):
                self._report(
                    f"- FC{calculation.config.order} inverse column scale: "
                    f"{np.min(active):.6e} to {np.max(active):.6e}"
                )
            else:
                self._report(
                    f"- FC{calculation.config.order} inverse column scale: no active columns"
                )
            offset += count

    def _report(self, message):
        if self.verbose:
            print(message, flush=True)


def _orbit_parameter_groups(calculations):
    """Return contiguous slices, one per symmetry-irreducible cluster orbit."""
    groups = []
    offset = 0
    for calculation in calculations:
        for orbit in calculation.orbit_space.orbits:
            groups.append(slice(offset, offset + orbit.dimension))
            offset += orbit.dimension
    return tuple(groups)


class _StreamingGramSystem:
    """Normal-equation sufficient statistics accumulated without storing A."""

    def __init__(self, gram, rhs, target_norm, reporter=None, cache_directory=None):
        self.gram = gram
        self.rhs = rhs
        self.target_norm = target_norm
        self.reporter = reporter
        self.cache_directory = cache_directory

    @classmethod
    def from_operator(cls, operator, target, *, cache_directory: str | Path | None = None):
        started = perf_counter()
        gram_nbytes = operator.fit_n_parameters**2 * np.dtype(np.float64).itemsize
        use_recovery_cache = gram_nbytes >= 64 * 1024**2
        if cache_directory is not None:
            cache_directory = (
                Path(cache_directory).expanduser().resolve()
                / f"gram-{_gram_recovery_key(operator, target)}"
            )
        elif use_recovery_cache:
            cache_key = _gram_recovery_key(operator, target)
            cache_directory = Path.cwd() / ".mlfcs-cache" / f"gram-{cache_key}"
        if cache_directory is not None and (cache_directory / "complete").exists():
            gram = np.load(cache_directory / "gram.npy")
            rhs = np.load(cache_directory / "rhs.npy")
            target_norm = float(np.load(cache_directory / "target_norm.npy"))
            if operator.reporter is not None:
                operator.reporter(
                    f"Recovered completed streamed Gram system from cache "
                    f"({perf_counter() - started:.2f} s)"
                )
            return cls(gram, rhs, target_norm, operator.reporter, cache_directory)
        use_device_gram = operator.device_gram
        if use_device_gram:
            gram = jax.device_put(
                np.zeros((operator.fit_n_parameters, operator.fit_n_parameters), dtype=float),
                operator.program.device,
            )
            rhs = jax.device_put(
                np.zeros(operator.fit_n_parameters, dtype=float), operator.program.device
            )
        else:
            gram = np.zeros(
                (operator.fit_n_parameters, operator.fit_n_parameters), dtype=float, order="F"
            )
            rhs = np.zeros(operator.fit_n_parameters, dtype=float)
        target_shaped = np.asarray(target).reshape(operator.force_shape)
        builders, effective_batch_size = _prepare_physical_design_builders(operator)
        rows_per_structure = int(np.prod(operator.force_shape[1:]))
        operator.program.gram_feature_passes += 1
        if operator.reporter is not None:
            tile_counts = [group.tile_count for group in builders]
            operator.reporter(
                f"Accumulating streamed Gram system: {operator.fit_n_parameters} x "
                f"{operator.fit_n_parameters} ({gram.nbytes / 1024**2:.1f} MiB), "
                f"effective_batch_size={effective_batch_size}, "
                f"backend={'JAX device' if use_device_gram else 'SciPy/OpenBLAS CPU'}"
            )
            operator.reporter(
                f"- Physical design kernel groups: {len(builders)}, "
                f"{sum(tile_counts)} bounded tiles"
            )

        for begin in range(0, len(operator.displacements), effective_batch_size):
            end = min(begin + effective_batch_size, len(operator.displacements))
            force_rows = (end - begin) * rows_per_structure
            if use_device_gram:
                design = jax.device_put(
                    np.zeros((force_rows, operator.n_parameters), dtype=float),
                    operator.program.device,
                )
            else:
                design = np.zeros((force_rows, operator.n_parameters), dtype=float)
            displacement_batch = jax.device_put(
                operator.displacements[begin:end], operator.program.device
            )
            for group in builders:
                order_started = perf_counter()
                columns = group.columns
                contributions = group.kernel(
                    displacement_batch,
                    operator.covariance,
                    *group.device_arguments,
                )
                # A physical tile contains only its local parameter columns.
                # Scatter after device execution so XLA never lowers one full
                # parameter-wide output for each kernel group.
                if use_device_gram:
                    design = accumulate_physical_design(design, contributions, group.device_columns)
                else:
                    for contribution, tile_columns in zip(
                        np.asarray(contributions), columns, strict=True
                    ):
                        contribution = contribution.reshape(force_rows, -1)
                        design[:, tile_columns] += contribution
                if operator.reporter is not None and begin == 0:
                    contributions.block_until_ready()
                    operator.reporter(
                        f"- Compiled FC{group.order} physical design kernel in "
                        f"{perf_counter() - order_started:.2f} s"
                    )
            if operator.parameter_map is not None:
                if use_device_gram:
                    # The sparse map is uploaded once in bounded COO chunks.
                    # This keeps physical design, reduction, and Gram updates
                    # on the device instead of round-tripping through SciPy.
                    design = operator.device_reduction(force_rows).apply(design)
                else:
                    design = np.asarray(operator.parameter_map.T @ design.T).T
            force = target_shaped[begin:end].reshape(-1)
            if use_device_gram:
                gram, rhs = _update_device_statistics(
                    gram,
                    rhs,
                    design,
                    jax.device_put(force, operator.program.device),
                )
            else:
                gram = dsyrk(
                    1.0,
                    a=design,
                    c=gram,
                    beta=1.0,
                    trans=1,
                    lower=0,
                    overwrite_c=1,
                )
                rhs += design.T @ force
            if operator.reporter is not None and (
                begin == 0 or end == len(operator.displacements) or end % 20 == 0
            ):
                if use_device_gram:
                    gram.block_until_ready()
                operator.reporter(
                    f"- Gram structures: {end}/{len(operator.displacements)}, "
                    f"elapsed={perf_counter() - started:.2f} s"
                )
        if use_device_gram:
            gram = np.asarray(jax.device_get(gram))
            rhs = np.asarray(jax.device_get(rhs))
        else:
            upper = np.triu(np.asarray(gram))
            gram = upper + np.triu(upper, 1).T
        if operator.reporter is not None:
            operator.reporter(f"- Streamed Gram system ready in {perf_counter() - started:.2f} s")
        target_norm = float(np.vdot(target, target))
        if cache_directory is not None:
            cache_directory.mkdir(parents=True, exist_ok=True)
            np.save(cache_directory / "gram.npy", gram)
            np.save(cache_directory / "rhs.npy", rhs)
            np.save(cache_directory / "target_norm.npy", np.asarray(target_norm))
            (cache_directory / "complete").write_text("mlfcs streamed Gram recovery cache\n")
        return cls(gram, rhs, target_norm, operator.reporter, cache_directory)

    def exact_column_scale(self):
        norm = np.sqrt(np.maximum(np.diag(self.gram), 0.0))
        threshold = max(float(np.max(norm)) * 1e-12, np.finfo(float).tiny)
        result = np.zeros_like(norm)
        active = norm > threshold
        result[active] = 1.0 / norm[active]
        return result

    def force_metrics(self, parameters, target):
        residual_squared = max(
            float(
                parameters @ self.gram @ parameters - 2 * parameters @ self.rhs + self.target_norm
            ),
            0.0,
        )
        relative = (
            float(np.sqrt(residual_squared / self.target_norm))
            if self.target_norm > 0
            else (0.0 if residual_squared == 0 else float("inf"))
        )
        return float(np.sqrt(residual_squared / len(target))), relative

    def order_force_rms(self, parameters, orders, counts, n_equations):
        result = {}
        offset = 0
        for order, count in zip(orders, counts, strict=True):
            values = parameters[offset : offset + count]
            block = self.gram[offset : offset + count, offset : offset + count]
            result[order] = float(np.sqrt(max(float(values @ block @ values), 0.0) / n_equations))
            offset += count
        return result

    def solve(self, scale, constraints, *, tolerance, max_iterations, damping, verbose):
        return solve_gram_system(
            self.gram,
            self.rhs,
            self.target_norm,
            scale,
            constraints,
            tolerance=tolerance,
            max_iterations=max_iterations,
            damping=damping,
            verbose=verbose,
            reporter=self.reporter,
        )


def _gram_recovery_key(operator, target):
    """Fingerprint every numerical input needed to safely reuse a failed run."""
    digest = sha256(b"mlfcs-streaming-gram-v3-compact-coordinates-null-space")
    arrays = [operator.displacements, np.asarray(operator.covariance), np.asarray(target)]
    if operator.parameter_map is not None:
        parameter_map = sparse.csc_matrix(operator.parameter_map)
        arrays.extend([parameter_map.data, parameter_map.indices, parameter_map.indptr])
    for tensor in operator.program.parameterizations:
        arrays.extend(
            [
                tensor.parameter_indices,
                tensor.parameter_mask,
                tensor.representative_from_pivots,
                tensor.rotations,
                tensor.component_permutations,
                tensor.coordinates,
                tensor.image_mask,
            ]
        )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()[:20]


def _force_metrics(predicted, reference):
    residual = np.asarray(predicted) - np.asarray(reference)
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.linalg.norm(reference))
    relative = float(np.linalg.norm(residual) / denominator) if denominator else float("inf")
    return rmse, relative


def _order_force_rms_from_reduced_gram(
    gram_system,
    reduced_parameters,
    parameter_map,
    operator,
    physical_parameters,
    orders,
    counts,
    n_equations,
):
    """Evaluate order contributions cheaply unless constraints mix IFC orders."""
    mapping = sparse.csc_matrix(parameter_map)
    boundaries = np.cumsum([0, *counts])
    column_orders = []
    mixed = False
    for column in range(mapping.shape[1]):
        rows = mapping.indices[mapping.indptr[column] : mapping.indptr[column + 1]]
        owners = np.searchsorted(boundaries[1:], rows, side="right")
        if len(np.unique(owners)) != 1:
            mixed = True
            break
        column_orders.append(int(owners[0]))
    if not mixed:
        result = {}
        column_orders = np.asarray(column_orders)
        for owner, order in enumerate(orders):
            selected = np.flatnonzero(column_orders == owner)
            values = reduced_parameters[selected]
            block = gram_system.gram[np.ix_(selected, selected)]
            result[order] = float(np.sqrt(max(float(values @ block @ values), 0.0) / n_equations))
        return result

    # Adjacent-order rotational identities genuinely couple orders.  In that
    # less common case, evaluate every physical order in one shared feature
    # pass instead of regenerating the same Wick features once per order.
    predicted = operator.matvec_by_order(physical_parameters)
    return {
        order: float(np.linalg.norm(predicted[order]) / np.sqrt(n_equations)) for order in orders
    }


def _normalize_constraint_rows(constraints):
    """Equilibrate equality rows without changing their common null space."""
    if constraints.shape[0] == 0:
        return constraints
    norms = np.sqrt(np.asarray(constraints.multiply(constraints).sum(axis=1)).reshape(-1))
    scale = np.ones_like(norms)
    active = norms > np.finfo(float).tiny
    scale[active] = 1.0 / norms[active]
    return sparse.diags(scale) @ constraints
