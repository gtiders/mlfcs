from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from math import factorial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy import sparse
from scipy.linalg import pinvh
from scipy.linalg.blas import dsyrk
from scipy.sparse.linalg import LinearOperator, cg, lsmr, minres

from mlfcs.api import ForceConstantCalculation
from mlfcs.fitting.constraints import build_joint_constraints, build_wick_to_taylor_transform
from mlfcs.fitting.data import FitDataset, ReferenceSupercell
from mlfcs.model import ForceConstants, SparseOrderForceConstants
from mlfcs.reconstruction.asr import (
    _project_parameters,
    maximum_acoustic_sum_rule_drift,
    project_acoustic_sum_rule,
)
from mlfcs.runtime import JaxPlatform, configure_jax


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
    estimated_condition: float
    asr_drift: dict[int, tuple[float, float]]


@dataclass(slots=True)
class FittingResult:
    force_constants: ForceConstants
    parameters: np.ndarray
    parameter_scale: np.ndarray
    covariance: np.ndarray
    diagnostics: FittingDiagnostics


@dataclass(frozen=True, slots=True)
class _OrderTensor:
    order: int
    parameter_indices: np.ndarray
    parameter_mask: np.ndarray
    representative_from_pivots: np.ndarray
    rotations: np.ndarray
    component_permutations: np.ndarray
    coordinates: np.ndarray
    image_mask: np.ndarray


class ForceConstantFitter:
    """Jointly fit consecutive symmetry-reduced IFC orders from ASE force snapshots."""

    def __init__(
        self,
        primitive: Atoms,
        reference: Atoms,
        *,
        supercell: tuple[int, int, int],
        orders: tuple[int, ...] = (2, 3),
        cutoffs: dict[int, float | int | None] | None = None,
        symprec: float = 1e-5,
        jax_platform: JaxPlatform = "auto",
        verbose: bool = True,
    ):
        configure_jax(jax_platform)
        self.geometry = ReferenceSupercell.from_atoms(primitive, reference, tolerance=symprec)
        if not np.array_equal(self.geometry.supercell_matrix, np.diag(supercell)):
            raise ValueError("supercell does not match the reference-supercell matrix")
        self.primitive = self.geometry.primitive
        self.reference = self.geometry.reference
        self.supercell = supercell
        self.orders = tuple(sorted(set(orders)))
        if not self.orders or self.orders[0] < 2:
            raise ValueError("orders must contain integers greater than or equal to 2")
        if self.orders != tuple(range(self.orders[0], self.orders[-1] + 1)):
            raise ValueError("orders must be consecutive so adjacent-order effects are identifiable")
        self.cutoffs = dict(cutoffs or {})
        self.symprec = symprec
        self.jax_platform = jax_platform
        self.verbose = verbose
        order_text = "+".join(f"FC{order}" for order in self.orders)
        self._report(f"Preparing independent {order_text} fitting parameterization")
        self.calculations = tuple(
            ForceConstantCalculation(
                self.primitive,
                order=order,
                supercell=supercell,
                cutoff=self.cutoffs.get(order),
                symprec=symprec,
                jax_platform=jax_platform,
                verbose=verbose,
            )
            for order in self.orders
        )
        self._validate_internal_order()
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
        solver: str = "lsmr",
        batch_size: int = 1,
        validation_split: float = 0.1,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        damping: float = 0.0,
        seed: int = 0,
        acoustic_sum_rule: bool = True,
        rotational_invariance: int = 0,
        precondition: bool = True,
        precondition_probes: int = 16,
        dense_dtype: str = "float64",
    ) -> FittingResult:
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if batch_size < 1 or batch_size > 4:
            raise ValueError("batch_size must be between 1 and 4")
        if max_iterations < 1 or precondition_probes < 1:
            raise ValueError("batch_size, max_iterations, and precondition_probes must be positive")
        if tolerance <= 0 or damping < 0:
            raise ValueError("tolerance must be positive and damping must be non-negative")
        if solver not in {"dense", "lsmr", "cached_lsmr", "gram"}:
            raise ValueError("solver must be 'dense', 'lsmr', 'cached_lsmr', or 'gram'")
        dataset = FitDataset.from_atoms(self.geometry, structures)
        permutation = self.geometry.internal_permutation
        displacements = dataset.displacements[:, permutation]
        forces = dataset.forces[:, permutation]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(structures))
        n_validation = round(len(indices) * validation_split)
        validation = indices[:n_validation]
        training = indices[n_validation:]
        if not len(training):
            raise ValueError("validation split leaves no training structures")
        covariance = _symmetrized_covariance(displacements[training], self.calculations[0])
        operator = _BatchedForceOperator(
            displacements[training],
            covariance,
            self.order_tensors,
            self.n_parameters,
            batch_size,
            reporter=self._report if self.verbose else None,
        )
        cached_operator = None
        gram_system = None
        solve_operator = operator
        if solver == "cached_lsmr":
            cached_operator = _CachedForceOperator.from_operator(operator)
            solve_operator = cached_operator
        target = forces[training].reshape(-1)
        if solver == "gram":
            gram_system = _StreamingGramSystem.from_operator(operator, target)
        if precondition:
            if gram_system is not None:
                parameter_scale = gram_system.exact_column_scale()
                scale_probes = "exact streamed"
            elif cached_operator is not None:
                parameter_scale = cached_operator.exact_column_scale()
                scale_probes = "exact cached"
            else:
                parameter_scale = operator.estimate_column_scale(precondition_probes, rng)
                scale_probes = str(precondition_probes)
                parameter_scale = self._stabilize_parameter_scale(parameter_scale)
            self._report_parameter_scale(parameter_scale, scale_probes)
        else:
            parameter_scale = np.ones(self.n_parameters)
            self._report("- Parameter preconditioning disabled")
        scaled_operator = solve_operator.scaled(parameter_scale)
        constraints = build_joint_constraints(
            self.calculations,
            acoustic=acoustic_sum_rule,
            rotational_mode=rotational_invariance,
            covariance=covariance if rotational_invariance else None,
        )
        self._report(
            f"Constraint system: {constraints.matrix.shape[0]} independent candidate rows "
            f"({constraints.translational_rows} ASR, "
            f"{constraints.rotational_rows} rotational before compression)"
        )
        self._report(f"Solving the force-only least-squares problem with {solver}")
        self._report(f"- Equations: {len(target)}, unknowns: {self.n_parameters}")
        scaled_constraints = constraints.matrix @ sparse.diags(parameter_scale)
        solve_constraints = _normalize_constraint_rows(scaled_constraints)
        if solver == "gram":
            solution = gram_system.solve(
                parameter_scale,
                solve_constraints,
                tolerance=tolerance,
                max_iterations=max_iterations,
                damping=damping,
                verbose=self.verbose,
            )
            scaled_parameters, stop_code, iterations, residual_norm, normal_residual, condition = solution
        elif solver == "dense":
            solution = _solve_dense(
                scaled_operator,
                solve_constraints,
                target,
                dtype=np.dtype(dense_dtype),
                tolerance=tolerance,
                reporter=self._report if self.verbose else None,
            )
            scaled_parameters, stop_code, iterations, residual_norm, normal_residual, condition = solution
        elif constraints.matrix.shape[0]:
            solution = _solve_constrained_lsmr(
                scaled_operator,
                solve_constraints,
                target,
                tolerance=tolerance,
                max_iterations=max_iterations,
                damping=damping,
                verbose=self.verbose,
            )
            scaled_parameters, stop_code, iterations, residual_norm, normal_residual, condition = solution
        else:
            raw = lsmr(
                scaled_operator, target, damp=damping, atol=tolerance, btol=tolerance,
                maxiter=max_iterations, show=self.verbose,
            )
            scaled_parameters, stop_code, iterations = raw[:3]
            residual_norm, normal_residual, condition = raw[3], raw[4], raw[6]
        if solve_constraints.shape[0]:
            # Krylov stopping criteria control the full KKT residual and can
            # leave a visible equality-constraint tail.  Finish in null(C)
            # before converting back to physical FC parameters.
            projection_tolerance = tolerance / max(
                float(np.linalg.norm(scaled_parameters)), 1.0
            )
            scaled_parameters = _project_parameters(
                solve_constraints,
                np.asarray(scaled_parameters),
                tolerance=projection_tolerance,
            )
        if cached_operator is not None:
            cached_operator.close()
        parameters_numpy = np.asarray(scaled_parameters) * parameter_scale
        drifts = self._constraint_drifts(parameters_numpy, constraints)
        if gram_system is not None:
            training_metrics = gram_system.force_metrics(parameters_numpy, target)
        else:
            training_metrics = _force_metrics(operator.matvec(parameters_numpy), target)
        if n_validation:
            validation_operator = _BatchedForceOperator(
                displacements[validation],
                covariance,
                self.order_tensors,
                self.n_parameters,
                batch_size,
                reporter=self._report if self.verbose else None,
            )
            validation_metrics = _force_metrics(
                validation_operator.matvec(parameters_numpy), forces[validation].reshape(-1)
            )
        else:
            validation_metrics = training_metrics
        if gram_system is not None:
            counts = [
                sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
                for calculation in self.calculations
            ]
            order_force_rms = gram_system.order_force_rms(
                parameters_numpy, self.orders, counts, len(target)
            )
        else:
            order_force_rms = self._order_force_rms(operator, parameters_numpy)
        self._report("Force fitting summary")
        self._report(f"- Training relative error: {100 * training_metrics[1]:.6f} %")
        self._report(f"- Validation relative error: {100 * validation_metrics[1]:.6f} %")
        self._report(f"- Training force RMSE: {training_metrics[0]:.10e} eV/Å")
        self._report(f"- Validation force RMSE: {validation_metrics[0]:.10e} eV/Å")
        for order, rms in order_force_rms.items():
            self._report(f"- FC{order} force contribution RMS: {rms:.10e} eV/Å")
        self._report(
            f"- Solver iterations={iterations}, stop_code={stop_code}, "
            f"condition≈{condition:.6e}"
        )
        sparse_values = _expand_sparse(
            parameters_numpy,
            self.calculations,
            self.index.n_primitive,
            len(self.canonical_supercell),
        )
        taylor_transform = build_wick_to_taylor_transform(self.calculations, covariance)
        taylor_parameters = np.asarray(taylor_transform @ parameters_numpy)
        sparse_values = _expand_sparse(
            taylor_parameters,
            self.calculations,
            self.index.n_primitive,
            len(self.canonical_supercell),
        )
        force_constants = ForceConstants(
            {},
            self.canonical_supercell.copy(),
            metadata={
                "method": "joint_force_fit",
                "solver": solver,
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
            float(condition),
            drifts,
        )
        result = FittingResult(
            force_constants,
            parameters_numpy,
            parameter_scale,
            covariance,
            diagnostics,
        )
        if gram_system is not None:
            gram_system.close()
        return result

    def _constraint_drifts(self, parameters, constraints):
        residual = constraints.matrix @ parameters
        maximum = float(np.max(np.abs(residual))) if len(residual) else 0.0
        self._report(f"- Maximum joint constraint residual: {maximum:.6e}")
        return {"joint": (maximum, maximum)}

    def _report_parameter_scale(self, parameter_scale, probes):
        if probes in {"exact cached", "exact streamed"}:
            source = "disk cache" if probes == "exact cached" else "streamed Gram matrix"
            self._report(f"Column-norm preconditioning (exact from {source})")
        else:
            self._report(f"Column-norm preconditioning ({probes} Hutchinson probes)")
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
            values = parameter_scale[offset : offset + count]
            active = values[values > 0]
            self._report(
                f"- FC{calculation.config.order} inverse column scale: "
                f"{np.min(active):.6e} to {np.max(active):.6e}"
            )
            offset += count

    def _stabilize_parameter_scale(self, parameter_scale):
        """Limit stochastic column-estimation outliers independently by order."""
        result = np.asarray(parameter_scale).copy()
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
            values = result[offset : offset + count]
            active = values[values > 0]
            if len(active):
                lower, upper = np.quantile(active, (0.01, 0.99))
                values[values > 0] = np.clip(values[values > 0], lower, upper)
            offset += count
        return result

    def _order_force_rms(self, operator, parameters):
        result = {}
        offset = 0
        for calculation in self.calculations:
            count = sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
            selected = np.zeros_like(parameters)
            selected[offset : offset + count] = parameters[offset : offset + count]
            force = operator.matvec(selected)
            result[calculation.config.order] = float(np.sqrt(np.mean(force**2)))
            offset += count
        return result

    def _apply_asr(self, parameters, *, enabled):
        if not enabled:
            self._report("- ASR disabled; constraint matrices were not constructed")
            return parameters.copy(), {
                calculation.config.order: (float("nan"), float("nan"))
                for calculation in self.calculations
            }
        output = parameters.copy()
        drifts = {}
        offset = 0
        for calculation in self.calculations:
            values = []
            for orbit in calculation.orbit_space.orbits:
                values.append(output[offset : offset + orbit.dimension])
                offset += orbit.dimension
            before = maximum_acoustic_sum_rule_drift(calculation.orbit_space, values)
            projected = project_acoustic_sum_rule(calculation.orbit_space, values) if enabled else values
            after = maximum_acoustic_sum_rule_drift(calculation.orbit_space, projected)
            begin = offset - sum(map(len, projected))
            for value in projected:
                output[begin : begin + len(value)] = value
                begin += len(value)
            drifts[calculation.config.order] = (before, after)
            self._report(f"- Max drift of FC{calculation.config.order}: {before:.6e} -> {after:.6e}")
        return output, drifts

    def _validate_internal_order(self):
        internal = self.reference[self.geometry.internal_permutation]
        canonical = self.calculations[0].supercell
        _, lengths = find_mic(
            internal.positions - canonical.positions, canonical.cell, pbc=True
        )
        if not np.array_equal(internal.numbers, canonical.numbers) or np.max(lengths) > 1e-4:
            raise ValueError("reference atom mapping does not match MLFCS internal geometry")

    def _report(self, message):
        if self.verbose:
            print(message, flush=True)


class _BatchedForceOperator(LinearOperator):
    """Matrix-free force design matrix evaluated in bounded JAX batches."""

    def __init__(
        self,
        displacements,
        covariance,
        tensors,
        n_parameters,
        batch_size,
        reporter=None,
    ):
        self.displacements = np.asarray(displacements)
        self.covariance = jnp.asarray(covariance)
        self.tensors = tensors
        self.n_parameters = n_parameters
        self.batch_size = batch_size
        self.force_shape = self.displacements.shape
        self.reporter = reporter
        self.forward_calls = 0
        self.transpose_calls = 0

        def forward(parameters, batch):
            return _predict_force(parameters, batch, self.covariance, self.tensors)

        def projected(parameters, batch, residual):
            return jnp.vdot(forward(parameters, batch), residual)

        self._forward = jax.jit(forward)
        self._transpose = jax.jit(jax.grad(projected, argnums=0))
        super().__init__(dtype=np.dtype(np.float64), shape=(self.displacements.size, n_parameters))

    def _matvec(self, parameters):
        self.forward_calls += 1
        started = perf_counter()
        parameters = jnp.asarray(np.asarray(parameters).reshape(-1))
        output = np.empty(self.force_shape, dtype=float)
        for begin in range(0, len(self.displacements), self.batch_size):
            end = min(begin + self.batch_size, len(self.displacements))
            output[begin:end] = np.asarray(
                self._forward(parameters, jnp.asarray(self.displacements[begin:end]))
            )
        self._report_operator("A·x", self.forward_calls, started)
        return output.reshape(-1)

    def _rmatvec(self, residual):
        self.transpose_calls += 1
        started = perf_counter()
        residual = np.asarray(residual).reshape(self.force_shape)
        parameters = jnp.zeros(self.n_parameters, dtype=jnp.float64)
        output = np.zeros(self.n_parameters, dtype=float)
        for begin in range(0, len(self.displacements), self.batch_size):
            end = min(begin + self.batch_size, len(self.displacements))
            output += np.asarray(
                self._transpose(
                    parameters,
                    jnp.asarray(self.displacements[begin:end]),
                    jnp.asarray(residual[begin:end]),
                )
            )
        self._report_operator("Aᵀ·r", self.transpose_calls, started)
        return output

    def scaled(self, parameter_scale):
        scale = np.asarray(parameter_scale)
        return LinearOperator(
            self.shape,
            matvec=lambda values: self.matvec(scale * np.asarray(values).reshape(-1)),
            rmatvec=lambda residual: scale * self.rmatvec(residual),
            dtype=np.float64,
        )

    def estimate_column_scale(self, probes, rng):
        squared_norm = np.zeros(self.n_parameters)
        if self.reporter is not None:
            self.reporter(f"Estimating column norms with {probes} stochastic probes")
        for probe_index in range(probes):
            probe = rng.choice((-1.0, 1.0), size=self.shape[0])
            squared_norm += self.rmatvec(probe) ** 2
            if self.reporter is not None:
                self.reporter(f"- Column-norm probe: {probe_index + 1}/{probes}")
        column_norm = np.sqrt(squared_norm / probes)
        threshold = max(float(np.max(column_norm)) * 1e-12, np.finfo(float).tiny)
        scale = np.zeros_like(column_norm)
        identified = column_norm > threshold
        scale[identified] = 1.0 / column_norm[identified]
        return scale

    def batch_normal(self, physical_parameters, selected):
        batch = jnp.asarray(self.displacements[selected])
        predicted = self._forward(jnp.asarray(physical_parameters), batch)
        zeros = jnp.zeros(self.n_parameters, dtype=jnp.float64)
        return np.asarray(self._transpose(zeros, batch, predicted))

    def _report_operator(self, name, call, started):
        if self.reporter is not None and (call <= 2 or call % 5 == 0):
            self.reporter(
                f"- {name} call {call}: {len(self.displacements)} structures in "
                f"{perf_counter() - started:.2f} s"
            )


class _CachedForceOperator(LinearOperator):
    """Disk-backed design matrix built automatically in bounded JAX batches."""

    def __init__(self, matrix, temporary_directory, reporter=None):
        self.matrix = matrix
        self._temporary_directory = temporary_directory
        self.reporter = reporter
        super().__init__(dtype=np.dtype(np.float64), shape=matrix.shape)

    @classmethod
    def from_operator(cls, operator):
        # Prefer a real writable filesystem over a potentially memory-backed
        # system /tmp. The private directory is still fully automatic and is
        # removed by close().
        cache_parent = Path.cwd()
        temporary_directory = TemporaryDirectory(prefix=".mlfcs-design-", dir=cache_parent)
        path = f"{temporary_directory.name}/force-design.dat"
        matrix = np.memmap(path, mode="w+", dtype=np.float64, shape=operator.shape)
        started = perf_counter()

        builders = []
        maximum_order = max(tensor.order for tensor in operator.tensors)
        orbit_block = 4 if maximum_order >= 4 else 16
        effective_batch_size = 1 if maximum_order >= 4 else operator.batch_size
        kernels = {}
        for tensor in operator.tensors:
            for orbit_begin in range(0, len(tensor.parameter_indices), orbit_block):
                chunk, column_begin, column_end = _slice_order_tensor(
                    tensor,
                    orbit_begin,
                    min(orbit_begin + orbit_block, len(tensor.parameter_indices)),
                )
                image_basis = _image_parameter_basis(chunk)
                capacity = len(chunk.parameter_indices) * chunk.parameter_indices.shape[1]
                kernel_key = (tensor.order, len(chunk.parameter_indices), capacity)
                if kernel_key not in kernels:
                    order = tensor.order

                    def design_batch(
                        displacements,
                        parameter_indices,
                        parameter_mask,
                        representative,
                        rotations,
                        permutations,
                        coordinates,
                        image_mask,
                        image_basis,
                        *,
                        order=order,
                        capacity=capacity,
                    ):
                        dynamic = _OrderTensor(
                            order,
                            parameter_indices,
                            parameter_mask,
                            representative,
                            rotations,
                            permutations,
                            coordinates,
                            image_mask,
                        )
                        return _force_design_batch(
                            displacements,
                            operator.covariance,
                            (dynamic,),
                            (image_basis,),
                            capacity,
                        )

                    kernels[kernel_key] = jax.jit(design_batch)
                builders.append(
                    (
                        column_begin,
                        column_end,
                        capacity,
                        kernels[kernel_key],
                        (
                            chunk.parameter_indices,
                            chunk.parameter_mask,
                            chunk.representative_from_pivots,
                            chunk.rotations,
                            chunk.component_permutations,
                            chunk.coordinates,
                            chunk.image_mask,
                            image_basis,
                        ),
                    )
                )
        rows_per_structure = int(np.prod(operator.force_shape[1:]))
        if operator.reporter is not None:
            gib = matrix.nbytes / 1024**3
            operator.reporter(
                f"Building automatic disk cache: {matrix.shape[0]} x {matrix.shape[1]} "
                f"({gib:.2f} GiB), effective_batch_size={effective_batch_size}"
            )
        for begin in range(0, len(operator.displacements), effective_batch_size):
            end = min(begin + effective_batch_size, len(operator.displacements))
            displacement_batch = jnp.asarray(operator.displacements[begin:end])
            force_rows = (end - begin) * rows_per_structure
            row_begin = begin * rows_per_structure
            for column_begin, column_end, capacity, build_design, arguments in builders:
                matrix[
                    row_begin : row_begin + force_rows, column_begin:column_end
                ] = np.asarray(
                    build_design(displacement_batch, *map(jnp.asarray, arguments))
                ).reshape(force_rows, capacity)[:, : column_end - column_begin]
            if operator.reporter is not None and (
                begin == 0 or end == len(operator.displacements) or end % 20 == 0
            ):
                operator.reporter(
                    f"- Cached structures: {end}/{len(operator.displacements)}"
                )
        matrix.flush()
        if operator.reporter is not None:
            operator.reporter(f"- Design cache ready in {perf_counter() - started:.2f} s")
        return cls(matrix, temporary_directory, operator.reporter)

    def _matvec(self, parameters):
        return np.asarray(self.matrix @ np.asarray(parameters).reshape(-1))

    def _rmatvec(self, residual):
        return np.asarray(self.matrix.T @ np.asarray(residual).reshape(-1))

    def scaled(self, parameter_scale):
        scale = np.asarray(parameter_scale)
        return LinearOperator(
            self.shape,
            matvec=lambda values: self.matvec(scale * np.asarray(values).reshape(-1)),
            rmatvec=lambda residual: scale * self.rmatvec(residual),
            dtype=np.float64,
        )

    def exact_column_scale(self):
        column_norm = np.sqrt(np.einsum("ij,ij->j", self.matrix, self.matrix))
        threshold = max(float(np.max(column_norm)) * 1e-12, np.finfo(float).tiny)
        scale = np.zeros_like(column_norm)
        active = column_norm > threshold
        scale[active] = 1.0 / column_norm[active]
        return scale

    def close(self):
        self.matrix.flush()
        del self.matrix
        self._temporary_directory.cleanup()


class _StreamingGramSystem:
    """Normal-equation sufficient statistics accumulated without storing A."""

    def __init__(self, gram, rhs, target_norm, reporter=None, cache_directory=None):
        self.gram = gram
        self.rhs = rhs
        self.target_norm = target_norm
        self.reporter = reporter
        self.cache_directory = cache_directory

    @classmethod
    def from_operator(cls, operator, target):
        started = perf_counter()
        gram_nbytes = operator.n_parameters**2 * np.dtype(np.float64).itemsize
        use_recovery_cache = gram_nbytes >= 64 * 1024**2
        cache_directory = None
        if use_recovery_cache:
            cache_key = _gram_recovery_key(operator, target)
            cache_directory = Path.cwd() / ".mlfcs-cache" / f"gram-{cache_key}"
        if cache_directory is not None and (cache_directory / "complete").exists():
            gram = np.load(cache_directory / "gram.npy")
            rhs = np.load(cache_directory / "rhs.npy")
            target_norm = float(np.load(cache_directory / "target_norm.npy"))
            if operator.reporter is not None:
                operator.reporter(
                    f"Recovered completed streamed Gram system from internal cache "
                    f"({perf_counter() - started:.2f} s)"
                )
            return cls(
                gram, rhs, target_norm, operator.reporter, cache_directory
            )
        use_gpu = jax.default_backend() == "gpu"
        if use_gpu:
            gram = jnp.zeros(
                (operator.n_parameters, operator.n_parameters), dtype=jnp.float64
            )
            rhs = jnp.zeros(operator.n_parameters, dtype=jnp.float64)
        else:
            gram = np.zeros(
                (operator.n_parameters, operator.n_parameters), dtype=float, order="F"
            )
            rhs = np.zeros(operator.n_parameters, dtype=float)
        target_shaped = np.asarray(target).reshape(operator.force_shape)
        builders, effective_batch_size = _prepare_fused_design_builders(operator)
        rows_per_structure = int(np.prod(operator.force_shape[1:]))
        if operator.reporter is not None:
            operator.reporter(
                f"Accumulating streamed Gram system: {operator.n_parameters} x "
                f"{operator.n_parameters} ({gram.nbytes / 1024**2:.1f} MiB), "
                f"effective_batch_size={effective_batch_size}, "
                f"backend={'JAX GPU' if use_gpu else 'SciPy/OpenBLAS CPU'}"
            )

        @partial(jax.jit, donate_argnums=(0, 1))
        def update_device_statistics(current_gram, current_rhs, design, force):
            return current_gram + design.T @ design, current_rhs + design.T @ force

        for begin in range(0, len(operator.displacements), effective_batch_size):
            end = min(begin + effective_batch_size, len(operator.displacements))
            force_rows = (end - begin) * rows_per_structure
            if use_gpu:
                design = jnp.zeros((force_rows, operator.n_parameters), dtype=jnp.float64)
            else:
                design = np.zeros((force_rows, operator.n_parameters), dtype=float)
            displacement_batch = jnp.asarray(operator.displacements[begin:end])
            for order, build_design, arguments in builders:
                order_started = perf_counter()
                contribution = build_design(
                    displacement_batch, *map(jnp.asarray, arguments)
                ).reshape(force_rows, operator.n_parameters)
                if use_gpu:
                    design = design + contribution
                else:
                    design += np.asarray(contribution)
                if operator.reporter is not None and begin == 0:
                    contribution.block_until_ready()
                    operator.reporter(
                        f"- Compiled FC{order} fused design kernel in "
                        f"{perf_counter() - order_started:.2f} s"
                    )
            force = target_shaped[begin:end].reshape(-1)
            if use_gpu:
                gram, rhs = update_device_statistics(
                    gram, rhs, design, jnp.asarray(force)
                )
            else:
                gram = dsyrk(
                    1.0, a=design, c=gram, beta=1.0, trans=1, lower=0,
                    overwrite_c=1,
                )
                rhs += design.T @ force
            if operator.reporter is not None and (
                begin == 0 or end == len(operator.displacements) or end % 20 == 0
            ):
                if use_gpu:
                    gram.block_until_ready()
                operator.reporter(
                    f"- Gram structures: {end}/{len(operator.displacements)}, "
                    f"elapsed={perf_counter() - started:.2f} s"
                )
        if use_gpu:
            gram = np.asarray(gram)
            rhs = np.asarray(rhs)
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
            (cache_directory / "complete").write_text(
                "mlfcs streamed Gram recovery cache\n"
            )
        return cls(
            gram, rhs, target_norm, operator.reporter, cache_directory
        )

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
                parameters @ self.gram @ parameters
                - 2 * parameters @ self.rhs
                + self.target_norm
            ),
            0.0,
        )
        return (
            float(np.sqrt(residual_squared / len(target))),
            float(np.sqrt(residual_squared / self.target_norm)),
        )

    def order_force_rms(self, parameters, orders, counts, n_equations):
        result = {}
        offset = 0
        for order, count in zip(orders, counts, strict=True):
            values = parameters[offset : offset + count]
            block = self.gram[offset : offset + count, offset : offset + count]
            result[order] = float(
                np.sqrt(max(float(values @ block @ values), 0.0) / n_equations)
            )
            offset += count
        return result

    def solve(self, scale, constraints, *, tolerance, max_iterations, damping, verbose):
        scale = np.asarray(scale)
        normal = self.gram * scale[:, None] * scale[None, :]
        if damping:
            normal.flat[:: len(normal) + 1] += damping**2
        rhs = scale * self.rhs
        n_parameters = len(rhs)
        n_constraints = constraints.shape[0]
        started = perf_counter()
        previous = started
        iterations = 0
        if n_constraints:
            projector = _ConstraintNullSpace(constraints, reporter=self.reporter)
            projected_rhs = projector.project(rhs)

            def multiply(values):
                return projector.project(normal @ projector.project(values))

            system = LinearOperator(
                (n_parameters,) * 2,
                matvec=multiply,
                rmatvec=multiply,
                dtype=np.float64,
            )

            def callback(values):
                nonlocal iterations, previous
                iterations += 1
                now = perf_counter()
                if verbose and (iterations <= 5 or iterations % 100 == 0):
                    drift = np.linalg.norm(constraints @ values[:n_parameters], ord=np.inf)
                    gradient = multiply(values) - projected_rhs
                    relative_gradient = np.linalg.norm(gradient) / max(
                        np.linalg.norm(projected_rhs), np.finfo(float).tiny
                    )
                    print(
                        f"Projected CG iteration {iterations}: relative gradient="
                        f"{relative_gradient:.6e}, max constraint residual={drift:.6e}, "
                        f"step={now - previous:.3f} s, elapsed={now - started:.2f} s",
                        flush=True,
                    )
                previous = now

            parameters, info = cg(
                system,
                projected_rhs,
                x0=np.zeros(n_parameters),
                rtol=tolerance,
                atol=0.0,
                maxiter=max_iterations,
                callback=callback,
            )
            parameters = projector.project(parameters)
            stationarity = projector.project(normal @ parameters - rhs)
        else:
            def callback(_values):
                nonlocal iterations
                iterations += 1

            parameters, info = cg(
                normal,
                rhs,
                x0=np.zeros(n_parameters),
                rtol=tolerance,
                atol=0.0,
                maxiter=max_iterations,
                callback=callback,
            )
            stationarity = normal @ parameters - rhs
        residual_squared = max(
            float(parameters @ normal @ parameters - 2 * parameters @ rhs + self.target_norm),
            0.0,
        )
        return (
            parameters,
            int(info),
            iterations,
            float(np.sqrt(residual_squared)),
            float(np.linalg.norm(stationarity)),
            float("nan"),
        )

    def close(self):
        if self.cache_directory is not None and self.cache_directory.exists():
            shutil.rmtree(self.cache_directory)


def _gram_recovery_key(operator, target):
    """Fingerprint every numerical input needed to safely reuse a failed run."""
    digest = sha256(b"mlfcs-streaming-gram-v1")
    arrays = [operator.displacements, np.asarray(operator.covariance), np.asarray(target)]
    for tensor in operator.tensors:
        arrays.extend(
            [tensor.parameter_indices, tensor.parameter_mask,
             tensor.representative_from_pivots, tensor.rotations,
             tensor.component_permutations, tensor.coordinates, tensor.image_mask]
        )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()[:20]


class _ConstraintNullSpace:
    """Implicit orthogonal projector onto null(C), including redundant rows."""

    def __init__(self, constraints, reporter=None):
        self.constraints = sparse.csr_matrix(constraints)
        row_gram = (self.constraints @ self.constraints.T).toarray()
        row_gram = (row_gram + row_gram.T) * 0.5
        self.row_gram_inverse, self.rank = pinvh(
            row_gram, return_rank=True, check_finite=False
        )
        if reporter is not None:
            reporter(
                f"Implicit constraint null space: numerical rank={self.rank}/"
                f"{self.constraints.shape[0]}, redundant rows="
                f"{self.constraints.shape[0] - self.rank}"
            )

    def project(self, values):
        values = np.asarray(values)
        residual = self.constraints @ values
        multipliers = self.row_gram_inverse @ residual
        return values - self.constraints.T @ multipliers


def _prepare_design_builders(operator):
    """Create internal fixed-shape design kernels; no cache controls enter the API."""
    builders = []
    maximum_order = max(tensor.order for tensor in operator.tensors)
    orbit_block = 4 if maximum_order >= 4 else 16
    effective_batch_size = 1 if maximum_order >= 4 else operator.batch_size
    kernels = {}
    for tensor in operator.tensors:
        for orbit_begin in range(0, len(tensor.parameter_indices), orbit_block):
            chunk, column_begin, column_end = _slice_order_tensor(
                tensor, orbit_begin,
                min(orbit_begin + orbit_block, len(tensor.parameter_indices)),
            )
            image_basis = _image_parameter_basis(chunk)
            capacity = len(chunk.parameter_indices) * chunk.parameter_indices.shape[1]
            key = (tensor.order, len(chunk.parameter_indices), capacity)
            if key not in kernels:
                order = tensor.order

                def design_batch(displacements, *arguments, order=order, capacity=capacity):
                    dynamic = _OrderTensor(order, *arguments[:-1])
                    return _force_design_batch(
                        displacements, operator.covariance, (dynamic,),
                        (arguments[-1],), capacity,
                    )

                kernels[key] = jax.jit(design_batch)
            builders.append((
                column_begin, column_end, capacity, kernels[key],
                (chunk.parameter_indices, chunk.parameter_mask,
                 chunk.representative_from_pivots, chunk.rotations,
                 chunk.component_permutations, chunk.coordinates,
                 chunk.image_mask, image_basis),
            ))
    return builders, effective_batch_size


def _prepare_fused_design_builders(operator):
    """Create shape-bucketed orbit scans without worst-case orbit padding."""
    builders = []
    for tensor in operator.tensors:
        order = tensor.order
        image_counts = np.sum(tensor.image_mask, axis=1)
        dimension_counts = np.sum(tensor.parameter_mask, axis=1)
        shapes = sorted(set(zip(image_counts.tolist(), dimension_counts.tolist())))
        for n_images, n_dimensions in shapes:
            selected = np.flatnonzero(
                (image_counts == n_images) & (dimension_counts == n_dimensions)
            )
            bucket = _bucket_order_tensor(tensor, selected, n_images, n_dimensions)
            image_basis = _image_parameter_basis(bucket)

            def design_bucket(
                displacements,
                parameter_indices,
                parameter_mask,
                representative,
                rotations,
                permutations,
                coordinates,
                image_mask,
                image_basis,
                *,
                order=order,
            ):
                initial = jnp.zeros(
                    (len(displacements), displacements.shape[1] * 3, operator.n_parameters),
                    dtype=jnp.float64,
                )

                def orbit_step(design, values):
                    indices, mask, rep, rotation, permutation, coordinate, images, basis = values
                    orbit = _OrderTensor(
                        order, indices[None, :], mask[None, :], rep[None, :, :],
                        rotation[None, :, :, :], permutation[None, :, :],
                        coordinate[None, :, :, :, :], images[None, :],
                    )
                    contribution = _force_design_batch(
                        displacements, operator.covariance, (orbit,),
                        (basis[None, :, :, :],), operator.n_parameters,
                    )
                    return design + contribution, None

                result, _ = jax.lax.scan(
                    orbit_step, initial,
                    (parameter_indices, parameter_mask, representative, rotations,
                     permutations, coordinates, image_mask, image_basis),
                )
                return result

            builders.append((
                order,
                jax.jit(design_bucket),
                (bucket.parameter_indices, bucket.parameter_mask,
                 bucket.representative_from_pivots, bucket.rotations,
                 bucket.component_permutations, bucket.coordinates,
                 bucket.image_mask, image_basis),
            ))
    return builders, operator.batch_size


def _bucket_order_tensor(tensor, selected, n_images, n_dimensions):
    """Remove image and parameter padding for one orbit-shape bucket."""
    return _OrderTensor(
        tensor.order,
        tensor.parameter_indices[selected, :n_dimensions],
        tensor.parameter_mask[selected, :n_dimensions],
        tensor.representative_from_pivots[selected, :, :n_dimensions],
        tensor.rotations[selected, :n_images],
        tensor.component_permutations[selected, :n_images],
        tensor.coordinates[selected, :n_images],
        tensor.image_mask[selected, :n_images],
    )


def _force_metrics(predicted, reference):
    residual = np.asarray(predicted) - np.asarray(reference)
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.linalg.norm(reference))
    relative = float(np.linalg.norm(residual) / denominator) if denominator else float("inf")
    return rmse, relative


def _normalize_constraint_rows(constraints):
    """Equilibrate equality rows without changing their common null space."""
    if constraints.shape[0] == 0:
        return constraints
    norms = np.sqrt(np.asarray(constraints.multiply(constraints).sum(axis=1)).reshape(-1))
    scale = np.ones_like(norms)
    active = norms > np.finfo(float).tiny
    scale[active] = 1.0 / norms[active]
    return sparse.diags(scale) @ constraints


def _solve_constrained_lsmr(
    operator,
    constraints,
    target,
    *,
    tolerance,
    max_iterations,
    damping,
    verbose,
):
    """Solve the equality-constrained least-squares KKT system matrix-free."""
    n_parameters = operator.shape[1]
    n_constraints = constraints.shape[0]

    def matvec(values):
        parameters = values[:n_parameters]
        multipliers = values[n_parameters:]
        normal = operator.rmatvec(operator.matvec(parameters))
        if damping:
            normal = normal + damping**2 * parameters
        return np.concatenate(
            [normal + constraints.T @ multipliers, constraints @ parameters]
        )

    kkt = LinearOperator(
        (n_parameters + n_constraints,) * 2,
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    rhs = np.concatenate([operator.rmatvec(target), np.zeros(n_constraints)])
    counter = 0
    solve_started = perf_counter()
    previous_iteration = solve_started

    def callback(values):
        nonlocal counter, previous_iteration
        counter += 1
        now = perf_counter()
        if verbose and (counter <= 5 or counter % 10 == 0):
            constraint_residual = np.linalg.norm(
                constraints @ values[:n_parameters], ord=np.inf
            )
            print(
                f"KKT iteration {counter}: max constraint residual="
                f"{constraint_residual:.6e}, step={now - previous_iteration:.3f} s, "
                f"elapsed={now - solve_started:.2f} s",
                flush=True,
            )
        previous_iteration = now

    solution, info = minres(
        kkt,
        rhs,
        rtol=tolerance,
        maxiter=max_iterations,
        callback=callback,
        show=False,
        check=False,
    )
    parameters = solution[:n_parameters]
    force_residual = operator.matvec(parameters) - target
    stationarity = operator.rmatvec(force_residual) + constraints.T @ solution[n_parameters:]
    return (
        parameters,
        int(info),
        counter,
        float(np.linalg.norm(force_residual)),
        float(np.linalg.norm(stationarity)),
        float("nan"),
    )


def _project_scaled_constraints(parameters, constraints, tolerance):
    if constraints.shape[0] == 0:
        return parameters
    return _project_parameters(constraints, np.asarray(parameters), tolerance=tolerance)


def _solve_dense(operator, constraints, target, *, dtype, tolerance, reporter):
    """Materialize the design matrix and solve an exact dense least-squares problem."""
    started = perf_counter()
    matrix = np.empty(operator.shape, dtype=dtype)
    n_parameters = operator.shape[1]
    basis = np.zeros(n_parameters)
    for column in range(n_parameters):
        basis[column] = 1.0
        matrix[:, column] = operator.matvec(basis)
        basis[column] = 0.0
        if reporter is not None and (column < 5 or (column + 1) % 100 == 0):
            reporter(f"- Dense design columns: {column + 1}/{operator.n_parameters}")
    matrix64 = np.asarray(matrix, dtype=np.float64)
    if constraints.shape[0]:
        # Dense backend uses a KKT solve; intended only for small problems.
        normal = matrix64.T @ matrix64
        kkt = np.block(
            [[normal, constraints.T.toarray()],
             [constraints.toarray(), np.zeros((constraints.shape[0], constraints.shape[0]))]]
        )
        rhs = np.concatenate([matrix64.T @ target, np.zeros(constraints.shape[0])])
        solution = np.linalg.lstsq(kkt, rhs, rcond=tolerance)[0][:n_parameters]
    else:
        solution = np.linalg.lstsq(matrix64, target, rcond=tolerance)[0]
    residual = matrix64 @ solution - target
    if reporter is not None:
        reporter(f"- Dense solve completed in {perf_counter() - started:.2f} s")
    return (
        solution,
        0,
        1,
        float(np.linalg.norm(residual)),
        float(np.linalg.norm(matrix64.T @ residual)),
        float(np.linalg.cond(matrix64)),
    )


def _pack_order(calculation, offset):
    orbit_space = calculation.orbit_space
    order = orbit_space.order
    orbits = orbit_space.orbits
    n_orbits = len(orbits)
    max_images = max(len(orbit.images) for orbit in orbits)
    max_dimension = max(orbit.dimension for orbit in orbits)
    components = np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int32)
    parameter_indices = np.zeros((n_orbits, max_dimension), dtype=np.int32)
    parameter_mask = np.zeros_like(parameter_indices, dtype=bool)
    representatives = np.zeros((n_orbits, 3**order, max_dimension))
    rotations = np.zeros((n_orbits, max_images, 3, 3))
    permutations = np.zeros((n_orbits, max_images, 3**order), dtype=np.int32)
    coordinates = np.zeros(
        (n_orbits, max_images, len(calculation.index.translations) // calculation.index.n_primitive, 3**order, order),
        dtype=np.int32,
    )
    image_mask = np.zeros((n_orbits, max_images), dtype=bool)
    translations = np.unique(calculation.index.translations, axis=0)
    base = np.arange(3**order).reshape((3,) * order)
    for orbit_index, orbit in enumerate(orbits):
        dimension = orbit.dimension
        images = len(orbit.images)
        parameter_indices[orbit_index, :dimension] = np.arange(offset, offset + dimension)
        parameter_mask[orbit_index, :dimension] = True
        representatives[orbit_index, :, :dimension] = np.linalg.solve(
            orbit.basis[orbit.pivots].T, orbit.basis.T
        ).T
        for image_index, image in enumerate(orbit.images):
            rotations[orbit_index, image_index] = image.action.rotation
            permutations[orbit_index, image_index] = base.transpose(
                image.action.permutation
            ).ravel()
            for translation_index, translation in enumerate(translations):
                atoms = [
                    calculation.index.translate_atom(atom, translation)
                    for atom in image.cluster
                ]
                coordinates[orbit_index, image_index, translation_index] = (
                    np.asarray(atoms)[None, :] * 3 + components
                )
        image_mask[orbit_index, :images] = True
        offset += dimension
    return (
        _OrderTensor(
            order,
            parameter_indices,
            parameter_mask,
            representatives,
            rotations,
            permutations,
            coordinates,
            image_mask,
        ),
        offset,
    )


def _image_parameter_basis(tensor):
    """Map each symmetry-image tensor component to independent parameters."""
    order = tensor.order
    representative = tensor.representative_from_pivots
    result = np.zeros(
        (*tensor.rotations.shape[:2], 3**order, representative.shape[-1]),
        dtype=float,
    )
    for orbit in range(len(representative)):
        for image in range(tensor.rotations.shape[1]):
            rotation = tensor.rotations[orbit, image]
            for dimension in range(representative.shape[-1]):
                value = representative[orbit, :, dimension].reshape((3,) * order)
                for axis in range(order):
                    value = np.tensordot(rotation, value, axes=((1,), (axis,)))
                    value = np.moveaxis(value, 0, axis)
                result[orbit, image, :, dimension] = value.reshape(-1)
    component_indices = tensor.component_permutations[..., None]
    return np.take_along_axis(result, component_indices, axis=2)


def _slice_order_tensor(tensor, begin, end):
    """Extract a contiguous orbit block and make its parameter indices local."""
    mask = tensor.parameter_mask[begin:end]
    active = tensor.parameter_indices[begin:end][mask]
    column_begin = int(np.min(active))
    column_end = int(np.max(active)) + 1
    indices = tensor.parameter_indices[begin:end].copy() - column_begin
    indices[~mask] = 0
    return (
        _OrderTensor(
            tensor.order,
            indices,
            mask.copy(),
            tensor.representative_from_pivots[begin:end],
            tensor.rotations[begin:end],
            tensor.component_permutations[begin:end],
            tensor.coordinates[begin:end],
            tensor.image_mask[begin:end],
        ),
        column_begin,
        column_end,
    )


def _force_design_batch(displacements, covariance, tensors, image_bases, n_parameters):
    """Construct exact force-design rows directly from the linear FC basis."""

    def one_structure(displacement):
        design = jnp.zeros((displacement.size, n_parameters), dtype=jnp.float64)
        for tensor, image_basis in zip(tensors, image_bases, strict=True):
            order = tensor.order
            coordinates = jnp.asarray(tensor.coordinates)
            parameter_indices = jnp.asarray(tensor.parameter_indices)
            coefficient_mask = (
                jnp.asarray(tensor.image_mask)[:, :, None, None, None]
                * jnp.asarray(tensor.parameter_mask)[:, None, None, None, :]
            )
            basis = jnp.asarray(image_basis)[:, :, None, :, :]
            for axis in range(order):
                remaining = np.delete(np.arange(order), axis)
                lower = _wick(
                    displacement, covariance, coordinates[..., remaining], order - 1
                )
                contribution = (
                    -lower[..., None] * basis * coefficient_mask / factorial(order)
                )
                force_coordinates = coordinates[..., axis, None]
                parameter_coordinates = parameter_indices[:, None, None, None, :]
                design = design.at[
                    force_coordinates, parameter_coordinates
                ].add(contribution)
        return design

    return jax.vmap(one_structure)(displacements)


def _predict_force(parameters, displacements, covariance, tensors):
    def one_structure(displacement):
        force = jnp.zeros(displacement.size, dtype=jnp.float64)
        for tensor in tensors:
            order = tensor.order
            indices = jnp.asarray(tensor.parameter_indices)
            local_parameters = parameters[indices] * jnp.asarray(tensor.parameter_mask)
            representative = jnp.einsum(
                "ocd,od->oc", jnp.asarray(tensor.representative_from_pivots), local_parameters
            ).reshape((-1,) + (3,) * order)
            rotated = _rotate_images(
                representative, jnp.asarray(tensor.rotations), order
            )
            image_tensors = jnp.take_along_axis(
                rotated, jnp.asarray(tensor.component_permutations), axis=2
            )
            coordinates = jnp.asarray(tensor.coordinates)
            mask = jnp.asarray(tensor.image_mask)
            for axis in range(order):
                remaining = np.delete(np.arange(order), axis)
                lower = _wick(
                    displacement, covariance, coordinates[..., remaining], order - 1
                )
                contribution = (
                    -lower
                    * image_tensors[:, :, None, :]
                    * mask[:, :, None, None]
                    / factorial(order)
                )
                force = force.at[coordinates[..., axis].reshape(-1)].add(
                    contribution.reshape(-1)
                )
        return force.reshape(displacement.shape)

    return jax.vmap(one_structure)(displacements)


def _wick(displacement, covariance, coordinates, order):
    flattened = displacement.reshape(-1)
    values = flattened[coordinates]
    if order == 0:
        return jnp.ones(coordinates.shape[:-1], dtype=displacement.dtype)
    if order == 1:
        return values[..., 0]
    first = coordinates[..., 0]
    result = values[..., 0] * _wick(
        displacement, covariance, coordinates[..., 1:], order - 1
    )
    for partner in range(1, order):
        remaining = np.delete(np.arange(order), (0, partner))
        result -= covariance[first, coordinates[..., partner]] * _wick(
            displacement, covariance, coordinates[..., remaining], order - 2
        )
    return result


def _rotate(tensor, rotation, order):
    result = tensor
    for axis in range(order):
        result = jnp.tensordot(rotation, result, axes=((1,), (axis,)))
        result = jnp.moveaxis(result, 0, axis)
    return result


def _rotate_images(tensors, rotations, order):
    def one_tensor(tensor, operations):
        return jax.vmap(
            lambda operation: _rotate(tensor, operation, order).reshape(-1)
        )(operations)

    return jax.vmap(one_tensor)(tensors, rotations)


def _symmetrized_covariance(displacements, calculation):
    flattened = displacements.reshape(len(displacements), -1)
    covariance = flattened.T @ flattened / len(flattened)
    covariance = covariance.reshape(len(calculation.supercell), 3, len(calculation.supercell), 3)
    result = np.zeros_like(covariance)
    count = 0
    translations = np.unique(calculation.index.translations, axis=0)
    for shift in translations:
        translated = np.asarray(
            [calculation.index.translate_atom(atom, shift) for atom in range(len(calculation.supercell))]
        )
        translation_inverse = np.argsort(translated)
        translated_covariance = covariance[translation_inverse][:, :, translation_inverse, :]
        for permutation, rotation in zip(
            calculation.symmetry.atom_permutations,
            calculation.symmetry.cartesian_rotations,
            strict=True,
        ):
            rotated = np.einsum(
                "ag,igjd,bd->iajb", rotation, translated_covariance, rotation, optimize=True
            )
            inverse = np.argsort(permutation)
            result += rotated[inverse][:, :, inverse, :]
            count += 1
    result = result.reshape(flattened.shape[1], flattened.shape[1]) / count
    return (result + result.T) * 0.5


def _expand_sparse(parameters, calculations, n_primitive, n_supercell):
    result = {}
    offset = 0
    for calculation in calculations:
        clusters = []
        tensors = []
        for orbit in calculation.orbit_space.orbits:
            values = parameters[offset : offset + orbit.dimension]
            offset += orbit.dimension
            representative = orbit.basis @ np.linalg.solve(orbit.basis[orbit.pivots], values)
            for image in orbit.images:
                clusters.append(image.cluster)
                tensor = representative.reshape((3,) * calculation.config.order)
                for axis in range(calculation.config.order):
                    tensor = np.tensordot(
                        image.action.rotation, tensor, axes=((1,), (axis,))
                    )
                    tensor = np.moveaxis(tensor, 0, axis)
                tensors.append(np.transpose(tensor, image.action.permutation))
        result[calculation.config.order] = SparseOrderForceConstants(
            calculation.config.order,
            n_primitive,
            n_supercell,
            np.asarray(clusters, dtype=np.int32),
            np.asarray(tensors),
        )
    return result


def _wick_to_taylor_sparse(force_constants, covariance):
    """Convert centered multivariate Wick coefficients to ordinary Taylor IFCs.

    Contracting an order-(m+2k) Wick tensor with k covariance pairs contributes
    ``(-1)**k / (2**k * k!)`` to the ordinary order-m Taylor tensor.  The sparse
    cluster representation lets us perform these contractions without ever
    materializing a high-order full-supercell array.
    """
    wick = {
        order: SparseOrderForceConstants(
            values.order,
            values.n_primitive,
            values.n_supercell,
            values.clusters.copy(),
            values.tensors.copy(),
        )
        for order, values in force_constants.items()
    }
    result = {
        order: SparseOrderForceConstants(
            values.order,
            values.n_primitive,
            values.n_supercell,
            values.clusters.copy(),
            values.tensors.copy(),
        )
        for order, values in wick.items()
    }
    covariance = np.asarray(covariance).reshape(
        next(iter(result.values())).n_supercell,
        3,
        next(iter(result.values())).n_supercell,
        3,
    )
    maximum_order = max(result, default=0)
    for target_order in sorted(result):
        tensors_by_cluster = {
            tuple(map(int, cluster)): tensor.copy()
            for cluster, tensor in zip(
                result[target_order].clusters, result[target_order].tensors, strict=True
            )
        }
        for source_order in range(target_order + 2, maximum_order + 1, 2):
            if source_order not in wick:
                continue
            pairs = (source_order - target_order) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            source = wick[source_order]
            for cluster, tensor in zip(source.clusters, source.tensors, strict=True):
                contracted = tensor
                for pair in range(pairs):
                    left = target_order + 2 * pair
                    atom_left, atom_right = int(cluster[left]), int(cluster[left + 1])
                    contracted = np.einsum(
                        "...ab,ab->...",
                        contracted,
                        covariance[atom_left, :, atom_right, :],
                        optimize=True,
                    )
                key = tuple(map(int, cluster[:target_order]))
                tensors_by_cluster[key] = tensors_by_cluster.get(
                    key, np.zeros((3,) * target_order)
                ) + coefficient * contracted
        # Dict insertion order preserves the existing reconstruction/export
        # contract. Contraction-only clusters are appended deterministically as
        # source orders are traversed; unaffected orders remain byte-stable.
        clusters = np.asarray(tuple(tensors_by_cluster), dtype=np.int32)
        tensors = np.asarray([tensors_by_cluster[tuple(cluster)] for cluster in clusters])
        original = result[target_order]
        result[target_order] = SparseOrderForceConstants(
            target_order,
            original.n_primitive,
            original.n_supercell,
            clusters,
            tensors,
        )
    return result
