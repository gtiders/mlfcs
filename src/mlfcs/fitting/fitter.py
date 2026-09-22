from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from ase import Atoms

from mlfcs.constraints.translational import TranslationalASRProjector
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.gram import GramBuilder, GramStatistics
from mlfcs.fitting.parameterization import pack_order
from mlfcs.fitting.taylor.model import TaylorModel
from mlfcs.force_constants.expansion import expand_fitted_orders
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.interactions.space import InteractionSpace, ReferenceFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FittingResult:
    """One physical-coordinate fit followed by an optional ASR projection."""

    force_constants: ForceConstants
    fitting_parameters: np.ndarray
    unprojected_parameters: np.ndarray
    parameter_scale: np.ndarray
    gram_statistics: GramStatistics
    iterations: int
    training_force_rmse: float
    training_relative_force_error: float
    unprojected_training_force_rmse: float
    unprojected_training_relative_force_error: float
    order_force_rms: dict[int, float]
    stop_code: int
    residual_norm: float
    solver_normal_equation_residual: float
    maximum_asr_residual_before: float
    maximum_asr_residual_after: float
    asr_parameter_correction: float
    asr_projection_iterations: int


class ForceConstantFitter:
    """Jointly fit consecutive IFC orders and optionally project the result onto ASR."""

    def __init__(
        self,
        primitive: Atoms,
        reference: Atoms,
        *,
        orders: tuple[int, ...] = (2, 3),
        cutoffs: dict[int, float | int] | None = None,
        max_body_orders: dict[int, int | None] | None = None,
        symprec: float = 1e-5,
    ):
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
        self._taylor = TaylorModel()
        order_text = "+".join(f"FC{order}" for order in self.orders)
        logger.info("Preparing physical %s fitting parameterization", order_text)
        self.calculations = tuple(
            InteractionSpace.from_frame(
                frame,
                order=order,
                cutoff=self.cutoffs[order],
                max_body_order=self.max_body_orders.get(order),
                symprec=symprec,
            )
            for order in self.orders
        )
        offset = 0
        tensors = []
        counts = []
        for calculation in self.calculations:
            tensor, offset = pack_order(calculation, offset)
            tensors.append(tensor)
            count = sum(orbit.dimension for orbit in calculation.realized_orbit_space.orbits)
            counts.append(count)
            logger.info(
                "- FC%d: %d orbits, %d physical parameters",
                tensor.order,
                len(calculation.realized_orbit_space.orbits),
                count,
            )
        self.order_tensors = tuple(tensors)
        self._order_counts = tuple(counts)
        self.n_parameters = offset
        if self.n_parameters != sum(self._order_counts):
            raise RuntimeError("compiled design and orbit spaces disagree on parameter count")
        self._projectors = tuple(
            TranslationalASRProjector.from_orbit_space(calculation.primitive_orbit_space)
            for calculation in self.calculations
        )
        self.index = self.calculations[0].index
        self.canonical_supercell = self.calculations[0].supercell
        self._design_identity = self._physical_design_identity()
        logger.info("- Joint physical parameter count: %d", self.n_parameters)

    def fit(
        self,
        gram: GramStatistics,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        acoustic_sum_rule: bool = True,
        asr_tolerance: float = 1e-10,
        precondition: bool = True,
        allow_unconverged: bool = False,
    ) -> FittingResult:
        """Solve the physical Gram and then optionally project each IFC order onto ASR."""
        if not isinstance(gram, GramStatistics):
            raise TypeError("fit expects a GramStatistics object")
        gram.require_design(self._design_identity)
        if gram.gram.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(
                f"the Gram system has shape {gram.gram.shape}, expected "
                f"{(self.n_parameters, self.n_parameters)} physical parameters"
            )
        if gram.rhs.shape != (self.n_parameters,):
            raise ValueError(
                f"the Gram right-hand side has shape {gram.rhs.shape}, expected "
                f"{(self.n_parameters,)}"
            )
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not np.isfinite(asr_tolerance) or asr_tolerance <= 0:
            raise ValueError("asr_tolerance must be finite and positive")

        if precondition:
            parameter_scale = gram.exact_column_scale()
            self._report_parameter_scale(parameter_scale)
        else:
            parameter_scale = np.ones(gram.gram.shape[0])
            logger.info("- Parameter preconditioning disabled")
        logger.info("Solving the unconstrained force-only least-squares problem")
        logger.info("- Equations: %d, physical unknowns: %d", gram.n_equations, self.n_parameters)
        solution = gram.solve(
            parameter_scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        scaled_parameters, stop_code, iterations, residual_norm, normal_residual = solution
        if stop_code != 0 and not allow_unconverged:
            raise RuntimeError(
                "force-constant fitting did not converge: "
                f"stop_code={stop_code}, iterations={iterations}, "
                f"normal residual={normal_residual:.6e}; "
                "set allow_unconverged=True only to inspect the incomplete solution"
            )
        unprojected = np.asarray(scaled_parameters) * parameter_scale
        unprojected_metrics = gram.force_metrics(unprojected)

        parameters = unprojected.copy()
        before = 0.0
        after = 0.0
        correction_squared = 0.0
        projection_iterations = 0
        offset = 0
        for order, count, projector in zip(
            self.orders,
            self._order_counts,
            self._projectors,
            strict=True,
        ):
            block = parameters[offset : offset + count]
            initial = projector.maximum_residual(block)
            before = max(before, initial)
            if acoustic_sum_rule:
                projection = projector.project(block, tolerance=asr_tolerance)
                parameters[offset : offset + count] = projection.parameters
                final = projection.final_residual
                correction_squared += projection.correction_norm**2
                projection_iterations += projection.iterations
                logger.info(
                    "- FC%d post-fit ASR drift: %.10e -> %.10e, relative correction %.10e",
                    order,
                    projection.initial_residual,
                    projection.final_residual,
                    projection.relative_correction,
                )
            else:
                final = initial
                logger.info("- FC%d ASR drift: %.10e (projection disabled)", order, initial)
            after = max(after, final)
            offset += count

        training_metrics = gram.force_metrics(parameters)
        order_force_rms = gram.order_force_rms(
            parameters,
            self.orders,
            self._order_counts,
            gram.n_equations,
        )
        logger.info("Force fitting summary")
        logger.info(
            "- Unprojected training relative error: %.6f %%",
            100 * unprojected_metrics[1],
        )
        logger.info("- Training relative error: %.6f %%", 100 * training_metrics[1])
        logger.info("- Training force RMSE: %.10e eV/Å", training_metrics[0])
        for order, rms in order_force_rms.items():
            logger.info("- FC%d force contribution RMS: %.10e eV/Å", order, rms)
        logger.info("- Solver iterations=%d, stop_code=%d", iterations, stop_code)
        if stop_code != 0:
            logger.warning(
                "Returning unconverged fitting solution: stop_code=%d, iterations=%d, "
                "residual=%.6e",
                stop_code,
                iterations,
                normal_residual,
            )

        lowering = self._taylor.lower(None, parameters[: self.n_parameters])
        expansion_started = perf_counter()
        logger.info("Expanding post-processed Taylor parameters into sparse physical IFCs")
        residual_sparse = expand_fitted_orders(lowering.taylor_parameters, self.calculations)
        sparse_values = dict(residual_sparse)
        logger.info(
            "- Expanded %d sparse tensors in %.2f s",
            sum(len(value.tensors) for value in sparse_values.values()),
            perf_counter() - expansion_started,
        )
        correction_norm = float(np.sqrt(correction_squared))
        force_constants = ForceConstants(
            {},
            self.canonical_supercell.copy(),
            metadata={
                "method": "joint_force_fit",
                "solver": "gram",
                "fitted_with": "taylor",
                "force_constants_basis": "taylor",
                "cutoff_angstrom": self.calculations[-1].cutoff,
                "cutoff_angstrom_by_order": {
                    calculation.config.order: calculation.cutoff
                    for calculation in self.calculations
                },
                "acoustic_sum_rule": acoustic_sum_rule,
                "asr_projection_tolerance": asr_tolerance if acoustic_sum_rule else None,
                "maximum_asr_residual_before": before,
                "maximum_asr_residual_after": after,
                "asr_parameter_correction": correction_norm,
                "training_equations": gram.n_equations,
            },
            sparse=sparse_values,
            relation=self.geometry,
        )
        return FittingResult(
            force_constants=force_constants,
            fitting_parameters=parameters,
            unprojected_parameters=unprojected,
            parameter_scale=parameter_scale,
            gram_statistics=gram,
            iterations=int(iterations),
            training_force_rmse=training_metrics[0],
            training_relative_force_error=training_metrics[1],
            unprojected_training_force_rmse=unprojected_metrics[0],
            unprojected_training_relative_force_error=unprojected_metrics[1],
            order_force_rms=order_force_rms,
            stop_code=int(stop_code),
            residual_norm=float(residual_norm),
            solver_normal_equation_residual=float(normal_residual),
            maximum_asr_residual_before=before,
            maximum_asr_residual_after=after,
            asr_parameter_correction=correction_norm,
            asr_projection_iterations=projection_iterations,
        )

    def prepare_gram(
        self,
        structures: list[Atoms] | tuple[Atoms, ...],
    ) -> GramStatistics:
        """Build one reusable Gram system in the complete physical parameter space."""
        dataset = FitDataset.from_atoms(self.geometry, structures)
        prepared = self._taylor.prepare(
            calculations=self.calculations,
            training_displacements=dataset.displacements,
            parameterizations=self.order_tensors,
        )
        return GramBuilder.from_operator(
            prepared.operator,
            dataset.forces.reshape(-1),
            metadata=self._design_identity,
        )

    def _physical_design_identity(self) -> dict[str, object]:
        def structure_document(atoms: Atoms) -> dict[str, object]:
            return {
                "numbers": [int(value) for value in atoms.numbers],
                "cell": [
                    [float(value).hex() for value in row]
                    for row in np.asarray(atoms.cell, dtype=float)
                ],
                "scaled_positions": [
                    [float(value).hex() for value in row]
                    for row in atoms.get_scaled_positions(wrap=True)
                ],
            }

        orbit_documents = []
        for calculation in self.calculations:
            entries = []
            for orbit in calculation.primitive_orbit_space.orbits:
                representative = orbit.representative
                entries.append(
                    {
                        "sites": [int(value) for value in representative.sites],
                        "translations": [
                            [int(value) for value in row] for row in representative.translations
                        ],
                        "dimension": int(orbit.dimension),
                    }
                )
            orbit_documents.append(entries)
        document = {
            "schema": 1,
            "primitive": structure_document(self.primitive),
            "reference": structure_document(self.reference),
            "orders": list(self.orders),
            "cutoffs": {str(order): float(self.cutoffs[order]).hex() for order in self.orders},
            "max_body_orders": {
                str(order): self.max_body_orders.get(order) for order in self.orders
            },
            "symprec": float(self.symprec).hex(),
            "orbits": orbit_documents,
        }
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return {
            "design_schema": 1,
            "design_fingerprint": sha256(encoded).hexdigest(),
            "physical_parameter_count": self.n_parameters,
            "orders": list(self.orders),
        }

    def _report_parameter_scale(self, parameter_scale):
        logger.info("Column-norm preconditioning in physical coordinates")
        offset = 0
        for calculation, count in zip(
            self.calculations,
            self._order_counts,
            strict=True,
        ):
            values = parameter_scale[offset : offset + count]
            active = values[values > 0]
            if len(active):
                logger.info(
                    "- FC%d inverse column scale: %.6e to %.6e",
                    calculation.config.order,
                    np.min(active),
                    np.max(active),
                )
            else:
                logger.info(
                    "- FC%d inverse column scale: no active columns",
                    calculation.config.order,
                )
            offset += count
