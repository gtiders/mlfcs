"""Independent HDF5 polynomial-potential and ASE Calculator prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import h5py
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mlfcs.fitting import ForceConstantFitter
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.backends.wick.lowering import (
    build_wick_to_taylor_transform,
    lowered_fc1,
)
from mlfcs.fitting.design_operator import ForceDesignOperator
from mlfcs.force_constants.expansion import expand_fitted_orders
from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.io.hdf5 import read_hdf5, write_hdf5
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.supercell import build_supercell

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).with_name("results.json")


def _predict_with_design_operator(parameters, displacements, covariance, parameterizations):
    operator = ForceDesignOperator(
        displacements,
        covariance,
        parameterizations,
        len(parameters),
        batch_size=max(1, len(displacements)),
        axis_derivatives=wick_axis_derivatives,
    )
    return operator.matvec(parameters).reshape(np.asarray(displacements).shape)


@dataclass(slots=True)
class PolynomialPotential:
    """Reference-relative Taylor polynomial realized in one finite cell."""

    force_constants: ForceConstants
    relation: StructureRelation
    fc1: np.ndarray | None = None

    @classmethod
    def from_force_constants(
        cls,
        force_constants: ForceConstants,
        reference: Atoms | None = None,
        fc1: np.ndarray | None = None,
    ) -> PolynomialPotential:
        primitive = force_constants.relation.primitive
        target = primitive if reference is None else reference
        realized = realize_force_constants(force_constants, target)
        if fc1 is not None and np.asarray(fc1).shape != (len(primitive), 3):
            raise ValueError("FC1 must have shape (n_primitive, 3)")
        return cls(realized, realized.relation, None if fc1 is None else np.asarray(fc1))

    def evaluate_displacement(self, displacement: np.ndarray) -> tuple[float, np.ndarray, dict]:
        u = np.asarray(displacement, dtype=float)
        if u.shape != (len(self.relation.reference), 3):
            raise ValueError("displacement shape differs from the realized reference")
        energy = 0.0
        forces = np.zeros_like(u)
        by_order = {}
        if self.fc1 is not None:
            atom_fc1 = self.fc1[self.relation.primitive_index]
            order_energy = float(np.einsum("ia,ia->", atom_fc1, u))
            order_force = -atom_fc1
            energy += order_energy
            forces += order_force
            by_order[1] = {"energy": order_energy, "forces": order_force.copy()}
        cells = self.relation.index.cell_representatives
        for order, sparse in sorted(self.force_constants.sparse.items()):
            order_energy = 0.0
            order_force = np.zeros_like(u)
            for sites, translations, tensor in zip(
                sparse.sites, sparse.translations, sparse.tensors, strict=True
            ):
                labels = np.vstack((np.zeros((1, 3), dtype=np.int32), translations))
                for cell in cells:
                    atoms = np.asarray(
                        [
                            self.relation.index.atom(int(site), cell + translation)
                            for site, translation in zip(sites, labels, strict=True)
                        ],
                        dtype=np.int32,
                    )
                    operands = [tensor]
                    operands.append(list(range(order)))
                    for axis, atom in enumerate(atoms):
                        operands.extend([u[atom], [axis]])
                    operands.append([])
                    value = float(np.einsum(*operands, optimize=True)) / factorial(order)
                    order_energy += value
                    for axis, atom in enumerate(atoms):
                        keep = [value for value in range(order) if value != axis]
                        derivative_operands = [tensor, list(range(order))]
                        for tensor_axis in keep:
                            derivative_operands.extend([u[atoms[tensor_axis]], [tensor_axis]])
                        derivative_operands.append([axis])
                        gradient = np.einsum(*derivative_operands, optimize=True)
                        order_force[atom] -= gradient / factorial(order)
            energy += order_energy
            forces += order_force
            by_order[order] = {"energy": order_energy, "forces": order_force}
        return energy, forces, by_order

    def evaluate_atoms(self, atoms: Atoms) -> tuple[float, np.ndarray, dict]:
        return self.evaluate_displacement(self.relation.displacement(atoms))


class PrototypeMLFCSCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, potential: PolynomialPotential):
        super().__init__()
        self.potential = potential

    @classmethod
    def from_hdf5(cls, source: str | Path, *, reference: Atoms | None = None):
        return cls(PolynomialPotential.from_force_constants(read_hdf5(source), reference))

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces, _ = self.potential.evaluate_atoms(atoms)
        self.results = {"energy": energy, "forces": forces}


def _write_candidate_hdf5(path: Path, force_constants: ForceConstants, fc1: np.ndarray) -> None:
    write_hdf5(path, force_constants)
    with h5py.File(path, "a") as handle:
        terms = handle.create_group("reference_terms")
        entry = terms.create_dataset("fc1", data=np.asarray(fc1))
        entry.attrs["unit"] = "eV/angstrom"
        entry.attrs["basis"] = "cartesian"
        entry.attrs["semantics"] = "primitive-site Taylor energy gradient"


def _read_candidate_hdf5(path: Path) -> tuple[ForceConstants, np.ndarray]:
    force_constants = read_hdf5(path)
    with h5py.File(path, "r") as handle:
        fc1 = np.asarray(handle["reference_terms/fc1"], dtype=float)
    return force_constants, fc1


def _norm_metrics(left, right):
    delta = np.asarray(left) - np.asarray(right)
    return {
        "maximum_absolute": float(np.max(np.abs(delta), initial=0.0)),
        "relative": float(np.linalg.norm(delta) / max(np.linalg.norm(right), 1e-300)),
    }


def wick_equivalence() -> dict:
    primitive = Atoms("Ar", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = build_supercell(primitive, (3, 3, 3))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: 4.1, 3: 4.1, 4: 4.1},
        max_body_orders={2: 2, 3: 2, 4: 2},
    )
    rng = np.random.default_rng(8147)
    parameters = rng.normal(scale=0.2, size=fitter.n_parameters)
    covariance = np.eye(3 * len(reference)) * 0.003
    displacements = rng.normal(scale=0.025, size=(7, len(reference), 3))
    wick_force = np.asarray(
        _predict_with_design_operator(
            jnp.asarray(parameters),
            jnp.asarray(displacements),
            jnp.asarray(covariance),
            fitter.order_tensors,
        )
    )
    transform = build_wick_to_taylor_transform(fitter.calculations, covariance)
    taylor_parameters = np.asarray(transform @ parameters)
    taylor_design_force = np.asarray(
        _predict_with_design_operator(
            jnp.asarray(taylor_parameters),
            jnp.asarray(displacements),
            jnp.zeros_like(jnp.asarray(covariance)),
            fitter.order_tensors,
        )
    )
    sparse = expand_fitted_orders(taylor_parameters, fitter.calculations)
    base = ForceConstants(
        {}, reference.copy(), metadata={"force_constants_basis": "taylor"},
        sparse=sparse, relation=fitter.geometry
    )
    fc1 = lowered_fc1(fitter.calculations, parameters, covariance)
    without = PolynomialPotential.from_force_constants(base, reference)
    with_fc1 = PolynomialPotential.from_force_constants(base, reference, fc1)
    force_without = np.asarray([without.evaluate_displacement(u)[1] for u in displacements])
    force_complete = np.asarray([with_fc1.evaluate_displacement(u)[1] for u in displacements])
    target_larger = build_supercell(primitive, (4, 3, 3))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "with-fc1.h5"
        _write_candidate_hdf5(path, base, fc1)
        restored, restored_fc1 = _read_candidate_hdf5(path)
        reload_reference = PolynomialPotential.from_force_constants(
            restored, reference, restored_fc1
        )
        reload_force = np.asarray(
            [reload_reference.evaluate_displacement(u)[1] for u in displacements]
        )
        larger = PolynomialPotential.from_force_constants(restored, target_larger, restored_fc1)
        larger_u = rng.normal(scale=0.02, size=(len(target_larger), 3))
        larger_before = PolynomialPotential.from_force_constants(base, target_larger, fc1)
        larger_reload = larger.evaluate_displacement(larger_u)[1]
        larger_direct = larger_before.evaluate_displacement(larger_u)[1]
        orders_after_reload = list(restored.orders)
    order_contributions = with_fc1.evaluate_displacement(displacements[0])[2]
    return {
        "n_parameters": fitter.n_parameters,
        "fc1_shape": list(fc1.shape),
        "fc1_maximum": float(np.max(np.abs(fc1))),
        "formal_taylor_design_vs_sparse_evaluator": _norm_metrics(
            force_without, taylor_design_force
        ),
        "wick_vs_current_intertwiner_taylor": _norm_metrics(taylor_design_force, wick_force),
        "wick_vs_taylor_with_fc1": _norm_metrics(force_complete, wick_force),
        "hdf5_reload_with_fc1": _norm_metrics(reload_force, force_complete),
        "larger_supercell_reload": _norm_metrics(larger_reload, larger_direct),
        "orders_after_reload": orders_after_reload,
        "order_force_norms": {
            str(order): float(np.linalg.norm(values["forces"]))
            for order, values in order_contributions.items()
        },
    }


def folded_covariance_counterexample() -> dict:
    primitive = Atoms("Ar", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    reference = build_supercell(primitive, (4, 1, 1))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: 4.1, 3: 4.1, 4: 4.1},
        max_body_orders={2: 2, 3: 2, 4: 2},
    )
    from mlfcs.fitting.constraints import build_joint_constraints
    from mlfcs.fitting.linear_solvers import explicit_constraint_null_space

    constraints = build_joint_constraints(fitter.calculations, acoustic=True).matrix
    parameter_map = explicit_constraint_null_space(constraints).toarray()
    rng = np.random.default_rng(9)
    parameters = parameter_map @ rng.normal(scale=0.2, size=parameter_map.shape[1])
    covariance = np.eye(3 * len(reference)) * 0.003
    displacement = rng.normal(scale=0.025, size=(10, len(reference), 3))
    transform = build_wick_to_taylor_transform(fitter.calculations, covariance)
    wick_force = np.asarray(
        _predict_with_design_operator(
            jnp.asarray(parameters), jnp.asarray(displacement),
            jnp.asarray(covariance), fitter.order_tensors
        )
    )
    taylor_force = np.asarray(
        _predict_with_design_operator(
            jnp.asarray(transform @ parameters), jnp.asarray(displacement),
            jnp.zeros_like(jnp.asarray(covariance)), fitter.order_tensors
        )
    )
    taylor_force = taylor_force - lowered_fc1(
        fitter.calculations, parameters, covariance
    )[None, :, :]
    return {
        "reference_matrix": [4, 1, 1],
        "cutoff_angstrom": 4.1,
        "maximum_absolute_force_difference": float(np.max(np.abs(wick_force - taylor_force))),
        "relative_force_difference": float(
            np.linalg.norm(wick_force - taylor_force) / np.linalg.norm(wick_force)
        ),
        "interpretation": (
            "R=0 and R=+/-1 fold along the one-cell y/z directions; finite covariance "
            "contraction contains a source-only harmonic response outside transferable FC2"
        ),
    }


def fc1_equivalence() -> dict:
    primitive = Atoms(
        "GaAs",
        cell=np.eye(3) * 5.6,
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        pbc=True,
    )
    fitter = ForceConstantFitter(
        primitive,
        primitive.copy(),
        orders=(3,),
        cutoffs={3: 3.0},
    )
    rng = np.random.default_rng(4)
    parameters = rng.normal(scale=0.2, size=fitter.n_parameters)
    covariance = np.eye(6) * 0.04
    displacement = rng.normal(scale=0.03, size=(9, 2, 3))
    wick_force = np.asarray(
        _predict_with_design_operator(
            jnp.asarray(parameters),
            jnp.asarray(displacement),
            jnp.asarray(covariance),
            fitter.order_tensors,
        )
    )
    fc1 = lowered_fc1(fitter.calculations, parameters, covariance)
    sparse = expand_fitted_orders(parameters, fitter.calculations)
    base = ForceConstants(
        {}, primitive.copy(), {"force_constants_basis": "taylor"}, sparse, fitter.geometry
    )
    without = PolynomialPotential.from_force_constants(base, primitive)
    complete = PolynomialPotential.from_force_constants(base, primitive, fc1)
    force_without = np.asarray([without.evaluate_displacement(u)[1] for u in displacement])
    force_complete = np.asarray([complete.evaluate_displacement(u)[1] for u in displacement])
    expected = np.repeat((-fc1)[None, :, :], len(displacement), axis=0)
    larger_reference = build_supercell(primitive, (2, 1, 1))
    larger_u = rng.normal(scale=0.02, size=(len(larger_reference), 3))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "fc1-candidate.h5"
        _write_candidate_hdf5(path, base, fc1)
        restored, restored_fc1 = _read_candidate_hdf5(path)
        reload_force = PolynomialPotential.from_force_constants(
            restored, primitive, restored_fc1
        ).evaluate_displacement(displacement[0])[1]
        larger_before = PolynomialPotential.from_force_constants(
            base, larger_reference, fc1
        ).evaluate_displacement(larger_u)[1]
        larger_after = PolynomialPotential.from_force_constants(
            restored, larger_reference, restored_fc1
        ).evaluate_displacement(larger_u)[1]
    return {
        "fc1_shape": list(fc1.shape),
        "fc1_maximum": float(np.max(np.abs(fc1))),
        "wick_minus_no_fc1_equals_minus_fc1": _norm_metrics(
            wick_force - force_without, expected
        ),
        "wick_vs_complete_taylor": _norm_metrics(force_complete, wick_force),
        "candidate_hdf5_reload": _norm_metrics(reload_force, force_complete[0]),
        "candidate_hdf5_larger_supercell": _norm_metrics(larger_after, larger_before),
    }


def existing_hdf5_cases() -> dict:
    rng = np.random.default_rng(22)
    paths = {
        "fc2": ROOT / "examples/fitting/Si/harmonic/results/mlfcs.h5",
        "fc234_wick_fit": ROOT / "examples/fitting/Si/anharmonic/results/mlfcs.h5",
        "fc23_finite_difference": ROOT / "examples/finite-difference/K4As4Pt2/results/three-phonon/mlfcs.h5",
    }
    output = {}
    for name, path in paths.items():
        fc = read_hdf5(path)
        variants = {name: fc}
        if name == "fc234_wick_fit":
            variants["fc23_wick_fit"] = ForceConstants(
                {}, fc.supercell.copy(), dict(fc.metadata),
                {order: fc.sparse[order] for order in (2, 3)}, fc.relation
            )
        for variant_name, variant in variants.items():
            potential = PolynomialPotential.from_force_constants(variant)
            displacement = rng.normal(scale=1e-3, size=(len(potential.relation.reference), 3))
            energy, force, by_order = potential.evaluate_displacement(displacement)
            atoms = potential.relation.reference.copy()
            atoms.positions += displacement
            atoms.calc = PrototypeMLFCSCalculator(potential)
            step = 1e-6
            plus = displacement.copy()
            minus = displacement.copy()
            plus[0, 0] += step
            minus[0, 0] -= step
            numeric_force = -(
                potential.evaluate_displacement(plus)[0]
                - potential.evaluate_displacement(minus)[0]
            ) / (2 * step)
            output[variant_name] = {
                "orders": list(variant.orders),
                "energy_eV": energy,
                "force_norm": float(np.linalg.norm(force)),
                "ase_energy_difference": abs(atoms.get_potential_energy() - energy),
                "ase_force": _norm_metrics(atoms.get_forces(), force),
                "energy_gradient_force_x_atom_0_absolute_difference": abs(
                    numeric_force - force[0, 0]
                ),
                "order_energy": {str(k): float(v["energy"]) for k, v in by_order.items()},
            }
    return output


def main() -> None:
    result = {
        "conclusion": "SMALL REFACTOR plus an optional FC1 schema extension; add a Wick-contraction folding guard",
        "wick_equivalence": wick_equivalence(),
        "folded_covariance_counterexample": folded_covariance_counterexample(),
        "fc1_equivalence": fc1_equivalence(),
        "existing_hdf5_cases": existing_hdf5_cases(),
        "formal_source_modified": False,
    }
    RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
