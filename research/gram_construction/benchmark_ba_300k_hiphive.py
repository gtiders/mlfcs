#!/usr/bin/env python3
"""Serial FC2+FC3+FC4 design and Gram accumulation benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from ase.io import read
from mlfcs import ForceConstantFitter
from mlfcs.fitting.constraints import build_joint_constraints
from mlfcs.fitting.dataset import FitDataset
from mlfcs.fitting.design_operator import ForceDesignOperator
from mlfcs.fitting.gram import GramBuilder
from mlfcs.fitting.linear_solvers import explicit_constraint_null_space
from mlfcs.fitting.taylor.features import taylor_axis_derivatives
ROOT = Path(__file__).resolve().parents[2]
PRIMITIVE = ROOT / "tutorial/Ba8Ga16Ge30/T300K/primitive.vasp"
REFERENCE = ROOT / "tutorial/Ba8Ga16Ge30/T300K/supercell.vasp"
SNAPSHOTS = ROOT / "tutorial/Ba8Ga16Ge30/T300K/nve.extxyz"
CUTOFFS = {2: 4.874560779052022, 3: 4.049670647733018, 4: 3.738180056175719}
def build_mlfcs():
    primitive, reference = read(PRIMITIVE), read(REFERENCE); snapshots = read(SNAPSHOTS, index=":")
    fitter = ForceConstantFitter(primitive, reference, orders=(2, 3, 4), cutoffs=CUTOFFS, max_body_orders={2: None, 3: None, 4: None}, symprec=1e-4)
    dataset = FitDataset.from_atoms(fitter.geometry, snapshots); constraints = explicit_constraint_null_space(build_joint_constraints(fitter.calculations, acoustic=True).matrix)
    operator = ForceDesignOperator(dataset.displacements, np.empty(0), fitter.order_tensors, fitter.n_parameters, 4, parameter_map=constraints, axis_derivatives=taylor_axis_derivatives)
    return operator, dataset.forces.reshape(-1), fitter.n_parameters, constraints.shape[1]
def run_mlfcs():
    operator, target, physical, reduced = build_mlfcs(); import jax
    batch = jax.device_put(operator.displacements[:4], operator.program.device)
    for group in operator.program.groups: group.kernel(batch, operator.basis_state, *group.device_arguments).block_until_ready()
    started = time.perf_counter(); stats = GramBuilder.from_operator(operator, target, batch_size=4); elapsed = time.perf_counter() - started
    return {"backend": "mlfcs", "frames": len(operator.displacements), "batch_size": 4, "physical_parameters": physical, "asr_parameters": reduced, "initialization_excluded": True, "accumulation_seconds": elapsed, "gram_shape": list(stats.gram.shape)}
def run_hiphive():
    from hiphive import ClusterSpace
    from hiphive.cutoffs import CutoffMaximumBody
    from hiphive.force_constant_model import ForceConstantModel
    primitive, reference = read(PRIMITIVE), read(REFERENCE); snapshots = read(SNAPSHOTS, index=":")
    cs = ClusterSpace(primitive, CutoffMaximumBody([CUTOFFS[2], CUTOFFS[3], CUTOFFS[4]], 4), symprec=1e-4); model = ForceConstantModel(reference, cs)
    fitter = ForceConstantFitter(primitive, reference, orders=(2, 3, 4), cutoffs=CUTOFFS, max_body_orders={2: None, 3: None, 4: None}, symprec=1e-4); dataset = FitDataset.from_atoms(fitter.geometry, snapshots)
    cvs = cs._cvs.toarray(); gram = np.zeros((cs.n_dofs, cs.n_dofs)); rhs = np.zeros(cs.n_dofs); started = time.perf_counter()
    for displacement, force in zip(dataset.displacements, dataset.forces, strict=True):
        matrix = model.get_fit_matrix(displacement); target = force.reshape(-1); gram += matrix.T @ matrix; rhs += matrix.T @ target
    elapsed = time.perf_counter() - started
    return {"backend": "hiphive", "frames": len(snapshots), "batch_size": 1, "physical_parameters": int(cs._cvs.shape[0]), "asr_parameters": int(cs.n_dofs), "initialization_excluded": True, "accumulation_seconds": elapsed, "gram_shape": list(gram.shape)}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--backend", choices=("mlfcs", "hiphive"), required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); result = run_mlfcs() if args.backend == "mlfcs" else run_hiphive(); print(json.dumps(result, indent=2, sort_keys=True)); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
if __name__ == "__main__": main()
