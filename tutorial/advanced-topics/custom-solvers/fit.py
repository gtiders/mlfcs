"""Solve the same force fit before and after normal-matrix construction."""

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from scipy.linalg import cho_factor, cho_solve

from mlfcs import ClusterMap, ClusterSpace, FitData, FitSystem, Supercell
from mlfcs.core.log_error import configure

ROOT = Path(__file__).resolve().parent


def fit():
    atoms = Atoms(
        symbols=["Si", "Ge"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.17, 0.31, 0.23]],
        cell=[[3.1, 0.2, 0.1], [0.1, 3.7, 0.3], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    space = ClusterSpace(atoms, cutoffs={2: 0.1}, max_body_orders={2: 1})
    supercell = Supercell.from_atoms(space.primitive, atoms)
    mapping = ClusterMap.build(space, supercell)

    # A small, reproducible training set with stored ASE forces. For real data,
    # replace this loop with structures read from an extxyz or another source.
    stiffness = np.array(
        [
            [[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]],
            [[3.0, 0.0, 0.1], [0.0, 4.0, 0.0], [0.1, 0.0, 5.0]],
        ]
    )
    rng = np.random.default_rng(7)
    structures = []
    for displacement in rng.normal(scale=0.02, size=(24, len(atoms), 3)):
        frame = atoms.copy()
        frame.positions += displacement
        forces = -np.einsum("aij,aj->ai", stiffness, displacement)
        frame.calc = SinglePointCalculator(frame, forces=forces)
        structures.append(frame)

    # Route 1: retain A and f. Scale its columns without forming A.T @ A.
    data = FitData.from_atoms(mapping, structures)
    design, forces = data.arrays()
    column_norm = np.linalg.norm(design, axis=0)
    if np.any(column_norm == 0):
        raise ValueError("the training design contains an unobserved parameter")
    direct_scale = 1.0 / column_norm
    scaled_direct, _, rank, _ = np.linalg.lstsq(design * direct_scale, forces, rcond=None)
    if rank != data.n_parameters:
        raise ValueError("the training design does not determine every parameter")
    direct = direct_scale * scaled_direct
    direct_model = data.force_constants(direct)
    direct_model.save(ROOT / "fc-direct.mlfcs")

    # Route 2: stream G = A.T @ A and b = A.T @ f, then solve G @ parameters = b.
    system = FitSystem.from_atoms(mapping, structures)
    scale = system.column_scale
    scaled_matrix = scale[:, None] * system.matrix * scale[None, :]
    scaled_rhs = scale * system.rhs
    scaled_parameters = cho_solve(cho_factor(scaled_matrix, lower=True), scaled_rhs)
    normal = scale * scaled_parameters
    normal_model = system.force_constants(normal)
    normal_model.save(ROOT / "fc-normal.mlfcs")

    np.testing.assert_allclose(normal, direct, rtol=1e-11, atol=1e-11)
    print(f"training structures: {len(structures)}")
    print(f"design: {design.shape}; normal matrix: {system.matrix.shape}")
    print(
        f"direct least-squares RMSE: {np.sqrt(np.mean((design @ direct - forces) ** 2)):.3e} eV/Å"
    )
    print(
        f"Cholesky normal-system RMSE: {np.sqrt(np.mean((design @ normal - forces) ** 2)):.3e} eV/Å"
    )
    print(f"maximum parameter difference: {np.max(np.abs(normal - direct)):.3e}")
    print("saved fc-direct.mlfcs and fc-normal.mlfcs")


if __name__ == "__main__":
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log:
        configure(stream=log)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                fit()
        except BaseException:
            traceback.print_exc(file=log)
            raise
