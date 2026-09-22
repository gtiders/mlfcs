"""Independent Cartesian Gaussian perturbations for explicit reference structures."""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms


def gaussian_displacements(
    reference: Atoms,
    *,
    snapshots: int,
    displacement: float = 0.01,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return recentered Gaussian Cartesian displacements in angstrom."""
    if not isinstance(reference, Atoms):
        raise TypeError("reference must be an ASE Atoms object")
    if snapshots < 1:
        raise ValueError("snapshots must be positive")
    if displacement <= 0:
        raise ValueError("displacement must be positive")
    rng = np.random.default_rng(random_seed)
    values = rng.normal(scale=displacement, size=(snapshots, len(reference), 3))
    values -= values.mean(axis=1, keepdims=True)
    return values


def perturb_structures(
    reference: Atoms,
    *,
    snapshots: int,
    method: Literal["gaussian"] = "gaussian",
    displacement: float = 0.01,
    random_seed: int | None = None,
) -> list[Atoms]:
    """Generate independent Cartesian Gaussian displacement structures."""
    if method != "gaussian":
        raise ValueError(
            "this tool only generates Cartesian Gaussian displacements; harmonic sampling "
            "lives in mlfcs.reciprocal.perturb_structures"
        )
    values = gaussian_displacements(
        reference,
        snapshots=snapshots,
        displacement=displacement,
        random_seed=random_seed,
    )
    structures = []
    for configuration, perturbation in enumerate(values):
        atoms = reference.copy()
        atoms.positions += perturbation
        atoms.info["mlfcs_configuration_id"] = configuration
        atoms.info["mlfcs_sampling_method"] = "gaussian"
        structures.append(atoms)
    return structures


__all__ = ["gaussian_displacements", "perturb_structures"]
