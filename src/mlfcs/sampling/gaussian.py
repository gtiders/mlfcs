"""Independent Cartesian Gaussian perturbation of a reference structure.

This is a plain displacement generator: every Cartesian component of every atom draws an
independent normal deviate of the requested width, and each snapshot is recentered so that
its arithmetic mean displacement is zero.  It knows nothing about reciprocal space, which
is why it lives outside :mod:`mlfcs.reciprocal` instead of inside the harmonic sampler.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

__all__ = ["gaussian_displacements", "perturb_structures"]


def gaussian_displacements(
    reference: Atoms,
    *,
    snapshots: int,
    displacement: float = 0.01,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return ``(snapshots, atoms, 3)`` recentered Gaussian displacements in angstrom."""
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
    """Generate independent Cartesian Gaussian displacement structures.

    Harmonic sampling moved to :mod:`mlfcs.reciprocal`, so this entry point only produces
    Cartesian Gaussian snapshots.  Asking it for ``method="harmonic"`` fails with the
    module that owns that workflow instead of silently falling back.
    """
    if method != "gaussian":
        raise ValueError(
            "this entry point only generates Cartesian Gaussian displacements; "
            "harmonic sampling lives in mlfcs.reciprocal.perturb_structures"
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
