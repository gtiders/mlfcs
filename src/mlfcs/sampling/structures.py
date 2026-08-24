from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms

from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.sampling.harmonic import HarmonicSampler, SamplingState

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingBatch:
    structures: tuple[Atoms, ...]
    displacements: np.ndarray
    method: Literal["gaussian", "harmonic"]
    state: SamplingState | None = None
    harmonic_sampler: HarmonicSampler | None = None


def _sample_perturbations(
    reference: Atoms,
    *,
    snapshots: int,
    method: Literal["gaussian", "harmonic"] = "gaussian",
    displacement: float = 0.01,
    force_constants: ForceConstants | None = None,
    temperature: float | None = None,
    statistics: Literal["quantum", "classical"] = "quantum",
    cutoff_frequency: float = 0.01,
    imaginary_modes: Literal["error", "absolute", "exclude"] = "error",
    imaginary_tolerance: float = 1e-6,
    max_displacement: float | None = None,
    random_seed: int | None = None,
) -> SamplingBatch:
    if not isinstance(reference, Atoms):
        raise TypeError("reference must be an ASE Atoms object")
    if snapshots < 1:
        raise ValueError("snapshots must be positive")
    if method not in {"gaussian", "harmonic"}:
        raise ValueError("method must be 'gaussian' or 'harmonic'")

    sampler = None
    state = None
    if method == "gaussian":
        if displacement <= 0:
            raise ValueError("displacement must be positive")
        if force_constants is not None or temperature is not None or max_displacement is not None:
            raise ValueError(
                "force_constants, temperature, and max_displacement require harmonic sampling"
            )
        if statistics != "quantum" or cutoff_frequency != 0.01:
            raise ValueError("statistics and cutoff_frequency require harmonic sampling")
        if imaginary_modes != "error" or imaginary_tolerance != 1e-6:
            raise ValueError("imaginary-mode options require harmonic sampling")
        rng = np.random.default_rng(random_seed)
        values = rng.normal(scale=displacement, size=(snapshots, len(reference), 3))
        values -= values.mean(axis=1, keepdims=True)
    else:
        if force_constants is None:
            raise ValueError("force_constants is required for harmonic sampling")
        if temperature is None:
            raise ValueError("temperature is required for harmonic sampling")
        if displacement != 0.01:
            raise ValueError("displacement does not control harmonic sampling")
        if 2 not in force_constants.orders:
            raise ValueError("force_constants must contain FC2")
        if force_constants.relation is None:
            raise ValueError("harmonic sampling requires force constants with a structure relation")
        primitive = force_constants.relation.primitive
        realized = realize_force_constants(force_constants, reference, primitive=primitive)
        compact = realized.materialize(2, max_bytes=None)
        sampler = HarmonicSampler(
            primitive,
            reference,
            compact,
            temperature=temperature,
            statistics=statistics,
            cutoff_frequency=cutoff_frequency,
            imaginary_modes=imaginary_modes,
            imaginary_tolerance=imaginary_tolerance,
            max_displacement=max_displacement,
        )
        values = sampler.sample(snapshots, random_seed=random_seed)
        state = sampler.state

    structures = []
    for configuration, perturbation in enumerate(values):
        atoms = reference.copy()
        atoms.positions += perturbation
        atoms.info["mlfcs_configuration_id"] = configuration
        atoms.info["mlfcs_sampling_method"] = method
        structures.append(atoms)
    logger.info("Generated %d %s perturbation structures", snapshots, method)
    if state is not None:
        logger.info(
            "Harmonic sampling: qpoints=%d, sampled_modes=%d/%d, minimum_frequency=%.8f THz",
            state.qpoints,
            state.sampled_modes,
            state.total_modes,
            state.minimum_frequency_thz,
        )
        if state.clipped_atoms:
            logger.warning(
                "Clipped %d atomic displacements in %d snapshots at %.6g Å",
                state.clipped_atoms,
                state.affected_snapshots,
                state.maximum_displacement,
            )
    return SamplingBatch(tuple(structures), values, method, state, sampler)


def perturb_structures(
    reference: Atoms,
    *,
    snapshots: int,
    method: Literal["gaussian", "harmonic"] = "gaussian",
    displacement: float = 0.01,
    force_constants: ForceConstants | None = None,
    temperature: float | None = None,
    statistics: Literal["quantum", "classical"] = "quantum",
    cutoff_frequency: float = 0.01,
    imaginary_modes: Literal["error", "absolute", "exclude"] = "error",
    imaginary_tolerance: float = 1e-6,
    max_displacement: float | None = None,
    random_seed: int | None = None,
) -> list[Atoms]:
    """Generate independent Cartesian or harmonic displacement structures."""
    return list(
        _sample_perturbations(
            reference,
            snapshots=snapshots,
            method=method,
            displacement=displacement,
            force_constants=force_constants,
            temperature=temperature,
            statistics=statistics,
            cutoff_frequency=cutoff_frequency,
            imaginary_modes=imaginary_modes,
            imaginary_tolerance=imaginary_tolerance,
            max_displacement=max_displacement,
            random_seed=random_seed,
        ).structures
    )
