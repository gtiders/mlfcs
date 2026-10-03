"""Prepared primitive array references for numerical consumers."""

from dataclasses import dataclass

import numpy as np

from mlfcs._arrays import integer_array, readonly


@dataclass(frozen=True, slots=True)
class PreparedClusterSpace:
    fingerprint: str
    positions: np.ndarray
    cell: np.ndarray
    rotations: np.ndarray
    cartesian_rotations: np.ndarray
    site_permutations: np.ndarray
    site_shifts: np.ndarray
    parameter_offsets: np.ndarray


def prepare_cluster_space(space):
    return PreparedClusterSpace(
        space.fingerprint,
        readonly(space.scaled_positions, np.float64),
        readonly(space.cell, np.float64),
        integer_array(space.symmetry.rotations),
        readonly(space.symmetry.cartesian_rotations, np.float64),
        integer_array(space.symmetry.site_permutations),
        integer_array(space.symmetry.site_shifts),
        integer_array(space.parameter_offsets),
    )
