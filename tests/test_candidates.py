"""Compare candidate ordering and tensor actions against original operations."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from oracles.candidates_reference import iter_candidates as reference_candidates

from mlfcs import ClusterSpace
from mlfcs.cluster_space import Cluster
from mlfcs.cluster_space.candidates import _candidate_labels
from mlfcs.core.tensors import _tensor_action


def _candidate_arrays(primitive, order, cutoff, body):
    return _candidate_labels(
        primitive.cell,
        primitive.scaled_positions,
        order=order,
        cutoff=cutoff,
        max_body_order=body,
    )


@pytest.mark.parametrize("order", (2, 3, 4, 5))
def test_candidates_match_original_labels_and_order(order):
    for atoms, cutoff in (
        (bulk("Ar", "sc", a=1), 1.1),
        (bulk("Si", "diamond", a=5.43), 2.5),
        (
            Atoms(
                "Ar",
                scaled_positions=[[0, 0, 0]],
                cell=[[1, 0, 0], [0.7, 1.3, 0], [0.3, 0.2, 1.5]],
                pbc=True,
            ),
            1.6,
        ),
    ):
        primitive = ClusterSpace(atoms, symprec=1e-05, cutoffs={2: 0.01})
        for body in (1, order):
            args = {"order": order, "cutoff": cutoff, "max_body_order": body}
            actual = [
                Cluster.from_labels(row).labels
                for row in _candidate_arrays(primitive, order, cutoff, body)
            ]
            expected = [c.labels for c in reference_candidates(primitive, **args)]
            assert actual == expected


def test_lattice_tensor_action_matches_numpy_contraction_and_permutation():
    rng = np.random.default_rng(71)
    rotation = np.array([[1, 7, 0], [0, 1, 0], [0, 0, -1]], dtype=np.int64)
    for order in range(2, 6):
        values = rng.integers(-3, 4, (3**order, 3), dtype=np.int64)
        permutation = tuple(reversed(range(order)))
        transformed = values.T.reshape((3,) + (3,) * order)
        for axis in range(order):
            transformed = np.tensordot(rotation, transformed, axes=((1,), (axis + 1,)))
            transformed = np.moveaxis(transformed, 0, axis + 1)
        expected = (
            np.transpose(transformed, (0,) + tuple(i + 1 for i in permutation)).reshape(3, -1).T
        )
        np.testing.assert_array_equal(
            _tensor_action(values, rotation, np.asarray(permutation, dtype=np.int64)), expected
        )
