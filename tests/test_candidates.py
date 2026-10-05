"""Candidate enumeration compared with the independent Python oracle."""

import pytest
from ase import Atoms
from ase.build import bulk
from oracles.candidates_reference import iter_candidates as reference_candidates

from mlfcs import ClusterSpace
from mlfcs.cluster_space import Cluster
from mlfcs.cluster_space.candidates import enumerate_candidate_labels


@pytest.mark.parametrize("order", (2, 3, 4, 5), ids=("FC2", "FC3", "FC4", "FC5"))
@pytest.mark.parametrize("body_mode", ("onsite", "full"))
@pytest.mark.parametrize(
    "atoms,cutoff",
    (
        pytest.param(bulk("Ar", "sc", a=1), 1.1, id="cubic-Ar"),
        pytest.param(bulk("Si", "diamond", a=5.43), 2.5, id="diamond-Si"),
        pytest.param(
            Atoms(
                "Ar",
                scaled_positions=[[0, 0, 0]],
                cell=[[1, 0, 0], [0.7, 1.3, 0], [0.3, 0.2, 1.5]],
                pbc=True,
            ),
            1.6,
            id="skewed-Ar",
        ),
    ),
)
def test_candidates_match_reference_labels_and_order(order, body_mode, atoms, cutoff):
    """Compare anchored labels and enumeration order for one material and truncation."""
    primitive = ClusterSpace(atoms, cutoffs={2: 0.01})
    body = 1 if body_mode == "onsite" else order
    settings = {"order": order, "cutoff": cutoff, "max_body_order": body}
    labels = enumerate_candidate_labels(primitive.cell, primitive.scaled_positions, **settings)
    actual = [Cluster.from_labels(row).labels for row in labels]
    expected = [cluster.labels for cluster in reference_candidates(primitive, **settings)]
    assert actual == expected
