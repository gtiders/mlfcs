"""Checks for the local contracts used by the production stages."""

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterMap, ClusterSpace
from mlfcs.algebra.integer import exact_product
from mlfcs.cluster_space.candidates import _candidate_labels, neighbors
from mlfcs.core import LatticeSite
from mlfcs.core.tensors import prove_tensor_action
from mlfcs.fitting.design import ForceDesign


def test_integer_product_proves_each_actual_dot_product():
    np.testing.assert_array_equal(exact_product([[2**40, 0]], [[0], [2**40]]), [[0]])
    # A zero final result does not make overflowing intermediate products safe.
    with pytest.raises(OverflowError):
        exact_product([[2**62, 2**62]], [[2], [-2]])


def test_tensor_proof_covers_axis_partial_sums():
    rotation = np.array([[1, 7, 0], [0, 1, 0], [0, 0, -1]])
    assert prove_tensor_action(rotation, 3, 5, subtraction=1) == 5 * 8**3 + 1
    with pytest.raises(OverflowError):
        prove_tensor_action(rotation, 3, 2**60)


def test_neighbor_sizing_stops_before_exceeding_output_capacity():
    offsets = np.zeros(2, dtype=np.int64)
    result = neighbors(
        np.zeros((1, 3)),
        np.eye(3),
        np.ones(3, dtype=np.int64),
        1.1,
        offsets,
        np.empty((0, 4), dtype=np.int64),
        np.empty((0, 3)),
        False,
        2,
    )
    assert result == -1


def test_site_action_rejects_unrepresentable_translation_before_cast():
    space = ClusterSpace(bulk("Ar", "sc", a=1), cutoffs={2: 0.1})
    with pytest.raises(OverflowError):
        space.symmetry.transform_site(0, LatticeSite(0, (2**70, 0, 0)))


def test_periodic_index_is_retained_and_query_shape_is_checked():
    atoms = bulk("Ar", "sc", a=1)
    mapping = ClusterMap(
        ClusterSpace(atoms, cutoffs={2: 0.1}), atoms, supercell_matrix=np.eye(3, dtype=np.int64)
    )
    assert not mapping._periodic.keys.flags.writeable
    assert not mapping._periodic.atom_indices.flags.writeable
    with pytest.raises(ValueError):
        mapping.quotient((0, 0))
    with pytest.raises(OverflowError):
        mapping.quotient((2**70, 0, 0))


@pytest.mark.parametrize("defect", ("shape", "dtype", "readonly"))
def test_workspace_checks_actual_buffers(defect):
    atoms = bulk("Ar", "sc", a=1)
    mapping = ClusterMap(
        ClusterSpace(atoms, cutoffs={2: 0.1}), atoms, supercell_matrix=np.eye(3, dtype=np.int64)
    )
    design = ForceDesign(mapping)
    workspace = design.allocate_workspace()
    array = workspace.scratch[0]
    if defect == "shape":
        workspace.scratch = (array[:, :-1],)
    elif defect == "dtype":
        workspace.scratch = (array.astype(np.int64),)
    else:
        array.setflags(write=False)
    with pytest.raises(ValueError, match="scratch"):
        design.matrix(np.zeros((1, 3)), workspace=workspace)


@pytest.mark.parametrize(
    "settings",
    (
        {"order": 1},
        {"max_body_order": 0},
        {"max_body_order": 3},
        {"cutoff": 0.0},
        {"cutoff": float("inf")},
    ),
)
def test_candidate_inputs_are_checked_before_neighbor_search(monkeypatch, settings):
    def no_neighbor_search(*args, **kwargs):
        pytest.fail("invalid truncation must be rejected before traversal")

    monkeypatch.setattr("mlfcs.cluster_space.candidates.neighbors", no_neighbor_search)
    arguments = {"order": 2, "cutoff": 1.0, "max_body_order": 2, **settings}
    with pytest.raises(ValueError):
        _candidate_labels(np.eye(3), np.zeros((1, 3)), **arguments)
