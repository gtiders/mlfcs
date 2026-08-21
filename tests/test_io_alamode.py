from xml.etree.ElementTree import parse

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import minkowski_reduce
from ase.units import Bohr, Rydberg
from supercell_helpers import make_supercell

from mlfcs.core.geometry import StructureRelation
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.io.alamode import AlamodeMirrorImageError, write_alamode


def _sparse(order, n_primitive, n_supercell, cluster, component, value):
    tensor = np.zeros((3,) * order)
    tensor[component] = value
    n_cells = n_supercell // n_primitive
    sites = np.asarray([[atom // n_cells for atom in cluster]], dtype=np.int32)
    cells = np.asarray([[atom % n_cells, 0, 0] for atom in cluster], dtype=np.int32)
    translations = (cells[1:] - cells[0]).reshape(1, order - 1, 3)
    return SparseOrderForceConstants(
        order,
        sites,
        translations,
        np.asarray([tensor]),
    )


def test_alamode_xml_preserves_mlfcs_atom_order_and_translation_map(tmp_path):
    primitive = Atoms(
        "NaCl",
        scaled_positions=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
        cell=np.diag((4.0, 5.0, 6.0)),
        pbc=True,
    )
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    fc2 = _sparse(2, 2, 4, (0, 1), (0, 0), 2.5)
    force_constants = ForceConstants({}, supercell, sparse={2: fc2})

    output = tmp_path / "force_constants.xml"
    force_constants.write(output, format="alamode")
    root = parse(output).getroot()

    positions = root.findall("Structure/Position/pos")
    assert [node.get("element") for node in positions] == ["Na", "Na", "Cl", "Cl"]
    mappings = root.findall("Symmetry/Translations/map")
    assert [(node.get("tran"), node.get("atom"), node.text) for node in mappings] == [
        ("1", "1", "1"),
        ("1", "2", "3"),
        ("2", "1", "2"),
        ("2", "2", "4"),
    ]

    # Atom 2 lies at half the supercell length from atom 1. ALAMODE represents
    # the two closest mirror images separately and divides the value between them.
    entries = root.findall("ForceConstants/HARMONIC/FC2")
    assert len(entries) == 2
    assert {entry.get("pair2") for entry in entries} == {"2 1 1", "2 1 6"}
    recovered = sum(float(entry.text) for entry in entries) * Rydberg / Bohr**2
    assert np.isclose(recovered, 2.5, atol=1e-13, rtol=1e-13)


def test_alamode_xml_writes_fc2_fc3_fc4_and_reuses_mirror_for_repeated_atom(tmp_path):
    primitive = Atoms("Si", positions=((0.0, 0.0, 0.0),), cell=np.eye(3) * 3.0, pbc=True)
    supercell, _ = make_supercell(primitive, (2, 1, 1))
    sparse = {
        2: _sparse(2, 1, 2, (0, 1), (0, 0), 1.0),
        3: _sparse(3, 1, 2, (0, 1, 1), (0, 0, 0), 2.0),
        4: _sparse(4, 1, 2, (0, 1, 1, 1), (0, 0, 0, 0), 3.0),
    }
    output = tmp_path / "force_constants.xml"
    ForceConstants({}, supercell, sparse=sparse).write(output, format="fcsxml")
    root = parse(output).getroot()

    for order, path, expected in (
        (2, "ForceConstants/HARMONIC/FC2", 1.0),
        (3, "ForceConstants/ANHARM3/FC3", 2.0),
        (4, "ForceConstants/ANHARM4/FC4", 3.0),
    ):
        entries = root.findall(path)
        # Repeated appearances of atom 2 must use one common mirror image;
        # they do not form a Cartesian product of independent images.
        assert len(entries) == 2
        recovered = sum(float(entry.text) for entry in entries) * Rydberg / Bohr**order
        assert np.isclose(recovered, expected, atol=1e-13, rtol=1e-13)


def test_alamode_xml_omits_text_noise_below_export_tolerance(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
    supercell, _ = make_supercell(primitive, (1, 1, 1))
    small = _sparse(2, 1, 1, [0, 0], [0, 0], 9e-9)
    retained = _sparse(3, 1, 1, [0, 0, 0], [0, 0, 0], 1e-8)
    output = tmp_path / "noise.xml"

    ForceConstants({}, supercell, sparse={2: small, 3: retained}).write(output, format="alamode")

    root = parse(output).getroot()
    assert not root.findall("ForceConstants/HARMONIC/FC2")
    retained_entries = root.findall("ForceConstants/ANHARM3/FC3")
    assert retained_entries
    assert all(float(entry.text) != 0.0 for entry in retained_entries)


def test_alamode_xml_selects_one_order_and_rejects_higher_orders(tmp_path):
    primitive = Atoms("Si", positions=((0.0, 0.0, 0.0),), cell=np.eye(3) * 3.0, pbc=True)
    supercell, _ = make_supercell(primitive, (1, 1, 1))
    sparse = {
        2: _sparse(2, 1, 1, (0, 0), (0, 0), 1.0),
        3: _sparse(3, 1, 1, (0, 0, 0), (0, 0, 0), 2.0),
    }
    force_constants = ForceConstants({}, supercell, sparse=sparse)
    output = tmp_path / "fc3.xml"
    force_constants.write(output, format="alamode_xml", order=3)
    root = parse(output).getroot()
    assert not root.findall("ForceConstants/HARMONIC/FC2")
    assert root.findall("ForceConstants/ANHARM3/FC3")

    fc5 = _sparse(5, 1, 1, (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), 1.0)
    with pytest.raises(ValueError, match="only orders 2, 3, and 4"):
        ForceConstants({}, supercell, sparse={5: fc5}).write(
            tmp_path / "fc5.xml", format="alamode", order=5
        )

    # Generic higher orders coexist with the FC2--FC4 XML subset.
    mixed = ForceConstants({}, supercell, sparse={2: sparse[2], 5: fc5})
    mixed.write(tmp_path / "mixed.xml", format="alamode")
    mixed_root = parse(tmp_path / "mixed.xml").getroot()
    assert mixed_root.findall("ForceConstants/HARMONIC/FC2")
    assert not mixed_root.findall("ForceConstants/ANHARM5/FC5")


def test_alamode_sparse_entries_follow_reordered_reference_labels(tmp_path):
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    supercell, _ = make_supercell(primitive, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    supercell = supercell[[1, 0]]
    from mlfcs.core.geometry import PeriodicIndex

    index = PeriodicIndex(
        supercell.arrays["primitive_index"],
        supercell.arrays["cell_translation"],
        supercell.info["mlfcs_supercell_matrix"],
    )
    first = index.representative(0)
    tail = index.atom(0, [1, 0, 0])
    sparse = _sparse(2, 1, 2, (first, tail), (0, 0), 2.0)
    output = tmp_path / "reordered.xml"
    ForceConstants({}, supercell, sparse={2: sparse}).write(output, format="alamode")

    entry = parse(output).getroot().find("ForceConstants/HARMONIC/FC2")
    assert entry is not None
    assert entry.attrib["pair1"] == "1 1"


def test_alamode_rebases_a_nonreduced_equivalent_supercell_before_27_image_encoding(tmp_path):
    primitive_cell = np.eye(3) * 4
    change = np.asarray([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
    # This pair's true minimum image uses a coefficient outside [-1, 1] in
    # ``change @ primitive_cell`` but is representable after Minkowski rebasing.
    position = np.asarray([1.73052316, 2.13823552, 1.69113868])
    primitive = Atoms("HHe", positions=[[0, 0, 0], position], cell=primitive_cell, pbc=True)
    reference = Atoms(
        "HHe",
        positions=[[0, 0, 0], position],
        cell=change @ primitive_cell,
        pbc=True,
    )
    relation = StructureRelation.from_atoms(primitive, reference)
    sparse = _sparse(2, 2, 2, (0, 1), (0, 0), 1.0)
    sparse.sites = np.asarray([[0, 1]], dtype=np.int32)
    sparse.translations = np.asarray([[[0, 0, 0]]], dtype=np.int32)
    values = ForceConstants(
        {},
        relation.reference,
        sparse={2: sparse},
        relation=relation,
    )
    with pytest.raises(AlamodeMirrorImageError):
        write_alamode(tmp_path / "raw.xml", values, orders=(2,))

    output = tmp_path / "rebased.xml"
    values.write(output, format="alamode")
    root = parse(output).getroot()
    lattice = (
        np.asarray(
            [
                [
                    float(value)
                    for value in root.find(f"Structure/LatticeVector/a{axis}").text.split()
                ]
                for axis in range(1, 4)
            ]
        )
        * Bohr
    )
    reduced, _ = minkowski_reduce(reference.cell, pbc=True)
    np.testing.assert_allclose(lattice, reduced)
    assert root.findall("ForceConstants/HARMONIC/FC2")
