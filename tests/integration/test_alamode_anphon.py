"""Optional end-to-end validation with ALAMODE's ``anphon`` reader."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from xml.etree.ElementTree import parse

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr

from mlfcs import build_supercell
from mlfcs.model import ForceConstants, SparseOrderForceConstants

pytestmark = pytest.mark.integration


def _anphon() -> str:
    executable = os.environ.get("MLFCS_ANPHON", "anphon")
    resolved = shutil.which(executable)
    if resolved is None:
        pytest.skip("ALAMODE anphon is unavailable; set MLFCS_ANPHON to enable this reader test")
    return resolved


def _run_anphon(tmp_path: Path, fcsxml: Path, *, lattice_parameter: float = 4.0) -> None:
    """Run a one-species harmonic calculation using a supplied FCSXML."""
    input_file = tmp_path / "anphon.in"
    input_file.write_text(
        "&general\n"
        "  PREFIX = mlfcs\n"
        "  MODE = phonons\n"
        f"  FCSXML = {fcsxml.name}\n"
        "  NKD = 1; KD = Si\n"
        "/\n\n"
        "&cell\n"
        f"  {lattice_parameter / Bohr:.15e}\n"
        "  1.0 0.0 0.0\n"
        "  0.0 1.0 0.0\n"
        "  0.0 0.0 1.0\n"
        "/\n\n"
        "&kpoint\n"
        "  1\n"
        "  G 0.0 0.0 0.0 G 0.0 0.0 0.0 1\n"
        "/\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [_anphon(), input_file.name],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Job finished" in completed.stdout
    assert (tmp_path / "mlfcs.bands").is_file()


def test_anphon_reads_mlfcs_fcsxml(tmp_path: Path):
    """Run the external harmonic reader against an MLFCS-produced FCSXML."""
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    supercell = build_supercell(primitive, (1, 1, 1))
    sparse = SparseOrderForceConstants(
        2,
        n_primitive=1,
        n_supercell=1,
        clusters=np.asarray([[0, 0]], dtype=np.int32),
        tensors=np.eye(3)[None, :, :],
    )
    fcsxml = tmp_path / "mlfcs.xml"
    ForceConstants({}, supercell, sparse={2: sparse}).write(fcsxml, format="alamode")

    _run_anphon(tmp_path, fcsxml)


def test_anphon_reads_27_image_expansion(tmp_path: Path):
    """Exercise ALAMODE's two-way half-supercell mirror encoding."""
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    supercell = build_supercell(primitive, (2, 1, 1))
    tensor = np.zeros((1, 3, 3))
    tensor[0, 0, 0] = 1.0
    sparse = SparseOrderForceConstants(
        2,
        n_primitive=1,
        n_supercell=2,
        clusters=np.asarray([[0, 1]], dtype=np.int32),
        tensors=tensor,
    )
    fcsxml = tmp_path / "half-supercell.xml"
    ForceConstants({}, supercell, sparse={2: sparse}).write(fcsxml, format="alamode")

    entries = parse(fcsxml).getroot().findall("ForceConstants/HARMONIC/FC2")
    assert len(entries) == 2
    assert {entry.attrib["pair2"] for entry in entries} == {"2 1 1", "2 1 6"}
    _run_anphon(tmp_path, fcsxml)
