"""Optional end-to-end validation with ALAMODE's ``anphon`` reader."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

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


def test_anphon_reads_mlfcs_fcsxml(tmp_path: Path):
    """Run the external harmonic reader against an MLFCS-produced FCSXML."""
    primitive = Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 4.0, pbc=True)
    supercell = build_supercell(primitive, (1, 1, 1))
    tensor = np.eye(3)[None, :, :]
    sparse = SparseOrderForceConstants(
        2,
        n_primitive=1,
        n_supercell=1,
        clusters=np.asarray([[0, 0]], dtype=np.int32),
        tensors=tensor,
    )
    fcsxml = tmp_path / "mlfcs.xml"
    ForceConstants({}, supercell, sparse={2: sparse}).write(fcsxml, format="alamode")

    lattice_parameter = 4.0 / Bohr
    input_file = tmp_path / "anphon.in"
    input_file.write_text(
        "&general\n"
        "  PREFIX = mlfcs\n"
        "  MODE = phonons\n"
        "  FCSXML = mlfcs.xml\n"
        "  NKD = 1; KD = Si\n"
        "/\n\n"
        "&cell\n"
        f"  {lattice_parameter:.15e}\n"
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
