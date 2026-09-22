"""The explicit-supercell and single-``symprec`` contract.

MLFCS 4.0 forces the caller to hand in the reference supercell and to name one length precision
for the primitive-supercell geometry.  These tests are the contract, not a description of the
current code: they lock the removed import paths, the removed keyword, the required ``reference``
argument of every computational entry point, and the rule that no geometry decision in the core
mapping files compares a dimensionless number with an angstrom tolerance.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import bulk

from mlfcs.structure.relation import StructureRelation

CORE_GEOMETRY_FILES = (
    "src/mlfcs/structure/relation.py",
    "src/mlfcs/structure/integer_lattice.py",
    "src/mlfcs/structure/supercell_mapping.py",
)

ENTRY_POINTS = (
    ("mlfcs.interactions.space", "InteractionSpace"),
    ("mlfcs.finite_difference.calculation", "FiniteDifferenceCalculation"),
    ("mlfcs.fitting.fitter", "ForceConstantFitter"),
    ("mlfcs.phonon.sscha.solver", "SSCHA"),
)


def _subprocess_env() -> dict[str, str]:
    """Environment for the child probes: the worktree's own source tree, deterministically."""
    import os

    root = Path(__file__).parents[1]
    return {**os.environ, "PYTHONPATH": f"{root / 'src'}:{root / 'tests'}"}


def _primitive_and_reference() -> tuple[Atoms, Atoms]:
    primitive = bulk("Ar", "fcc", a=5.26)
    reference = primitive.repeat((2, 2, 2))
    return primitive, reference


def test_the_old_tolerance_keyword_is_gone() -> None:
    """``tolerance`` is renamed, not aliased: the call has to fail by name."""
    primitive, reference = _primitive_and_reference()
    with pytest.raises(TypeError) as failure:
        StructureRelation.from_atoms(primitive, reference, tolerance=1e-5)
    assert "tolerance" in str(failure.value)
    parameters = inspect.signature(StructureRelation.from_atoms).parameters
    assert "symprec" in parameters and "tolerance" not in parameters


def test_the_relation_accepts_the_single_precision() -> None:
    """``symprec`` builds the relation and is recorded with its two residuals."""
    primitive, reference = _primitive_and_reference()
    relation = StructureRelation.from_atoms(primitive, reference, symprec=1e-5)
    assert relation.symprec == 1e-5
    assert relation.cell_residual < 1e-5
    assert relation.position_residual < 1e-5
    assert relation.supercell_matrix.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_the_precision_must_be_a_positive_length(value: float) -> None:
    """A non-positive or non-finite precision cannot describe a physical problem."""
    primitive, reference = _primitive_and_reference()
    with pytest.raises(ValueError):
        StructureRelation.from_atoms(primitive, reference, symprec=value)


@pytest.mark.parametrize(("module", "name"), ENTRY_POINTS)
def test_every_computation_entry_requires_an_explicit_reference(module: str, name: str) -> None:
    """No entry point may build its own supercell: ``reference`` has no default."""
    imported = __import__(module, fromlist=[name])
    signature = inspect.signature(getattr(imported, name))
    assert "reference" in signature.parameters, f"{module}.{name} lost its reference argument"
    parameter = signature.parameters["reference"]
    assert parameter.default is inspect.Parameter.empty, f"{module}.{name} has a default supercell"
    assert parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def test_the_root_and_structure_namespaces_no_longer_offer_the_builder() -> None:
    """The convenience builder moved to a leaf tool package; neither namespace re-exports it."""
    import mlfcs
    import mlfcs.structure

    assert not hasattr(mlfcs, "build_supercell")
    assert not hasattr(mlfcs.structure, "build_supercell")
    assert "build_supercell" not in mlfcs.__all__
    assert "build_supercell" not in mlfcs.structure.__all__


def test_the_tools_package_offers_the_supercell_builder() -> None:
    """The one supported path for building a supercell."""
    code = "from mlfcs.tools.supercell import build_supercell; print('ok')"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_no_mainline_package_imports_the_tools_package() -> None:
    """``tools`` is a leaf: the core packages must not reach for it."""
    root = Path("src/mlfcs")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] == "tools":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            if any(name == "mlfcs.tools" or name.startswith("mlfcs.tools.") for name in names):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, offenders


def test_the_supercell_module_left_the_structure_package() -> None:
    """The moved file must not linger as a shim."""
    assert not Path("src/mlfcs/structure/supercell.py").exists()


@pytest.mark.parametrize("path", CORE_GEOMETRY_FILES)
def test_core_geometry_uses_no_float_tolerance_literals(path: str) -> None:
    """Dimensionless numbers are never compared with an angstrom tolerance.

    The gate looks at comparison expressions and at the ``atol``/``rtol`` of
    ``allclose``/``isclose``, which is where a hidden threshold can hide; it deliberately does not
    forbid every float literal, so logging formats and physical inputs stay legal.
    """
    tree = ast.parse(Path(path).read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in {"allclose", "isclose"}:
                offenders.append(f"line {node.lineno}: {name}(...)")
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                if isinstance(operand, ast.Constant) and isinstance(operand.value, float):
                    offenders.append(f"line {node.lineno}: float literal in comparison")
    assert not offenders, (path, offenders)
