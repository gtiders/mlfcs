import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[3] / "reference_tools" / "compare_shengbte_fc4.py"
SPEC = spec_from_file_location("compare_shengbte_fc4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(path, sites, vectors, tensor):
    lines = ["1", "", "1"]
    lines.extend(" ".join(map(str, vector)) for vector in vectors)
    lines.append(" ".join(map(str, sites)))
    for direction in np.ndindex((3, 3, 3, 3)):
        lines.append(" ".join(str(value + 1) for value in direction) + f" {tensor[direction]}")
    path.write_text("\n".join(lines) + "\n")


def test_comparison_handles_axis_permutation_and_anchor_change(tmp_path):
    tensor = np.arange(81.0).reshape((3, 3, 3, 3))
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write(left, (1, 2, 3, 4), ((1, 0, 0), (0, 2, 0), (0, 0, 3)), tensor)

    permutation = (2, 0, 3, 1)
    positions = np.asarray(((0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)))
    reordered = positions[list(permutation)] - positions[permutation[0]]
    _write(
        right,
        tuple((1, 2, 3, 4)[index] for index in permutation),
        reordered[1:],
        tensor.transpose(permutation),
    )

    result = MODULE.compare_fc4(left, right)
    assert result.common_blocks == 1
    assert result.left_only == result.right_only == 0
    assert result.common_max_abs == 0.0
