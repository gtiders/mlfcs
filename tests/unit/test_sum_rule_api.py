import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ForceConstantCalculation


def test_higher_order_rotational_sum_rule_is_rejected_explicitly():
    calculation = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=3,
        supercell=(1, 1, 1),
        cutoff=-1,
        verbose=False,
    )

    with pytest.raises(ValueError, match="only for order=2"):
        calculation.reap(np.empty((0, 0, 3)), rotational_sum_rule=True)
