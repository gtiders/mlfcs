import numpy as np
from ase.build import bulk
from supercell_helpers import make_supercell

from mlfcs.anharmonic.common.fc2 import compact_fc2, expand_compact_fc2


def test_compact_fc2_translation_expansion_roundtrip():
    primitive = bulk("Si", "diamond", a=5.43)
    supercell, _ = make_supercell(primitive, (2, 2, 2))
    rng = np.random.default_rng(7)
    compact = rng.normal(size=(len(primitive), len(supercell), 3, 3))

    full = expand_compact_fc2(compact, supercell)

    assert full.shape == (len(supercell), len(supercell), 3, 3)
    np.testing.assert_allclose(compact_fc2(full, supercell), compact, atol=0, rtol=0)
