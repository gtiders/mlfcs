# 入门

MLFCS 提供 Python API，不提供命令行界面。一个计算明确分为构造计算、生成结构和提供力三个阶段。

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import ForceConstantCalculation
from mlfcs import build_supercell

primitive = bulk("Al", "fcc", a=4.05)
reference_supercell = build_supercell(primitive, (2, 2, 2))
calculation = ForceConstantCalculation(
    primitive, reference=reference_supercell, order=2
)
result = calculation.run(EMT())
write_force_constants(result, "fc2.h5", format="hdf5")
```

对于昂贵的 calculator，请使用[外部 `sow()`/`reap()` 工作流](../workflows/external-calculators.md)，
并在结果旁保存 manifest。详见[安装](installation.md)、[第一个 FC2](first-fc2.md)和[结构约定](structures.md)。
