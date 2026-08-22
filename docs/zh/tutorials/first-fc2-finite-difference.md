---
title: 第一个有限差分 FC2
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
examples:
  - examples/finite-difference/Si/harmonic
---

# 第一个有限差分 FC2

## 目标

使用直接 ASE calculator 计算谐性 FC2，并保存为原生稀疏 HDF5。

## 前置条件

使用 `uv sync` 安装项目。这个小例子使用 ASE 自带 EMT calculator，不需要外部电子结构程序。

## 步骤

~~~python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import ForceConstantCalculation, build_supercell, write_force_constants

primitive = bulk("Al", "fcc", a=4.05)
reference = build_supercell(primitive, (2, 2, 2))

calculation = ForceConstantCalculation(
    primitive,
    reference=reference,
    order=2,
    cutoff=None,
    displacement=0.01,
)
force_constants = calculation.run(EMT())
write_force_constants(force_constants, "mlfcs.h5", format="hdf5")
~~~

## 结果与解释

`mlfcs.h5` 包含原生 HDF5 v3 稀疏 exact-$R$ FC2。下游工作流需要稠密 phonopy 输出时，应使用[显式 writer](../how-to/read-and-write-ifcs.md)。

## 常见问题

primitive 和 reference 必须构成严格整数超胞关系。真实 calculator 通常需要比最小演示更大的 reference，并通过收敛性确定 cutoff。

## 下一步

仓库中的 [Si 谐性案例](../examples/si-finite-difference.md)从已归档 VASP 输出重建 FC2，并绘制声子谱。
