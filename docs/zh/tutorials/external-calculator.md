---
title: 外部计算器工作流
audience:
  - user
status: stable
code_verified: 4.0.0a4
examples:
  - examples/finite-difference/Si/harmonic
---

# 外部计算器工作流

## 目标

生成有序位移计划，用 VASP 或其他外部程序计算每个结构，并在没有隐藏原子重排的情况下重建力常数。

## 准备与 sow

~~~python
from pathlib import Path
from ase.io import read, write
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("primitive.vasp"),
    reference=read("reference.vasp"),
    order=3,
    cutoff=-5,
    displacement=0.01,
)
structures = calculation.sow()
for index, atoms in enumerate(structures):
    directory = Path("calculations") / f"{index:05d}"
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "POSCAR", atoms, format="vasp")
~~~

每个提交任务都要保存从零开始的 configuration index。MLFCS 除结构外不生成 VASP 输入，也不提交任务。

## 收集与 reap

~~~python
import numpy as np
from ase.io import read
from mlfcs import write_force_constants

forces = np.asarray([
    read(f"calculations/{index:05d}/vasprun.xml", index=-1).get_forces()
    for index in range(len(structures))
])
fc3 = calculation.reap(forces, acoustic_sum_rule=True)
write_force_constants(fc3, "mlfcs.h5", format="hdf5")
~~~

`forces[i]` 必须属于 `structures[i]`。任务乱序完成时，应传入受支持的 configuration-ID mapping，而不是猜测文件名字典序。

## 结果与下一步

结构 manifest、calculator 输入和原始输出应与计算来源一起保存。[Si 谐性案例](../examples/si-finite-difference.md)使用归档输出、独立力收集脚本、重建脚本和绘图脚本展示了这种分离。
