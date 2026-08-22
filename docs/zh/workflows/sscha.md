# 原生 SSCHA 模块

[English]

`mlfcs.physics.sscha` 使用 ASE 力快照迭代拟合有效 FC2，并由当前谐波哈密顿量采样下一轮正则
系综。它采用 MLFCS 自身的对称性约化 Gram 拟合器和 compact q 空间采样器。

## 方法

没有初始 FC2 时，第一轮使用小幅高斯笛卡尔位移。每批力都由
`ForceConstantFitter(orders=(2,))` 使用全部快照拟合，并在不可约参数空间施加 ASR。
后续轮次按照当前 FC2 采样：

```text
初始笛卡尔快照 → ASE 力 → 原生 Gram FC2 拟合
                                  │
                                  ▼
compact FC2 q 空间系综 → ASE 力 → 原生 Gram FC2 拟合 → 重复
```

采样器在由 reference 超胞矩阵定义的 reciprocal quotient q 点上傅里叶变换平移约化 FC2。程序对角化的是
`3 × 原胞原子数` 大小的矩阵，而不是一个 `3 × 超胞原子数` 的完整矩阵。共轭 q 点显式
配对，Gamma 点的三个质量加权平移方向被严格投影掉。

默认采用量子统计：

```text
variance(q_s) = hbar / (2 omega_s) coth[hbar omega_s / (2 kB T)]
```

设置 `statistics="classical"` 可改为经典的 `kB T / omega_s**2`。

## ASE Calculator 直接流程

```python
from ase.io import read
from mlfcs.physics.sscha import SSCHA

calculation = SSCHA(
    read("POSCAR"),
    reference=read("reference-supercell.vasp"),
    temperature=300.0,
    statistics="quantum",
    snapshots=1000,
    max_iterations=10,
    initial_displacement=0.01,
    random_seed=42,
    cutoff_frequency=0.01,  # THz
    imaginary_modes="error",
    max_displacement=None,  # 默认不裁剪正则系综
    mixing=1.0,             # 直接固定点更新
    log_level=1,
)
result = calculation.run(make_my_ase_calculator())
```

`run()` 逐个计算构型，避免复制大型计算器。只需要有效 FC2 时可设置
`calculate_free_energy=False`，从而不请求能量。

温度可以是标量或序列。序列会自动按升温顺序执行；默认将前一温度的最终 FC2 作为下一温度
的初始 FC2。传入 `continuation=False` 可让各温度独立从相同初始模型运行：

```python
series = SSCHA(
    read("POSCAR"),
    reference=read("reference-supercell.vasp"),
    temperature=[600, 300, 450],
    snapshots=1000,
    max_iterations=10,
).run(make_my_ase_calculator())

fc2_at_450K = series.at_temperature(450).force_constants
```

多温度对象同样通过 `run(calculator)` 工作。SSCHA 由给定 ASE calculator 直接计算内部生成的
系综，刻意不提供有限差分式的 `sow()`/`reap()` 接口，避免快照、温度和更新状态脱节。

## 稳定性控制

- `cutoff_frequency` 以 THz 为单位，排除除平移外的过低频模态；
- `imaginary_modes="error"` 为默认值，遇到不稳定试探哈密顿量直接终止；
- `imaginary_modes="absolute"` 使用虚频绝对值采样，并记录虚频诊断；
- `imaginary_modes="exclude"` 不采样虚频模态；
- `max_displacement=None` 保持严格正则分布。设置正数后启用 phonopy 风格的逐原子径向
  裁剪：方向不变，只缩短向量长度。程序报告被裁剪原子数和受影响快照数，因为裁剪后
  样本不再严格服从原正则系综。

`SSCHAIteration.ensemble` 保存 q 点数、模态数、虚频数、排除数和裁剪统计；
`fitting_relative_force_error` 保存原生拟合器的训练相对力误差，
`relative_force_constant_change` 保存线性混合后相对于本轮采样 FC2 的更新幅度；
`raw_relative_force_constant_change` 保存混合前的拟合更新幅度；初始化轮二者均为
`None`。公共迭代对象只暴露这些标量诊断，不重复公开内部采样哈密顿量。

`mixing` 控制自洽更新，不参与力常数回归：

```text
Phi_next = (1 - mixing) Phi_sampled + mixing Phi_fitted.
```

默认 `mixing=1` 与直接替换完全一致。小于 1 时，对下一轮采样使用的哈密顿量做欠松弛；
它可缓解有限随机样本造成的固定点振荡，但不改变当前快照所拟合出的 IFC。

## 结果与写出

```python
write_force_constants(result.force_constants, "mlfcs.h5", format="hdf5")
write_force_constants(result.force_constants, "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
write_force_constants(result.force_constants, "force_constants.xml", format="alamode", order=2)
```

`result.force_constants` 是标准的 lattice-labelled `ForceConstants`，只包含自洽后的有效
FC2，并具有和有限差分、拟合结果相同的 structure relation 与通用导出能力。history 只保存
诊断摘要；需要稠密数组时显式调用 `result.force_constants.materialize(2)`。

## 自由能

相容 q 点的同一组本征解用于计算每原胞量子谐波自由能。一轮自由能对应生成该轮快照的
试探 FC2，而当前 active FC2 是为下一次更新新拟合的 FC2。提供快照能量时：

```text
F = F_harm + mean[(E(u) - E(0) - 1/2 u Phi u) / 超胞原胞数]
```

误差为采样修正项均值的标准误差。如果 `max_displacement` 实际裁剪了样本，该分布已不再
严格正则，自由能估计必须解释为近似结果。
