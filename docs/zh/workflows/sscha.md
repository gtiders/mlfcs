# 原生 SSCHA 模块

[English]

`mlfcs.sscha` 使用 ASE 力快照迭代拟合有效 FC2，并由当前谐波哈密顿量采样下一轮正则
系综。它采用 MLFCS 自身的对称性约化 Gram 拟合器和 compact q 空间采样器，运行时不再
依赖 phonopy 或 symfc。

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

采样器在与对角超胞相容的 q 点上傅里叶变换平移约化 FC2。程序对角化的是
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
from mlfcs.sscha import SSCHA

sscha = SSCHA(
    read("POSCAR"),
    supercell=(3, 3, 3),
    temperature=300.0,
    statistics="quantum",
    snapshots=1000,
    max_iterations=10,
    initial_displacement=0.01,
    random_seed=42,
    cutoff_frequency=0.01,  # THz
    imaginary_modes="error",
    max_displacement=None,  # 默认不裁剪正则系综
    log_level=1,
)
sscha.run(make_my_ase_calculator())
```

`run()` 逐个计算构型，避免复制大型计算器。只需要有效 FC2 时可设置
`calculate_free_energy=False`，从而不请求能量。

## 外部 sow/reap 流程

```python
structures = sscha.sow()

for atoms in structures:
    dispatch(
        atoms.info["mlfcs_sscha_iteration"],
        atoms.info["mlfcs_configuration_id"],
        atoms,
    )

result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

位置数组必须严格遵循 `sow()` 顺序；字典必须完整包含从零到 `snapshots - 1` 的整数 ID。
力是必要输入，能量与未位移参考能量只用于自由能估计。

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
`relative_force_constant_change` 保存相对于本轮采样 FC2 的更新幅度；初始化轮该值为
`None`。公共迭代对象只暴露这些标量诊断，不重复公开内部采样哈密顿量。

## 结果、平均与写出

```python
previous = sscha.force_constants
iteration = sscha.step(calculator)

average = sscha.averaged_force_constants(last=5)
sscha.use_average(last=5)

sscha.write("FORCE_CONSTANTS", format="text")
sscha.write("fc2-300K.hdf5", format="hdf5")
```

`force_constants` 和每轮 FC2 使用 MLFCS 内部完整超胞原子顺序；平移约化数组可由
`compact_force_constants` 获取。文本和 HDF5 写出复用项目已有的 phonopy 兼容 writer，
但不会导入 phonopy。

## 自由能

相容 q 点的同一组本征解用于计算每原胞量子谐波自由能。一轮自由能对应生成该轮快照的
试探 FC2，而该轮 `force_constants` 是为下一次更新新拟合的 FC2。提供快照能量时：

```text
F = F_harm + mean[(E(u) - E(0) - 1/2 u Phi u) / 超胞原胞数]
```

误差为采样修正项均值的标准误差。如果 `max_displacement` 实际裁剪了样本，该分布已不再
严格正则，自由能估计必须解释为近似结果。
