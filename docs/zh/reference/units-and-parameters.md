---
title: 单位、cutoff 与公共参数
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 单位、cutoff 与公共参数

## 单位

| 量 | 单位或约定 |
|---|---|
| 晶格、坐标、位移、cutoff | Å |
| 力 | eV/Å |
| $n$ 阶 IFC | eV/Å$^n$ |
| 温度 | K |
| 声子频率 | THz |
| 自由能 | eV/primitive cell |
| `symprec` | Å 尺度的结构与对称性容差 |

## cutoff 的三种语义

- 正数：以 Å 为单位的实空间距离；高阶 cluster 要求所有原子对均落在 cutoff 内。
- 负整数：邻居壳层。例如 `-3` 选择第三与第四壳层中点作为安全半径。
- `None`：从当前 reference 的第一个周期像歧义边界减去 0.01 Å。

`None` 不是无穷远相互作用，也不是“保留有限超胞所有 Hessian 自由度”。它只给 primitive exact-$R$
模型选择当前 reference 中不会同时看到同一原子对多个周期像的最大安全半径。极性晶体仍需检查超胞收敛；
当前版本没有解析长程静电力扣除。

## `symprec`

`symprec` 同时参与 primitive/reference 对应、spglib 对称性识别和超胞原子映射。它不是拟合误差容差。
不同任务若使用不同 `symprec`，可能得到不同 orbit 和参数数，因此应把它记录在案例设置中。

## `tolerance`

- `ForceConstantFitter.fit(tolerance=...)`：迭代线性求解器停止阈值。
- `LoopSCPH(tolerance=...)`：相邻 SCPH 频率的 RMS 变化，单位 THz。
- `enforce_rotational_sum_rules(tolerance=...)`：无量纲化约束矩阵的谱秩阈值。

这些参数都不会修改 interaction cutoff，也不会把小 IFC 元素直接归零。

## `mixing`

SCPH 和 SSCHA 使用

$$
\Phi_{k+1}=(1-\alpha)\Phi_k+\alpha\Phi_{\mathrm{new}},
$$

其中 `mixing` 即 $\alpha$，范围为 $(0,1]$。较小值通常更稳但更慢；它是迭代松弛参数，不是物理参数。

## `symprec`：唯一的长度精度

核心原胞—超胞关系中只有一个用户可配置的长度精度：

| 参数 | 单位 | 覆盖的判定 |
|---|---:|---|
| `symprec` | Å | spglib 原胞对称性识别、原胞与显式超胞的整数复制关系、超胞原子到「原胞原子 + 整数格矢」的匹配、固定胞训练帧的胞身份 |

它表达一个统一的物理问题：两个周期几何对象在笛卡尔空间中相差多少时仍视为同一个对象。因此：

- 只有笛卡尔长度（或「每个原胞晶格系数」的笛卡尔长度）可以与 `symprec` 比较；无量纲矩阵元素、分数
  坐标差和角度都不允许直接比较；
- 不允许新增 `mapping_tolerance`、`cell_tolerance`、`position_tolerance` 等同义参数；
- `mlfcs.tools.structure_alignment` 的 `tolerance` 是**外部导入策略**，用于重排独立程序或 MD 产生的
  结构，不参与核心结构身份判定；
- spglib 的 `angle_tolerance` 由 spglib 自动处理（不传或传 `-1.0`），MLFCS 不暴露它，也不用它做
  原胞—超胞映射。

参考超胞必须由用户显式提供：`InteractionSpace`、`FiniteDifferenceCalculation`、`ForceConstantFitter`
和 `SSCHA` 的 `reference` 都是无默认值的必需参数。需要便利构造时显式调用
`mlfcs.tools.supercell.build_supercell(primitive, matrix, symprec=...)`；主线不依赖该工具，也不会根据
cutoff 猜测超胞。MLFCS 接受任何在 `symprec` 内与原胞构成整数复制关系的显式超胞，不要求它来自本工具。
