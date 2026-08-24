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

## `batch_size`

拟合中的 `batch_size` 只能取 1–4。它控制一次送入 design kernel 的结构帧数，影响吞吐量和临时内存，
不会重复样本或重复求解。MLFCS 不自动猜测硬件饱和点，用户应在相同数值设置下自行比较。

## `mixing`

SCPH 和 SSCHA 使用

$$
\Phi_{k+1}=(1-\alpha)\Phi_k+\alpha\Phi_{\mathrm{new}},
$$

其中 `mixing` 即 $\alpha$，范围为 $(0,1]$。较小值通常更稳但更慢；它是迭代松弛参数，不是物理参数。
