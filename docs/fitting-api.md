# 力拟合 API

统一的 `FitSystem` 保存一个仅使用力的最小二乘问题。只接受 `ForceDataset`，其 mapping、位移和目标力由统一数据入口准备。默认 `representation="normal"` 保存正规方程并使用 MINRES；`representation="raw"` 保存原始方程并使用列缩放的直接最小二乘。内置算法仅这两种，`solve()` 返回 `ForceConstants`。

## 初始化与求解

```python
from ase.io import iread, read
from mlfcs import ClusterMap, ClusterSpace, ForceDataset, FitSystem

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)
mapping = ClusterMap(space, read("SPOSCAR"))
data = ForceDataset(mapping, iread("train.extxyz", index=":"))
system = FitSystem(data)
model = system.solve()
print(system.rmse(model), "eV/Å")
model.save("fit.mlfcs")
```

超胞整数矩阵可以显式传给 `ClusterMap`，省略时自动推断。`FitSystem` 在准备内部力设计时检查超胞能否区分参数。每帧必须保持参考超胞的原子顺序、周期性和晶胞，且 ASE calculator 中已经保存有限的力。构造系统不调用 calculator。`ForceDataset` 一次收集位移与力，不保留帧 Atoms；同一个数据集可以复用于多个系统。没有 Atoms 便利入口。详见[数据集与 Ewald](dataset-api.md)。

## 我们对数据做了什么

`ForceDataset` 将每帧位置与超胞比较，通过最近周期像得到位移 $u_i$；`FitSystem` 再生成对称约化 Taylor 力设计矩阵 $A_i$。力按原子顺序展开，每个原子的 x/y/z 分量连续。列遵循 `cluster_space.parameter_offsets` 以及阶数、orbit、component 的参数布局。Taylor 负号、阶乘、张量基与周期像求和已经计入 $A_i$。

构造过程中不减去平均力，不施加逐帧权重，也不归一化数据。两种表示都对应物理参数 $\theta$ 和同一个问题：

$$
\min_\theta\|A\theta-f\|_2^2.
$$

| 表示 | 可访问的数据 | 算法 | 保存规模 |
|---|---|---|---|
| `raw` | `design_matrix`：$A$；`forces`：$f$ | dense least squares (SVD) | $O(mn)$ |
| `normal` | `normal_matrix`：$H$；`normal_rhs`：$g$ | MINRES | $O(n^2)$ |

$m$ 是全部笛卡尔力方程数，$n$ 是参数数。raw 按帧输入顺序拼接原始方程。normal 逐帧累积：

$$
H=\sum_i A_i^TA_i,\qquad g=\sum_i A_i^Tf_i,\qquad c=\sum_i f_i^Tf_i.
$$

两条路线都保存 `force_squared_norm`：$c$。其他共同属性为 `n_structures`、`n_equations`、`n_parameters`、`representation` 和 `cluster_space`。公开数组只读、保持物理尺度，求解不修改它们。访问另一种表示的专属数据会报 `ValueError`。normal 不保留原始行，不能恢复逐帧残差。

raw 构造时会暂存帧矩阵以便拼接，求解时会申请归一化工作矩阵，需要考虑最终矩阵以外的临时内存。normal 不保留各帧设计矩阵，但参数平方规模的矩阵仍可能很大。满列秩时 $\kappa_2(A^TA)=\kappa_2(A)^2$，病态数据形成正规方程时可能损失精度。

raw 的直接 SVD 求解时间随 $m$ 和 $n$ 增长较快（典型稠密过定系统约为 $O(mn^2)$），但不需要迭代容差。normal 的 MINRES 每步处理 $n\times n$ 矩阵，通常更快、存储更少；它通过形成 $A^TA$ 平方条件数，精度可能较差。

## 归一化属于求解器

`fitting/solve.py` 中的内部 `FitSolver` 类实现两种算法，由 `system.solve()` 调用。它始终进行参数列归一化，算法由方程表示固定决定。

raw 路线将每列缩放到单位二范数：

$$
s_j=\frac{1}{\|A_{:j}\|_2},\qquad S=\operatorname{diag}(s_j).
$$

直接求解器对 $ASz=f$ 做 SVD 最小二乘，随后还原物理参数 $\theta=Sz$。实现先按列最大绝对值缩放，再计算列范数，避免直接平方极大或极小的物理系数。归一化矩阵只存在于求解期间，公开的 $A$、$f$ 保持原值。数值秩使用 SciPy/LAPACK 的机器精度默认阈值；秩亏时返回缩放参数坐标下的最小范数解。

normal 路线取 $s_j=1/\sqrt{H_{jj}}$，MINRES 求解：

$$
SHSz=Sg,\qquad \theta=Sz.
$$

这保留当前列缩放迭代的数值逻辑。缩放平衡不同 FC 阶数的参数列，但无法补足缺失观测或消除线性相关。几何 folded exact rank 与训练方程的数值秩是两件事。零列报 `UnobservedParameterError`。秩亏问题在缩放坐标中求解，所选解不一定最小化物理 $\theta$ 的范数。

| 路线 | 求解设置 | 含义 |
|---|---|---|
| normal / MINRES | `rtol=1e-8`、`maxiter=1000` | MINRES 相对停止容差、迭代上限 |
| raw / direct least squares | 无用户求解参数 | 直接 SVD 分解；数值秩由机器精度默认阈值判定 |

normal 路线仍通过 `solve(rtol=..., maxiter=...)` 控制 MINRES。raw 路线调用 LAPACK 直接分解，不暴露迭代停止阈值。它仍依据奇异值阈值判断数值秩；该阈值采用机器精度默认值，不是用户拟合旋钮。给 raw 路线传入求解选项会报错。

```python
raw = FitSystem(data, representation="raw")
model = raw.solve()
```

## 如何使用外部求解器

公开数组是原始物理方程。外部求解器自行决定是否归一化。例如直接用 SciPy LSMR：

```python
from scipy.sparse.linalg import lsmr

A, f = raw.design_matrix, raw.forces
result = lsmr(A, f, atol=1e-10, btol=1e-10, maxiter=2000)
if result[1] not in (0, 1, 2, 4, 5):
    raise RuntimeError(f"external LSMR stopped with code {result[1]}")
parameters = result[0]
model = raw.force_constants(parameters)
print(raw.rmse(model))
```

这次直接调用没有 `raw.solve()` 内置的列归一化。接受解之前应检查外部求解器的停止状态。如果自行缩放，求解 $ASz\approx f$ 后须把 $Sz$ 传给 `force_constants()`，不要传入归一化坐标 $z$。参数向量须有限、长度为 `n_parameters`，且遵循原始参数列顺序。第 $p$ 阶力常数单位为 eV/Å$^p$，位移输入单位为 Å，力为 eV/Å。

normal 路线把 `normal_matrix`、`normal_rhs` 交给自己的对称方程求解器，再用 `system.force_constants(parameters)` 绑定物理解。此处保存的矩阵是 $H$。把 $(H,g)$ 再作为最小二乘问题会改变残差目标，不能称为使用原始力方程。

## 转换、合并与误差

- `raw.to_normal()` 返回同类型的正规系统；normal 的 `to_normal()` 返回自身。没有反向转换。
- `a + b` 累加 normal 统计量，或拼接 raw 方程。要求表示一致；用户负责保证物理参数布局一致，不检查模型几何或基的身份。混合合并前显式调用 `to_normal()`。
- `residual(model_or_parameters)`、`rmse(model_or_parameters)`、`relative_error(model_or_parameters)` 评估物理力误差。raw 直接算残差；normal 使用 $\theta^TH\theta-2\theta^Tg+c$，接近完美拟合时存在相消精度限制。
- FitSystem 不提供持久化 API。拟合结果用 ForceConstants.save() 保存；需要保存方程时，用户可自行存储公开数组。
- ASR 与旋转约束投影继续显式作用于求解得到的 `ForceConstants`。

删除 `FitData`、`FitSystem.from_atoms()`、`matrix`、`rhs`、`force_norm`、`column_scale` 和 `max_steps`。使用直接初始化、明确的数据属性及 `maxiter`。`solve()` 返回模型，`model.parameters()` 提取物理参数向量。
