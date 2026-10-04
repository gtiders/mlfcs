# 力拟合 API

统一的 `FitSystem` 保存一个仅使用力的最小二乘问题。直接传入 `ClusterMap` 和带有已保存力的 ASE 结构。默认 `representation="normal"` 保存正规方程并使用 MINRES；`representation="raw"` 保存原始方程并使用 LSMR。内置算法仅这两种，`solve()` 返回 `ForceConstants`。

## 初始化与求解

```python
from ase.io import iread, read
from mlfcs import ClusterMap, ClusterSpace, FitSystem

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
)
mapping = ClusterMap(space, read("SPOSCAR"))
system = FitSystem(mapping, iread("train.extxyz", index=":"))
model = system.solve(rtol=1e-8, maxiter=1000)
print(system.rmse(model), "eV/Å")
model.save("fit.mlfcs")
```

超胞整数矩阵可以显式传给 `ClusterMap`，省略时自动推断。`FitSystem` 在准备内部力设计时检查超胞能否区分参数。每帧必须保持参考超胞的原子顺序、周期性和晶胞，且 ASE calculator 中已经保存有限的力。构造系统不调用 calculator。帧迭代器只消费一次；构造第二个系统时须创建新迭代器。

## 我们对数据做了什么

每帧位置与参考超胞比较，通过笛卡尔最近周期像得到位移 $u_i$，再生成对称约化 Taylor 力设计矩阵 $A_i$。力按原子顺序展开，每个原子的 x/y/z 分量连续。列遵循 `cluster_space.parameter_offsets` 以及阶数、orbit、component 的参数布局。Taylor 负号、阶乘、张量基与周期像求和已经计入 $A_i$。

构造过程中不减去平均力，不施加逐帧权重，也不归一化数据。两种表示都对应物理参数 $\theta$ 和同一个问题：

$$
\min_\theta\|A\theta-f\|_2^2.
$$

| 表示 | 可访问的数据 | 算法 | 保存规模 |
|---|---|---|---|
| `raw` | `design_matrix`：$A$；`forces`：$f$ | LSMR | $O(mn)$ |
| `normal` | `normal_matrix`：$H$；`normal_rhs`：$g$ | MINRES | $O(n^2)$ |

$m$ 是全部笛卡尔力方程数，$n$ 是参数数。raw 按帧输入顺序拼接原始方程。normal 逐帧累积：

$$
H=\sum_i A_i^TA_i,\qquad g=\sum_i A_i^Tf_i,\qquad c=\sum_i f_i^Tf_i.
$$

两条路线都保存 `force_squared_norm`：$c$。其他共同属性为 `n_structures`、`n_equations`、`n_parameters`、`representation` 和 `cluster_space`。公开数组只读、保持物理尺度，求解不修改它们。访问另一种表示的专属数据会报 `ValueError`。normal 不保留原始行，不能恢复逐帧残差。

raw 构造时会暂存帧矩阵以便拼接，求解时会申请归一化工作矩阵，需要考虑最终矩阵以外的临时内存。normal 不保留各帧设计矩阵，但参数平方规模的矩阵仍可能很大。满列秩时 $\kappa_2(A^TA)=\kappa_2(A)^2$，病态数据形成正规方程时可能损失精度。

## 归一化属于求解器

`fitting/solve.py` 中的内部 `FitSolver` 类实现两种算法，由 `system.solve()` 调用。它始终进行参数列归一化，算法由方程表示固定决定。

raw 路线定义：

$$
s_j=\frac{1}{\|A_{:j}\|_2},\qquad S=\operatorname{diag}(s_j).
$$

LSMR 实际求解 $ASz\approx f$，随后还原物理参数 $\theta=Sz$。力向量不缩放。实现先按列最大绝对值缩放，再计算列范数，避免直接平方极大或极小的物理系数。归一化矩阵只存在于求解期间，公开的 $A$、$f$ 保持原值。

normal 路线取 $s_j=1/\sqrt{H_{jj}}$，MINRES 求解：

$$
SHSz=Sg,\qquad \theta=Sz.
$$

这保留当前列缩放迭代的数值逻辑。缩放平衡不同 FC 阶数的参数列，但无法补足缺失观测或消除线性相关。几何 folded exact rank 与训练方程的数值秩是两件事。零列报 `UnobservedParameterError`。秩亏问题在缩放坐标中求解，所选解不一定最小化物理 $\theta$ 的范数。

| 路线 | 默认参数 | 含义 |
|---|---|---|
| normal / MINRES | `rtol=1e-8`、`maxiter=1000` | MINRES 相对停止容差、迭代上限 |
| raw / LSMR | `atol=1e-8`、`btol=1e-8`、`conlim=1e8`、`maxiter=1000` | 矩阵/最小二乘停止容差、力向量停止容差、缩放后条件数上限、迭代上限 |

LSMR 对相容方程大致检查 $\|r\|\le\texttt{atol}\|AS\|\|z\|+\texttt{btol}\|f\|$，最小二乘检查控制 $\|(AS)^Tr\|$。`conlim=0` 关闭条件数上限检查。容差无量纲，针对归一化方程。达到迭代上限、条件数上限或未收敛会报错。传入另一算法的参数、`solver=` 或额外算法均不支持。完整停止规则见 [SciPy MINRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html) 与 [SciPy LSMR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)。

```python
raw = FitSystem(mapping, iread("train.extxyz", index=":"), representation="raw")
model = raw.solve(atol=1e-10, btol=1e-10, conlim=1e8, maxiter=2000)
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
