---
title: FC2 有限超胞 Observable Closure 最小调研
audience:
  - developer
status: research
code_verified: 4.0.0a4
examples:
  - research/fc2_observable_closure
---

# FC2 有限超胞 Observable Closure 最小调研

本研究只回答当前 transferable FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

是否存在

$$
\dim\ker M=0,
\qquad
\dim\operatorname{im}M<\dim\mathcal H_{\mathrm{SC}}.
$$

结论分为两层：KCl 在 **representation 层确实命中 sweet spot**，但计划规定的
去质心实际数据不能辨识完整 transferable + closure 联合空间，因此本轮总体结论为
**No-Go**，不进入正式架构设计。

## 有限 observable 空间的定义

案例使用 KCl 的 2 原子 primitive 和 $2\times2\times2$、16 原子 reference。
有限 FC2 先写成 translation-reduced compact Hessian：

$$
\Phi_{ab}([R])\in\mathbb R^{3\times3},
$$

其中 $[R]$ 属于 reference 的有限平移商群。该坐标天然满足 primitive translation
covariance，再通过有限 reference-compatible 空间群平均和 Hessian permutation

$$
\Phi_{ab}([R])
=
\Phi_{ba}([-R])^T
$$

得到正交投影 $P_{\rm SC}$。其 image 定义本研究中的
$\mathcal H_{\rm SC}$。本轮没有施加 ASR、Born–Huang 或 Huang 条件。

投影器同时验证：

$$
\|P_{\rm SC}^2-P_{\rm SC}\|<10^{-9},
\qquad
\|P_{\rm SC}-P_{\rm SC}^T\|<10^{-9}.
$$

这项检查用于及时发现 atom permutation、Cartesian rotation 或主动/被动变换
convention 错误。

## Transferable realization map

当前 `PrimitiveInteractionSpace` 的每个 FC2 独立参数逐列展开成 exact-$R$ tensor，
再 folding 到 KCl reference，最后投影到 $\mathcal H_{\rm SC}$ 的正交坐标。结果为：

| 指标 | 数值 |
|---|---:|
| primitive transferable 参数维数 | 4 |
| $\operatorname{rank}(M)$ | 4 |
| $\dim\ker M$ | 0 |
| $\dim\mathcal H_{\rm SC}$ | 13 |
| closure dimension | 9 |

$M$ 的奇异值为

$$
(4.89897949,\ 3.46410162,\ 1.73205081,\ 1.73205081),
$$

rank tolerance 为 $1.41\times10^{-14}$，不存在接近阈值的模糊奇异值。transferable
列投影到 observable basis 后的最大残差为 $2.01\times10^{-15}$。

因此这个真实案例严格满足：

$$
\dim\ker M=0,
\qquad
\dim\operatorname{im}M=4<13.
$$

这证明“当前可迁移 FC2 本身可辨识，但没有覆盖完整有限 Hessian”不是空想情形。

## 正交 closure

只使用一次 SVD，将 $M$ 的左正交余空间取为 $N$。数值结果为：

| 检查 | 数值 |
|---|---:|
| $\operatorname{rank}[M\ N]$ | 13/13 |
| $\|M^TN\|_2$ | $1.07\times10^{-32}$ |
| 最小 representation principal angle | $\pi/2$ |
| 随机 observable 坐标相对重建误差 | $1.51\times10^{-16}$ |
| 随机 full Hessian 相对重建误差 | $1.57\times10^{-16}$ |

所以在纯 representation 层，任意
$\phi\in\mathcal H_{\rm SC}$ 都可以唯一写成

$$
\phi=M\theta+N\eta.
$$

这里的 $\eta$ 只表示 current transferable FC2 basis 未覆盖的 finite-supercell
harmonic response。它不对应唯一的无限晶格 interaction，也不是 long-range FC2。

## Dataset identifiability

实际数据由案例 PolyMLP 计算：100 个 Gaussian Cartesian snapshots，标准差
$0.01$ Å，随机种子 42，每帧去除质心位移。由

$$
F_s=-H(\phi)u_s
$$

直接构造 design：

$$
X=
\left[
X_{\rm SC}M\quad X_{\rm SC}N
\right].
$$

结果为：

| 指标 | 数值 |
|---|---:|
| transferable dataset rank | 4/4 |
| closure dataset rank | 9/9 |
| joint dataset rank | 12/13 |
| joint nullity | 1 |
| 非零子空间 condition number | 6.26 |
| transferable/closure 最小 dataset principal angle | 0 |

联合 null vector 的 design residual 为 $1.37\times10^{-16}$，但对应 Hessian 的最大
ASR residual 为 $2.10$。这说明缺失方向不是 representation completion 的数值误差，
而是去质心位移无法观测的均匀平移 sector。transferable 与 closure 各自满秩，组合后
却在当前数据分布上共享一个不可区分方向。

只使用 transferable FC2 的 force RMSE 为 $5.87\times10^{-3}$ eV/Å；加入 closure
后的最小范数拟合 RMSE 降至 $6.46\times10^{-4}$ eV/Å。但更小的残差不能消除联合
parameter gauge，因此不能作为接入正式架构的理由。

## Aliasing 负对照

现有单原子、$4.1$ Å cutoff、$1\times1\times1$ reference 模型包含 3 个 primitive
参数，而 realization rank 只有 1：

$$
\dim\ker M=2.
$$

正式 `validate_realization_identifiability()` 正确抛出
`InteractionAliasingError`。本 prototype 没有关闭或绕过该生产检查，也没有用 closure
掩盖 primitive aliasing。

## 结论

本研究确认两点：

1. representation sweet spot 真实存在，且 SVD complement 在数值上非常稳定；
2. 未施加 ASR 的 full observable closure 与去质心实际数据并不联合可辨识。

按照预先锁定的验收条件，本轮结论为 **No-Go**。不修改正式 fitter、IFC schema、
SCPH、SSCHA 或 export。若未来重新研究，必须另立阶段讨论 ASR-constrained observable
space；不能把本轮 13 维无约束结果直接接入拟合器。

完整 prototype 和机器可读结果位于
`research/fc2_observable_closure/`。
