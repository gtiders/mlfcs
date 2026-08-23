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

本研究考察 transferable FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

能否在 KCl 的 $2\times2\times2$ reference 上，以一个严格正交的 source-only closure
补全有限超胞谐响应。closure 的语义仅为：

> finite-supercell harmonic response not represented by the current transferable FC2 basis

它不是 long-range FC2，也不对应唯一的无限晶格 interaction。

## 第一阶段：无约束 observable space

有限 FC2 使用 translation-reduced compact Hessian block

$$
\Phi_{ab}([R])\in\mathbb R^{3\times3}
$$

表示，并施加 reference-compatible 空间群与 Hessian permutation。得到：

| 指标 | 数值 |
|---|---:|
| primitive transferable 参数维数 | 4 |
| $\operatorname{rank}(M)$ | 4 |
| $\dim\ker M$ | 0 |
| $\dim\mathcal H_{\mathrm{SC}}$ | 13 |
| closure dimension | 9 |

representation sweet spot 确实存在；SVD closure 与 transferable span 正交，且
$[M\ N]$ 为 13/13 满秩。然而，对 100 个标准差 $0.01$ Å、随机种子 42、逐帧移除
质心位移的 Gaussian snapshots，联合 design 只有 12/13 列秩。

第一阶段的唯一 null vector 满足：

| 诊断 | 数值 |
|---|---:|
| centered design residual | $1.37\times10^{-16}$ |
| Hessian ASR maximum | $2.10$ |
| uniform-displacement force norm | $14.56$ |
| 投影到 ASR-allowed space 的相对范数 | $1.31\times10^{-15}$ |

所以它不是 transferable/closure completion 的数值退化，而是去质心采样无法观测、
同时又被 ASR 禁止的均匀平移响应。

## 第二阶段：ASR 直接进入表示空间

本阶段不做事后修正，而是定义

$$
\mathcal H_{\mathrm{SC}}^{\mathrm{ASR}}=\ker C_{\mathrm{ASR}}.
$$

observable Hessian basis 和现有 production transferable parameterization 分别求 ASR
null space，再在约束后的坐标中重建 realization map：

$$
M_{\mathrm{ASR}}=Z_{\mathrm{SC}}^T M Z_\theta.
$$

没有假定 ASR 只删除一维。实际 $C_{\mathrm{ASR}}$ 的 rank 为 2，因此：

| 指标 | 数值 |
|---|---:|
| observable dimension | $13\rightarrow11$ |
| transferable dimension | $4\rightarrow2$ |
| $\operatorname{rank}(M_{\mathrm{ASR}})$ | 2 |
| $\dim\ker M_{\mathrm{ASR}}$ | 0 |
| ASR-constrained closure dimension | 9 |
| $\operatorname{rank}[M_{\mathrm{ASR}}\ N_{\mathrm{ASR}}]$ | 11/11 |
| $\|M_{\mathrm{ASR}}^TN_{\mathrm{ASR}}\|_2$ | $1.02\times10^{-15}$ |
| 最小 representation principal angle | $\pi/2$ |

现有生产 ASR 参数空间与直接 Hessian ASR null space 的最大 principal angle 为
$4.48\times10^{-16}$。随机 ASR-allowed Hessian 的相对重建误差为
$4.39\times10^{-16}$，ASR、permutation 和 symmetry residual 分别不超过
$7.58\times10^{-15}$、$1.08\times10^{-15}$ 和 $8.01\times10^{-15}$。

将旧 closure 先投影到 ASR space 的 Flow A 在本案例中与重新构造的 Flow B 张成相同
9 维子空间，最大 principal angle 为 $2.27\times10^{-15}$。正式数学定义仍应采用 Flow B，
因为它直接在物理允许空间中构造 complement，而不依赖无约束 gauge。

## 四组数据秩对照

相同随机位移分别保留或移除质心，并分别使用无约束或 ASR-constrained basis：

| 位移处理 | 表示空间 | design rank | nullity |
|---|---|---:|---:|
| 移除质心 | 无约束 | 12/13 | 1 |
| 保留质心 | 无约束 | 13/13 | 0 |
| 移除质心 | ASR constrained | 11/11 | 0 |
| 保留质心 | ASR constrained | 11/11 | 0 |

ASR-constrained 的 centered 与 uncentered 奇异值逐项一致。这符合物理预期：满足 ASR
后，均匀平移不产生力，是否在输入位移中移除质心不再改变 design 的可辨识内容。

约束后 transferable、closure 和 joint rank 分别为 2/2、9/9 和 11/11；joint
condition number 为 4.02，最小 block principal angle 为 1.491 rad，没有近奇异迹象。

## 实际力拟合

| 模型 | rank | RMSE (eV/Å) | ASR maximum | 唯一解 |
|---|---:|---:|---:|---:|
| A. transferable only | 4/4 | $5.87\times10^{-3}$ | $7.94\times10^{-1}$ | 是 |
| B. transferable + unconstrained closure | 12/13 | $6.46\times10^{-4}$ | $9.76\times10^{-2}$ | 否 |
| C. ASR transferable + closure | 11/11 | $6.46\times10^{-4}$ | $1.27\times10^{-14}$ | 是 |

模型 C 保留了 B 的 force reconstruction 精度，同时删除了 gauge 并严格满足 ASR。
closure Hessian norm 占联合 Hessian norm 的 0.303，但这只能解释为当前表示的 residual
比例，不能解释为长程力占比。

## 结论与边界

第二阶段结论为 **GO**：该真实案例中的 ASR-constrained closure 数学上闭合、与
transferable span 可稳定分离，并可由去质心数据唯一辨识。这只表示最小原型允许进入
下一阶段架构讨论；本研究没有修改正式 fitter、IFC schema、SCPH、SSCHA 或 export。

单原子 aliasing 负对照仍为 3 个 primitive 参数、realization rank 1、kernel 维数 2，
并被生产检查正确拒绝。closure 不得用于掩盖 transferable representation 自身的 kernel。

完整原型和机器可读数值位于 `research/fc2_observable_closure/`。
