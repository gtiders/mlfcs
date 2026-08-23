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

## 第三阶段：结构与架构边界

第三阶段解释了 $13\rightarrow11$、$4\rightarrow2$、$9\rightarrow9$ 的来源。ASR 在
transferable span 和旧 closure 上的 restricted rank 都是 2，因此：

$$
\dim(T\cap\ker A)=2,
\qquad
\dim(C\cap\ker A)=7.
$$

分别约束旧分块只能得到 9 维，而整个允许空间为 11 维。缺少的两维是 $T$ 与 $C$ 的
ASR violation 相互抵消的 mixed directions。旧 closure 与新 closure 只有 7 维交集，
另外两个 principal angles 为 0.848 和 1.209 rad。因此必须先构造
$\mathcal H_{\rm SC}^{\rm ASR}$，再在其中重建 closure，不能分别投影旧分块。

当前四个 transferable 参数的两个独立 ASR 方程为

$$
\theta_0+4\theta_1+2\theta_2=0,
$$

$$
4\theta_1+2\theta_2+\theta_3=0.
$$

所以 4→2 是多个 orbit parameter 的线性约束，不是删除两个独立物理 IFC。正式
pivoted-QR map、prototype SVD map 和一个显式简单 map 张成相同 null space；exact-$R$
site/translation 语义保持不变。

compact observable Euclidean metric 展开到 full Hessian 后恰为 8 倍 Frobenius metric。
但是 symmetry/permutation projector 与 raw ASR projector 不交换，commutator 2-norm 为
0.5，因此应直接构造 subspace intersection，而不是顺序投影。

### Reference 和 cutoff

固定 $4.4391$ Å cutoff：

| reference | $\dim\mathcal H^{\rm ASR}_{\rm SC}$ | transferable | closure | target closure ratio |
|---|---:|---:|---:|---:|
| $2^3$ | 11 | 2 | 9 | 0.3027 |
| $3^3$ | 22 | 2 | 20 | 0.2674 |

closure dimension 随 source quotient 增长，并不存在从小 reference closure 到大 reference
closure 的 canonical injection。小胞 residue class 具有多个 exact-$R$ lifts，而 closure
没有提供选择某个 lift 的物理信息。$4^3$ 没有使用当前逐列 dense projector 构造：该路径
需要 2304 维 dense projector，实测缩放表明不适合作为未来正式算法。

固定 $2^3$ reference 时，cutoff 从 2.5 Å 增至 4.8 Å，closure dimension 从 11 降到 3，
target closure norm ratio 从 1.0 降到 0.0718。到 5.5 Å 时 constrained transferable
space 已出现一维 kernel，并被正式 alias check 拒绝。closure 不得掩盖该 kernel。

### 数据与 canonicality

3 个 seeds、4 个 frame 数和 3 个位移尺度组成的 36 组数据全部为 11/11 满秩，condition
number 为 3.95–5.51。$10^{-4}$ eV/Å force noise 只产生 $3.82\times10^{-4}$ 的相对
parameter change，没有异常噪声放大。

SVD coefficient $\eta$ 不是 canonical parameter。任意
$N\rightarrow NQ$、$\eta\rightarrow Q^T\eta$ 都表示同一 Hessian。随机 basis rotation
下 closure projector residual 为 $1.21\times10^{-15}$，Hessian residual 为
$3.82\times10^{-16}$。稳定对象应是 source fingerprint、closure projector 和重建后的
finite Hessian，而不是某个 backend 产生的 $\eta$。

### 架构判定

closure 不应进入 `OrderParameterization`、`SparseOrderForceConstants` 或 canonical HDF5。
推荐下一阶段研究 source-specific `FiniteHarmonicResponse` 与轻量 harmonic design block：
它和现有 orbit design 联合产生 columns，并复用现有 streamed Gram 和 solver。transferable
`ForceConstants` 保持不变，source residual 不能 realization 或导出到不同 reference。

第三阶段判定为：

$$
\boxed{\text{Mathematical GO + Prototype Recommended}}
$$

尚不提升为 Architectural GO；Production GO 本轮明确未给出。详细数据和推导见
`research/fc2_observable_closure/phase3-report.md` 与 `results-phase3.json`。

## 第四阶段：Minimal Architecture Prototype

第四阶段把 dense projector 原型替换为有限 pair-label 群轨道构造。每个 finite pair orbit
只求一次 $3\times3$ tensor stabilizer invariant，随后在生成的 observable coordinates 中
施加 ASR。该算法直接生成允许的列空间，不枚举 dense projector 的全部列。

| KCl reference | finite pair orbits | observable dimension | ASR dimension | closure dimension | 构造时间 | 原型新增内存峰值 |
|---|---:|---:|---:|---:|---:|---:|
| $2^3$ | 8 | 13 | 11 | 9 | 0.22 s | 0.11 MiB |
| $3^3$ | 12 | 24 | 22 | 20 | 0.79 s | 0.31 MiB |
| $4^3$ | 22 | 52 | 50 | 48 | 1.88 s | 0.79 MiB |

$2^3$ 和 $3^3$ 的新列空间与旧 dense 参考 projector 的相对差异均小于
$9\times10^{-16}$。原型没有把 closure 接入 canonical IFC，而是定义统一的内部
`DesignBlock`：现有 orbit design 和 source-only finite harmonic design 在同一个 batch 中
共同进入现有 streamed Gram 和 solver。

KCl $2^3$ 的端到端联合 design 为 11/11 满列秩。Gram 和 RHS 相对差异分别为
$4.30\times10^{-16}$ 和 $5.48\times10^{-16}$；总 Hessian 相对第三阶段 dense 参考的差异为
$4.26\times10^{-13}$，ASR 最大残差为 $1.87\times10^{-15}$。拟合 RMSE 为
$6.46\times10^{-4}$ eV/Å。

source ownership 通过 primitive fingerprint 与 HNF translation sublattice 定义。原子重排和
同一子晶格的整数幺模换基往返误差均为零，不同 source supercell 被拒绝。closure dimension
为零时不创建 design block；transferable alias 仍由原有检查拒绝；联合数据秩亏在求解前拒绝。

因此第四阶段结论为

$$
\boxed{\text{Architectural GO, but not Production GO}}.
$$

它只证明最小架构边界可行，尚未批准公共 API 或正式源码接入。原型、报告和机器结果分别位于
`architecture_prototype.py`、`phase4-report.md` 和 `results-phase4.json`。
