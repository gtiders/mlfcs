# FC2 Observable Closure 第三阶段：数学结构与架构边界

## 1. 结论摘要

本阶段继续使用 KCl、PolyMLP 和当前正式 transferable FC2 实现，但所有新计算仍位于
`research/fc2_observable_closure/`，没有修改正式源码。

结论分级如下：

- **Mathematical GO：确认。** ASR-constrained decomposition 完整、可辨识且数值稳定。
- **Architectural GO：暂不直接给出。** source reference 依赖和 observable basis 构造成本
  需要一个正式工程 prototype 验证。
- **Prototype Recommended：是。** 推荐实现最小的 source-specific harmonic residual
  prototype，不进入 canonical exact-$R$ IFC。
- **Production GO：未给出。** 本阶段明确禁止正式功能接入。

closure 的语义仍严格限定为：

> finite-supercell harmonic response not represented by the current transferable FC2 basis

它不是 long-range FC2、electrostatic FC2、missing exact-$R$ interaction 或唯一 real-space
tail。

## 2. 为什么维数是 13→11、4→2、9→9

令

$$
\mathcal H_{\rm SC}=T\oplus C,
$$

其中 $T=\operatorname{im}M$ 是 4 维 transferable observable span，$C=T^\perp$ 是
9 维无约束 closure。令 $A$ 为 observable ASR operator。实际 rank 为：

| 对象 | 维数 | restricted ASR rank | ASR kernel 维数 |
|---|---:|---:|---:|
| $T$ | 4 | 2 | 2 |
| $C$ | 9 | 2 | 7 |
| $\mathcal H_{\rm SC}$ | 13 | 2 | 11 |

因此 ASR 删除的两个 observable directions **不完全位于 $T$**，旧 closure 也**不天然
满足 ASR**。分别约束两个旧子空间只得到

$$
(T\cap\ker A)\oplus(C\cap\ker A),
$$

其维数为 $2+7=9$，比 $\dim\mathcal H_{\rm SC}^{\rm ASR}=11$ 少 2。缺少的两维由
$T$ 与 $C$ 中各自违反 ASR、但相加后 ASR violation 抵消的 mixed directions 构成。

这解释了为什么：

$$
\dim T:4\rightarrow2,
\qquad
\dim C:9\rightarrow9.
$$

constrained closure 不等于 $C\cap\ker A$。它是在 11 维允许空间内，相对于 2 维
$T_{\rm ASR}$ 重新定义的 9 维正交余空间。

旧 closure 和新 closure 只有 7 维交集。九个 principal angles 中七个为 FP64 数值零，
另外两个为 0.8481 和 1.2094 rad。这证明新 closure 不是旧 closure 的符号变化，而是发生
了两维 transferable/closure 重新分配。

因此三个流程中：

1. $(T\oplus C)\cap\ker A$ 给出正确的整体允许空间，但不直接给出唯一分块；
2. 先定义 $T_{\rm ASR}=T\cap\ker A$，再在 $\mathcal H_{\rm SC}^{\rm ASR}$ 中求正交余空间，
   能得到完整的 $2+9=11$ 分解；
3. 分别使用 $T\cap\ker A$ 和 $C\cap\ker A$ 只有 9 维，不完整。

正式定义必须采用第二种流程。

## 3. Transferable 参数的 4→2

当前 cutoff 下存在三个 FC2 orbit：

- primitive site 0 的 onsite orbit，1 个 invariant parameter；
- site 0–1 的最近邻 orbit，2 个 invariant parameters；
- primitive site 1 的 onsite orbit，1 个 invariant parameter。

按当前参数顺序写成 $\theta=(\theta_0,\theta_1,\theta_2,\theta_3)$，生产
`build_translational_constraints()` 的独立方程等价于

$$
\theta_0+4\theta_1+2\theta_2=0,
$$

$$
4\theta_1+2\theta_2+\theta_3=0.
$$

所以没有任何一个单独参数被“删除”。两个 onsite 参数由最近邻 tensor 的两个 invariant
组合决定。若取 $z=(\theta_1,\theta_2)$，可写成一个直观但非正交的 map：

$$
\theta=
\begin{pmatrix}
-4&-2\\
1&0\\
0&1\\
-4&-2
\end{pmatrix}z.
$$

正式 `explicit_constraint_null_space()` 使用 pivoted QR 生成另一组 reduced coordinates。
它与 prototype SVD null space 和上述简单 map 的 principal angle 都在 FP64 精度内为零，
约束 residual 也在数值零范围。因而 $z$ 不是“新的 FC2 参数”，只是原有 4 维
transferable parameter space 在 ASR 下的两维坐标。

将 $\theta=R_{\rm ASR}z$ 交给 `expand_primitive_parameters()` 后，site、exact-$R$
translation 和 sparse row 语义完全不变；ASR 只限制 tensor coefficient。

## 4. Metric 与算子兼容性

finite observable basis 在 compact coordinate Euclidean metric 下的正交残差为
$1.01\times10^{-15}$。展开成 full Hessian 后，其 Gram matrix 为

$$
B_{\rm full}^TB_{\rm full}=8I
$$

到 $5.83\times10^{-16}$ 的 off-diagonal norm。因子 8 正是 $2\times2\times2$
reference 的 primitive-cell 数量。因此本案例中 compact Euclidean metric 与 full Hessian
Frobenius metric 只差一个整体常数，closure orthogonality 具有明确的物理矩阵范数语义。

transferable 与 closure 的 compact-metric orthogonality residual 为
$3.57\times10^{-15}$，full Frobenius metric 下为 $2.96\times10^{-14}$。

raw compact space 上的 symmetry/permutation projector $P_{\rm sym}$ 与 ASR projector
$P_{\rm ASR}$ 并不交换：

$$
\|P_{\rm sym}P_{\rm ASR}-P_{\rm ASR}P_{\rm sym}\|_2=0.5.
$$

这不表示 symmetry 与 ASR 物理矛盾，而是说明两个正交投影的顺序应用不是交集投影。
primitive translation covariance 已内建于 compact 表示；point-group action 与 Hessian
permutation 共同定义 13 维 symmetry space；ASR 与其直接求交得到稳定的 11 维空间。
正式算法不能依赖反复顺序 projection。

## 5. Closure canonicality

canonical 的对象是子空间 projector

$$
P_C=N_{\rm ASR}N_{\rm ASR}^T,
$$

以及重建后的 source Hessian，而不是 SVD coefficient $\eta$。对任意正交 $Q$，

$$
N_{\rm ASR}\rightarrow N_{\rm ASR}Q,
\qquad
\eta\rightarrow Q^T\eta
$$

不改变物理 Hessian。随机旋转实验中 projector residual 为
$1.21\times10^{-15}$，坐标补偿后的 Hessian 相对误差为 $3.82\times10^{-16}$。

closure 对应九重零奇异子空间，SVD 的符号和内部旋转可随 LAPACK/backend 改变。因此不应
把 $\eta$ 保存成具有跨运行物理身份的参数，也没有必要人为 canonicalize 每一个 basis
vector。稳定数据对象应保存 source identity 和 reconstructed finite Hessian；basis 与
$\eta$ 只属于 fitting cache/diagnostics。

## 6. Reference-size sweep

固定同一 primitive 和 $4.4391158672$ Å transferable cutoff：

| reference | 原子数 | $\dim\mathcal H_{\rm SC}$ | $\dim\mathcal H_{\rm SC}^{\rm ASR}$ | $\dim T_{\rm ASR}$ | closure |
|---|---:|---:|---:|---:|---:|
| $2^3$ | 16 | 13 | 11 | 2 | 9 |
| $3^3$ | 54 | 24 | 22 | 2 | 20 |

transferable ASR space 保持两维，但 closure 随 source quotient 增大而从 9 维重组为 20 维。
PolyMLP 数值 Hessian 的分解为：

| reference | target norm | transferable norm | closure norm | closure ratio | closure norm/$\sqrt N$ |
|---|---:|---:|---:|---:|---:|
| $2^3$ | 14.437 | 13.760 | 4.370 | 0.3027 | 1.0926 |
| $3^3$ | 27.841 | 26.827 | 7.445 | 0.2674 | 1.0131 |

数值 Hessian 投影到 symmetry+ASR space 的相对 residual 分别为
$4.35\times10^{-9}$ 和 $6.30\times10^{-9}$，主要来自 $10^{-4}$ Å 数值差分。

closure ratio 随 reference 增大有所下降，但 closure dimension 增加且每 $\sqrt N$ norm
仍显著，不能宣称 closure 消失或已经收敛。小 reference 的 residue class 对应多个 exact-$R$
lift；closure 本身不提供选择某个 lift 的信息，因此不存在 canonical injection
$C_{\rm SC1}\rightarrow C_{\rm SC2}$。即使 $2^3$ 可整除 $4^3$，也只能在额外指定 real-space
分配规则后构造某种 lift，而那会引入 closure 原本没有的物理语义。

$4^3$ 有 128 个原子、2304 维 raw compact space。当前逐列 dense projector 原型需要
40.5 MiB projector，并按 $2^3/3^3$ 实测增长预计耗时数分钟；本阶段没有将它 materialize。
这不是数学限制，而是说明正式 prototype 必须从有限 pair orbit 直接生成允许 basis，不能
复制当前研究脚本的 dense projector。

## 7. Cutoff sweep

固定 $2^3$ reference：

| cutoff (Å) | transferable/ASR | closure dim | target closure ratio | condition | 状态 |
|---:|---:|---:|---:|---:|---|
| 2.5 | 2/0 | 11 | 1.0000 | 1.13 | 可用 |
| 3.2–4.439 | 4/2 | 9 | 0.3027 | 4.02 | 可用 |
| 4.8 | 10/8 | 3 | 0.0718 | 7.18 | 可用 |
| 5.5 | 12/10，rank 9 | — | — | — | alias，拒绝 |
| 6.0 | 12/10，rank 9 | — | — | — | alias，拒绝 |

cutoff 增大时，closure dimension 与实际 target closure norm 都系统性下降；二者不是同一
指标。$4.8$ Å 时仍存在 3 维 closure，但目标只在其中留下 7.18% Frobenius norm。
继续增大 cutoff 在该 source reference 上先触发 transferable alias，而不是达到 closure=0。
因此 regularization 不能替代更大 reference，也不能用 closure 掩盖 kernel。

closure 的 compact pair-distance norm 分布在 0、3.146、4.449、5.449 和 6.292 Å 等
finite distance block 上均有贡献。这只是 source Hessian residual 的分布，不是唯一
exact-$R$ tail。

## 8. Dataset robustness 与噪声

扫描 3 个 seed、4 个 frame 数和 3 个位移尺度，共 36 组数据，全部为 11/11 满列秩。
condition number 范围为 3.95–5.51。100 帧时：

| $\sigma$ (Å) | smallest singular value | condition | closure coefficient norm | RMSE (eV/Å) |
|---:|---:|---:|---:|---:|
| 0.003 | 0.0793–0.0807 | 3.95–4.03 | 1.544–1.546 | $(5.49$–$5.93)\times10^{-5}$ |
| 0.01 | 0.264–0.269 | 3.95–4.03 | 1.542–1.548 | $(6.11$–$6.59)\times10^{-4}$ |
| 0.03 | 0.793–0.807 | 3.95–4.03 | 1.536–1.554 | $(5.61$–$6.03)\times10^{-3}$ |

RMSE 随位移增大来自 FC2 无法表示 PolyMLP 非谐响应，不是 rank 恶化。即使 10 帧，所有
seed/尺度仍满列秩。

100 帧、$0.01$ Å 数据的 maximum covariance-proxy diagonal 为 13.78，maximum absolute
parameter correlation 为 0.613；归一化 transferable/closure cross-block 2-norm 为 0.0563。
加入 $10^{-8}$ 到 $10^{-4}$ eV/Å Gaussian force noise 后，relative parameter change 从
$3.82\times10^{-8}$ 线性增长到 $3.82\times10^{-4}$，没有异常放大。代数满秩在该数据上
也对应可用的数值 conditioning。

## 9. 架构候选比较

### 方案 I：塞入 `OrderParameterization`

不推荐。当前 `OrderParameterization` 保存 orbit images、Cartesian rotations、component
permutations 和 exact interaction coordinates。closure 没有 exact-$R$ identity，也没有
跨 reference 的 orbit 语义。把它伪装成普通 FC2 block 会污染参数与 HDF5 语义。

### 方案 II：finite harmonic observable layer

数学上最清楚。transferable realization 和 source-specific closure 在同一个 ASR-allowed
finite Hessian space内联合参数化，fitting 只看到 reduced coordinates。它能正确保留
两组参数的 cross Gram，并避免先后拟合偏差。

### 方案 III：拟合后独立 residual Hessian

若“独立”表示先拟合 transferable、再拟合残差，则不是联合最小二乘，结果依赖顺序，
不能保证 transferable/closure 的正交 metric 和 cross correlation。只有将两块同时交给
同一个 design/Gram solve 时才与方案 II 等价。

因此推荐 **方案 II 的物理模型，加上方案 III 的 source-specific ownership**。

## 10. 与当前源码的最小映射

当前调用关系表明：

- `PrimitiveInteractionSpace`、`InteractionSpace` 和 `OrderParameterization` 继续只描述
  transferable exact-$R$ interactions，不修改。
- `build_translational_constraints()` 与 `explicit_constraint_null_space()` 继续生成
  transferable ASR map，可直接复用。
- `FitDataset` 已提供固定 reference 的 displacement/force arrays，无需修改。
- `_StreamingGramSystem`、column scaling 和 linear solver 只依赖 design columns，可复用。
- `ForceDesignOperator` 的 kernel builder 当前只接受 orbit-based `OrderParameterization`，
  不能直接接受任意 finite Hessian basis。
- `expand_primitive_parameters()`、`SparseOrderForceConstants`、`ForceConstants` 和
  `realize_force_constants()` 必须保持 canonical exact-$R$ 语义，不能承载 closure。

推荐下一阶段只做下列 prototype：

1. 在 force-constant physical layer 新增内部 `FiniteHarmonicResponse`，拥有 source
   `StructureRelation` fingerprint 和 reconstructed compact/full Hessian；不拥有可迁移
   exact-$R$ rows。
2. 新增内部 `FiniteObservableSpace` builder，拥有 symmetry+ASR basis、transferable map
   和 closure projector。basis 可缓存，但 SVD coefficient 不作为稳定 metadata。
3. fitting 内新增轻量 `FiniteHarmonicDesignBlock`，按 $F=-\Phi u$ 对 batch 产生 closure
   columns；它与现有 orbit design columns 在同一 batch 中拼接，再交给现有 Gram/solver。
   不复制 Wick、Gram、preconditioner 或 least-squares engine。
4. fitter 输出继续包含 canonical transferable `ForceConstants`，另带一个 source-only
   `FiniteHarmonicResponse`。总 source Hessian 的 ownership 属于后者或一个明确的组合结果，
   不能静默写回 `sparse[2]`。
5. SCPH/SSCHA 只有在 source reference 完全匹配时才能显式消费 total effective Hessian；
   target-supercell realization、通用 HDF5 v3 和外部 writer 默认拒绝 closure。

正式 prototype 需要扩展 `ForceDesignOperator` 的 design-block protocol 和 fitter result，
但不需要修改 interaction enumeration、Wick basis、Gram 数值核、solver、canonical IFC 或
writer。边界局部且可以独立回滚。

## 11. 必须拒绝的退化情况

- $\ker M\neq0$：继续抛出 `InteractionAliasingError`。
- $\ker M_{\rm ASR}\neq0$：拒绝 closure 拟合。
- closure dimension 为 0：严格退化为现有 transferable FC2。
- joint dataset rank deficient：拒绝，不使用 ridge/Lasso 猜解。
- source structure fingerprint 不匹配：禁止解释、应用或导出 residual Hessian。

## 12. 最终判定

本阶段确认数学结构、metric、constraint 顺序、target decomposition 和采样 conditioning
均健康；同时证明 closure 是 source-supercell-specific residual representation，而不是
一种新的 transferable IFC parameterization。

因此最终判定为：

$$
\boxed{\text{Mathematical GO + Prototype Recommended}}
$$

尚不提升为 Architectural GO，因为高效 finite observable basis builder、composite design
接口和 source ownership 尚未在正式边界内做最小 prototype。Production GO 本轮禁止。
