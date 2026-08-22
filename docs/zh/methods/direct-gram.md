# Direct-Gram 与 matrix-free design 研究

本文记录 MLFCS 对直接计算

$$
G=X^T X,
\qquad
b=X^T y
$$

的独立研究。研究没有修改正式拟合路径、JAX design kernel、batch API、dense SYRK、Wick/Taylor 基底或 solver。

独立 prototype 位于 `research/direct_gram/prototype.py`，只用于复现小规模等价性和成本分析。

## 当前 design operator 的精确定义

设训练构型为 $s$，力坐标为 $r=(a,\alpha)$，物理参数为 $p$。对于 Taylor 阶数 $m$，每个 orbit $o$、对称 image $i$、平移 cell $t$、Cartesian 分量组合 $c=(c_1,\ldots,c_m)$ 和被省略的力轴 $j$，当前实现先生成

$$
q_{oitc}=(q_{oitc_1},\ldots,q_{oitc_m}),
$$

再从位移和经验协方差计算 leave-one-axis Wick 特征

$$
W^{(m-1)}_s(q_{oitc\setminus j}).
$$

`image_parameter_basis()` 将独立参数映射为对称 image 的 Cartesian tensor 分量，记为 $A_{oic,p}$。它包含：

- orbit representative 到独立参数的线性映射；
- Cartesian rotation；
- interaction component permutation；
- image 和 parameter mask；
- physical parameter index。

因此当前 design 元素是

$$
X_{s,rp}
=
-\frac{1}{m!}
\sum_{o,i,t,c,j}
W^{(m-1)}_s(q_{oitc\setminus j})
A_{oic,p}
\mathbf 1[r=q_{oitc_j}].
$$

不同 Taylor 阶数的 design block 在联合拟合中按列拼接。ASR 严格拟合时，physical parameter $θ$ 进一步写成

$$
\theta=Rz,
$$

其中 $R$ 是显式约束 null-space map；因此 reduced design 是 $XR$。

## Direct-Gram 的逐层展开

将一次 tile 中的所有静态项合并为系数映射 $H$，将构型相关的 Wick 项写为 $B_s$，形式上可写成

$$
X_s=B_sH.
$$

于是

$$
G=\sum_s H^T B_s^T B_s H,
\qquad
b=\sum_s H^T B_s^T y_s.
$$

这个等式是严格的，但不自动带来收益。若 $B_s$ 的 feature 数为 $K$，就需要计算 feature correlation

$$
C=\sum_s B_s^T B_s,
$$

其存储和计算通常是 $K^2$。如果不保存 $C$，就必须在同一力坐标上逐项计算 feature pair，复杂度变成

$$
O\left(\sum_r M_r^2\right),
$$

其中 $M_r$ 是该力坐标对应的有效 interaction contribution 数。这正是 Direct-Gram 候选重新出现 interaction-term 平方级计算的原因。

## 三个候选的结论

### 候选 A：interaction-term streaming

逐个 tile 或 interaction term 累积外积在数学上是 exact 的，但不同 tile 之间的 cross term 不能省略。若只累积 tile 对角块，得到的不是 $X^T X$；若补齐所有 tile pair，就需要保存或重复计算所有 tile，并且 FLOPs 接近 tile 数平方。

独立 prototype 在 Si 上计算了严格的 tile-pair Gram，并与显式 design 对照。结果如下：

| 案例 | 参数数 | force rows | tile 数 | $G$ 最大绝对误差 | $b$ 最大绝对误差 | tile-pair / 显式 Gram FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Si FC2，2 帧 | 11 | 384 | 3 | $1.33\times10^{-15}$ | $5.33\times10^{-15}$ | 0.79 |
| Si FC2+FC3，2 帧 | 106 | 384 | 12 | $1.33\times10^{-15}$ | $5.33\times10^{-15}$ | 1.29 |
| Si FC2+FC3+FC4，2 帧 | 574 | 384 | 257 | $1.33\times10^{-15}$ | $5.33\times10^{-15}$ | 190.99 |

FC4 以后 cross tile 项迅速主导成本，因此候选 A 不进入正式实现。

### 候选 B：feature factorization

候选 B 试图先计算 $C=B^T B$，再计算 $H^TCH$。它只有在 $C$ 能保持局部且远小于 design 时才可能有收益。

若按完整对称 Wick feature 空间估算，Si 的 $d=192$、最高 Wick degree 为 3 时 feature 数为

$$
\binom{192+3-1}{3}=1{,}198{,}144,
$$

对应的完整 $C$ 需要约 $10.4$ TiB；SnSe 的 $d=768$ 对应约 $41{,}797$ TiB。实际 interaction 支撑会减少 feature 数，但如果不构造全局 $C$，就会回到 force-row 内的 feature pair contraction，仍然是候选 A 的平方级问题。

因此，当前表示下没有找到既保持 exact、又避免 $K^2$、同时明显小于 design 的全局 feature factorization。候选 B 不进入正式实现。

### 候选 C：Taylor/Wick moment accumulation

FC2、FC3、FC4 的 force basis 最高分别是 Wick degree 1、2、3；Gram 的最高乘积 degree 为 6。因此 Taylor moment 路线至少需要六阶位移 moment。

对于 $d$ 个 Cartesian displacement，自由度完全对称的六阶 moment 数量为

$$
\binom{d+6-1}{6}.
$$

Si 的 $d=192$ 时约为 $7.52\times10^{10}$ 个分量，使用 float64 约 0.547 TiB；SnSe 的 $d=768$ 时约为 $2.91\times10^{14}$ 个分量，约 2114 TiB。Wick 乘积恒等式可以降低多项式 degree，但不能消除有限训练集中的 coordinate index 组合，也不能假设训练数据是严格 Gaussian。

所以 moment 路线只适合作为理论校验，不适合作为默认 exact Gram 算法。

## 训练规模基线

### Si

- FC2/FC3/FC4 参数数：11/95/468，总计 574；
- orbit 数：4/10/18；
- image 数：58/842/4426；
- translation cells：32；
- design tiles：FC2–FC4 联合为 257；
- 2 帧 force rows：384。

### SnSe

- 181 帧；
- 每帧 256 原子，即 768 个 force rows；
- physical/reduced 参数数约为 8572/5801；
- orbit 数：55/204/99；
- image 数：664/7280/6144；
- translation cells：32；
- design signatures：26；
- design tiles：476；
- 当前静态 design program 约 252.3 MiB；
- reduced dense Gram 约 256 MiB。

当前 SnSe profile 中主要耗时是 Wick、basis 和 interaction representation 到 design 的构造，而不是 Gram 的 dense SYRK。

## 约束与 batch streaming

ASR null-space 可以形式上提前进入 Direct-Gram：

$$
G_z=R^TGR,
\qquad
b_z=R^Tb.
$$

但当前 $R$ 虽然按约束连通分量构造，仍可能把多个 physical columns 混合。直接将它并入 interaction transform 会增加每个 term 的参数 contraction，并可能破坏静态 locality。没有测量证据表明提前 reduction 比当前先构造 physical design 再 reduction 更快，因此不改变当前约束流程。

无论采用哪种 exact 路线，configuration batch 都可以按

$$
G\leftarrow G+\Delta G(u_b),
\qquad
b\leftarrow b+\Delta b(u_b)
$$

进行 streaming；但这只解决训练集存储，不解决 interaction/feature pair 的平方级计算。`batch_size` 仍只影响执行批次，不改变数学结果。

## 最终结论

在当前 primitive-site、orbit、Cartesian tensor 和 Wick 参数化下：

- Direct-Gram 的等式成立；
- 小型 FC2 可以通过 tile-pair contraction 精确验证；
- FC3 开始 cross tile 成本超过显式 Gram；
- FC4 的 tile-pair 路线约为显式 Gram 的 191 倍；
- 全局 feature correlation 和六阶 moment 都需要不可承受的 $K^2$ 对象；
- 关闭 design matrix 并不会自动消除 interaction representation 和 Wick feature 的计算。

因此当前 Direct-Gram 研究结论为 **No-Go**：不修改正式拟合器，不增加 Direct-Gram API，不引入参数分块、自动 batch、moment 近似或新的运行时后端。研究 prototype 仅作为数学等价性和成本边界的回归工具保留。
