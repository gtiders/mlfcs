# 有限差分 signed displacement 对称轨道约化

## 1. 目标与边界

MLFCS 已经先对 interaction cluster、空间群 image 和 Cartesian tensor 分量进行轨道约化。这里研究的是更靠后的问题：对于一个已经选出的独立位移键，中心差分仍需要的正负位移构型能否继续通过严格晶体对称性减少。

该约化必须保持当前有限差分算子完全不变，不使用以下近似：

- 不假设 $F(-u)=-F(u)$；
- 不假设势能关于任意原子位移为偶函数；
- 不把数值上接近的位移视为相同；
- 不用张量分量对称性替代构型空间中的真实空间群映射。

## 2. 当前中心差分算子

对 $n$ 阶 IFC，需要对力求 $m=n-1$ 次位移导数。给定位移自由度

$$
K=(e_1,e_2,\ldots,e_m),
$$

其中 $e_k$ 是某个超胞原子和 Cartesian 方向对应的 $3N$ 维单位向量。符号向量为

$$
s=(s_1,\ldots,s_m),\qquad s_k\in\{-1,+1\}.
$$

实际位移为

$$
u_{K,s}=h\sum_{k=1}^{m}s_ke_k.
$$

当前递归中心差分为

$$
\mathcal D_KF
=
\frac{1}{(2h)^m}
\sum_{s\in\{-1,+1\}^m}
\chi(s)F(u_{K,s}),
$$

其中

$$
\chi(s)=\prod_{k=1}^{m}s_k.
$$

因此，每个已经约化的 displacement key 仍生成 $2^{n-1}$ 个 signed configurations。

当若干 $e_k$ 相同时，不同 $s$ 可能产生完全相同的 $u_{K,s}$。例如二次求同一自由度的导数得到

$$
\frac{F(2he)-2F(0)+F(-2he)}{4h^2}.
$$

当前实现会重复计算两个相同的零位移构型。把完全相同的位移先合并，并把权重相加，是无需使用晶体对称性的严格优化。

## 3. 空间群在位移和力上的作用

设 reference supercell 的一个空间群操作在 $3N$ 维 Cartesian 空间中的正交表示为 $Q_g$。它同时包含：

- 超胞原子 permutation；
- Cartesian 旋转矩阵。

对保持 reference 晶体不变的势能，有

$$
U(Q_gu)=U(u).
$$

对位移求梯度得到力的协变关系

$$
F(Q_gu)=Q_gF(u).
$$

这条关系才是从一个已计算 signed configuration 重建另一个 configuration 的物理依据。

空间反演只有在它确实属于 reference 的空间群并正确置换原子时才能使用。不能单独假设

$$
F(-u)=-F(u).
$$

## 4. signed configuration 上的严格群作用

若某个空间群操作满足

$$
Q_ge_k=\epsilon_k e_{\pi(k)},
\qquad \epsilon_k\in\{-1,+1\},
$$

其中 $\pi$ 是同一 displacement key 中导数轴的 permutation，则

$$
Q_gu_{K,s}=u_{K,s'},
$$

且

$$
s'_{\pi(k)}=\epsilon_ks_k.
$$

于是

$$
F(u_{K,s'})=Q_gF(u_{K,s}).
$$

中心差分权重满足

$$
\chi(s')=\left(\prod_k\epsilon_k\right)\chi(s),
$$

因为 permutation 不改变符号乘积。

只有满足上述 signed-monomial 条件的操作才能在现有 Cartesian stencil 内建立闭合映射。一般晶体旋转可能把 $x$ 方向变成多个 Cartesian 方向的线性组合；这种操作虽然对 IFC 张量约化有效，却不会把当前轴向有限差分构型映射到另一个已有构型，因而不能用于减少 signed calculations。

## 5. 应采用的全局构型轨道

约化对象不应仅限于单个 displacement key 内的符号向量。空间群可能把一个 key 的构型映射到另一个已计划 key 的构型。因此应对整个 displacement plan 定义作用：

$$
g\cdot(K,s)=(K',s'),
$$

前提是

$$
Q_gu_{K,s}=u_{K',s'}
$$

严格成立。

对所有 planned configurations 按该作用划分 orbit。每个 orbit 只计算一个确定的 representative $c_0$。若

$$
c=g\cdot c_0,
$$

则重建

$$
F_c=Q_gF_{c_0}.
$$

之后仍按原始 configuration 顺序填满完整力数组，并调用现有 `CentralDifferenceStencil.contract()`。这样 contraction、分母、符号权重和 reconstruction 都无需改变，约化只是减少真实力计算次数。

## 6. 实现数据结构

建议增加内部对象：

```python
SignedDisplacementOrbit(
    representative_configuration: int,
    members: tuple[int, ...],
    force_actions: tuple[ForceAction, ...],
)
```

其中 `ForceAction` 保存：

- reference atom permutation；
- Cartesian rotation；
- 从 representative 到 member 的方向；
- 调试用的空间群 operation index。

`DisplacementPlan` 保存完整逻辑 configurations 和较小的 evaluation representatives。`sow()` 只输出 representatives；`reap()` 首先用 `force_actions` 恢复完整 force table，再执行原来的 contraction。

## 7. 构造算法

1. 按现有方法生成全部逻辑 displacement configurations。
2. 将每个位移写成稀疏的整数 coefficient 向量；单位为输入步长 $h$。
3. 先合并 coefficient 向量完全相同的 configurations，并累加其 stencil 权重。
4. 从 reference symmetry 生成 $Q_g$。
5. 仅保留把 Cartesian 单位轴映射为 signed Cartesian 单位轴的操作。
6. 对每个 configuration 计算变换后的整数 coefficient 向量。
7. 只有变换结果精确存在于完整 plan 中时才建立边。
8. 用 union-find 或显式群闭包建立 configuration orbits。
9. 每个 orbit 取原始 configuration id 最小者作为 representative。
10. 保存一条确定的 representative-to-member force action。

浮点 Cartesian rotation 不能直接作为哈希键。必须先验证其目标分量在 `symprec` 内等于 $0$ 或 $\pm1$，再转成整数 signed permutation；未通过的操作直接不参与 signed reduction。

## 8. 与数值噪声的关系

对严格保持晶体对称性的势函数，重建关系是精确的。DFT 的 SCF 阈值、非对称 FFT 网格或外部计算器可能产生小的对称性破缺。减少计算后相当于把力响应严格投影到所声明的晶体对称性上。

这与当前 orbit reconstruction 的物理假设一致，但必须提供诊断模式：在测试和验证中同时计算少量被省略构型，报告

$$
\max_c\frac{\lVert F_c-Q_gF_{c_0}\rVert}
{\max(\lVert F_c\rVert,\epsilon)}.
$$

生产 API 不应静默使用近似匹配；若结构或计算器不满足指定对称性，应禁用该约化或明确报错。

## 9. 验证要求

必须逐级验证：

1. 重复自由度导致的相同位移合并，与完整 stencil 逐元素一致；
2. 人工构造的严格对称二次、三次和四次多项式势能，完整和约化结果一致；
3. FC2、FC3、FC4 覆盖不同 stabilizer 大小；
4. 覆盖反演存在和不存在的晶体；
5. 覆盖 Cartesian 旋转不是 signed permutation 的六方晶体，确认这些操作不会被错误采用；
6. 覆盖 reference 原子随机重排和非对角超胞；
7. 比较完整与约化路径的 displacement keys、重建 IFC、ASR 前后张量和导出结果；
8. 使用真实 ASE calculator 抽查被省略构型的力协变残差；
9. 报告每阶从原始 $N_{\mathrm{key}}2^{n-1}$ 到实际力计算数的缩减比例。

只有在完整和约化路径于确定性对称势上达到浮点舍入精度一致后，才能接入生产路径。该功能不改变 interaction orbit、IFC 参数空间、有限差分公式或输出格式。

