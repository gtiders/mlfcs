# 平移与旋转不变性

[空间群约化](symmetry-parameterization.md)已经确定了每个代表簇允许具有怎样的张量。但相互作用之间还存在求和关系。把整个晶体一起平移或旋转，不应改变内部能量；仅求解每个簇的稳定子约束，还不能保证这些条件。

平移不变性可以直接缩小拟合所用的参数空间。旋转条件还涉及原子间的实际距离，并与参考结构是否处于力平衡、零应力状态有关。

## 刚性平移为什么产生求和规则

对全部原子施加同一个位移，能量不变。将这个关系对位移继续求导，便得到任意阶力常数的声学求和规则（acoustic sum rule，ASR）：

$$
\sum_{i_p,\mathbf n_p}
\Phi^{(p)}_{a_1\ldots a_{p-1},(i_p,\mathbf n_p,\alpha_p)}=0.
$$

求和遍历最后一个原子及其全部晶格镜像，前面的原子位置和所有方向保持固定。指标置换对称性保证，对其他位置求和会给出等价条件。

对于截断模型，求和覆盖模型保留的全部相互作用，未保留的项按零处理。这意味着满足 ASR 的截断模型中，不同相互作用的参数不能任意独立赋值。

把各轨道的晶格基系数拼成 $c$，求和关系就成为

$$
C_{\rm ac}c=0.
$$

MLFCS 在晶格张量表示中组装这个整数矩阵。它与笛卡尔坐标中的 ASR 等价：同阶各项共享可逆变换 $(A^{\mathsf T})^{\otimes p}$，可以将这个变换提出求和之外。

## 在满足 ASR 的空间中选择坐标

既然所有允许的系数都满足同一组齐次方程，就可以只使用它们的自由变量。这样求解时不必先得到一组违反 ASR 的参数，再额外修补。

沿用上一章的记号，$W$ 是各轨道物理坐标转换组成的分块对角矩阵，$\theta=Wc$。MLFCS 认证三角方程的主元位置，以非主元系数为自由变量 $z$，通过回代恢复完整系数：

$$
c=Tz,
\qquad
C_{\rm ac}T=0,
\qquad
\theta=L_{\rm ac}z,\qquad L_{\rm ac}=WT.
$$

$T$ 将自由变量映射到完整晶格系数，$L_{\rm ac}$ 再把它们转换成物理参数。它们覆盖完整的实数零空间，留下的自由维数为

$$
d_{\rm ac}=n_{\rm parameters}-\operatorname{rank}C_{\rm ac}.
$$

这里求的是实数参数空间，不要求构造一个全局饱和整数格的基。秩和零空间结构由整数恒等式认证，而坐标映射及其伴随运算在浮点中完成，因此实际参数满足 ASR 到数值精度。相关认证方法见[进阶线性代数](advanced/exact-linear-algebra.md)。

ASR 与质量无关，也不改变代表张量中所选物理分量的含义。启用这项约束后，`ClusterSpace.n_parameters` 仍是原始物理参数数，`n_free_parameters` 才是拟合使用的自由维数。`ForceConstants` 保存的始终是 $\theta$，不是 $z$。

当前默认不准备 ASR 自由坐标；启用时各阶分别构造它们。这个选择影响拟合和重建使用的坐标，但不会自动投影手动赋给 `ForceConstants` 的任意系数。

## 拟合和有限差分怎样使用这些坐标

拟合直接观察位移下的原子力。将力设计矩阵 $X$ 与 $L_{\rm ac}$ 复合，就可以在允许空间内求解：

$$
\min_z\|XL_{\rm ac}z-f\|_2^2.
$$

每个自由变量都只沿着满足 ASR 的方向改变模型，而最小化的仍是原子力残差。

有限差分采用另一种顺序。它先从中央差分恢复各轨道的物理参数 $\theta_{\rm FD}$，然后求

$$
\min_z\|L_{\rm ac}z-\theta_{\rm FD}\|_2^2.
$$

这一步把已经重建的物理参数投影到 ASR 子空间，使用的是物理参数坐标的欧氏距离。它没有直接重新拟合全部原始力样本，也没有因为施加 ASR 而减少差分采样数。两条路线的完整过程将在[从原子力恢复力常数](reconstruction.md)中展开。

## 旋转不变性还需要距离信息

对二阶力常数，旋转条件可以写成相互作用的一阶、二阶空间矩。首先定义原子对之间的实际笛卡尔距离向量：

$$
\mathbf d_{ij}(\mathbf n)=(\mathbf n+\mathbf s_j-\mathbf s_i)A.
$$

它保留晶格镜像的相对位置，不能用折叠后的原子编号代替。在原子处于力平衡的前提下，Born–Huang 第一矩条件为

$$
\sum_{j,\mathbf n}
\left[
\Phi_{i\alpha,j\beta}(\mathbf n)d_{ij,\gamma}(\mathbf n)
-\Phi_{i\alpha,j\gamma}(\mathbf n)d_{ij,\beta}(\mathbf n)
\right]=0.
$$

这个条件对每个原胞原子分别成立，联系不同方向的回复力及其力臂。

若体系还处于零应力状态，Huang 第二矩条件进一步要求

$$
\sum_{i,j,\mathbf n}
\left[
\Phi_{i\alpha,j\beta}(\mathbf n)d_{ij,\gamma}(\mathbf n)d_{ij,\delta}(\mathbf n)
-\Phi_{i\gamma,j\delta}(\mathbf n)d_{ij,\alpha}(\mathbf n)d_{ij,\beta}(\mathbf n)
\right]=0.
$$

第二矩对原胞中的原子求总和。它并不是受应力结构普遍满足的条件，因此 MLFCS 默认不施加。两组关系的定义和前提可参阅 [Lin、Poncé 与 Marzari 的旋转不变性研究](https://doi.org/10.1038/s41524-022-00920-6)，式 (6) 和 (16)。

与 ASR 不同，这些方程直接包含实际浮点几何。当前实现保留这种几何，不将它近似成有理数后交给整数核求解。

## 怎样修正旋转条件而不改变 ASR

设待修正模型的物理参数为 $\theta$。MLFCS 在物理参数的欧氏度量下寻找最小修正，同时要求修正本身位于 ASR 子空间。这样不会改变原来的 ASR 残差。

令 $N$ 的列为 ASR 子空间的正交法向，$P_{\rm ac}=I-NN^{\mathsf T}$ 是投影到允许方向的算子。若 $M$ 收集旋转矩条件，修正满足

$$
\delta\theta\in\operatorname{im}P_{\rm ac},
\qquad
MP_{\rm ac}\delta\theta=M\theta.
$$

截断 SVD 给出保留奇异方向上的最小范数修正，返回 $\theta'=\theta-\delta\theta$。几何误差无法可靠分辨的方向不作修正。自由晶格坐标通常不是正交坐标，因此直接最小化 $\|\delta z\|$ 会对应不同的度量。

修正后，原来满足 ASR 的模型继续满足；原来已有的 ASR 违反则仍然存在。旋转修正不能代替 ASR 的构造。

实现用典型的非同位置原子间距归一化两种矩条件。自动奇异值阈值参考实测的原子位置对称误差和笛卡尔旋转的非正交误差，并保留机器精度下限。`symprec` 是匹配容差，不等于这些实测误差。

代码中的 `AcousticCoordinates` 提供上述自由坐标映射，`ForceConstants.enforce_rotation` 执行旋转矩修正。后者需要已准备的 ASR 坐标和非同位置的二阶相互作用，只改变二阶系数，高阶系数保持原值。

## 从轨道镜像组装声学方程

要重新实现高阶 ASR，需要确定每一行固定了什么。对每个有序轨道镜像 $(a_0,\ldots,a_{p-1})$，取前缀 $(a_0,\ldots,a_{p-2})$ 作为分组键。位置包含原胞编号与相对晶格平移；同一前缀的不同最后位置参与同一求和。

设轨道 $o$ 的第 $h$ 个镜像作用为 $T_{oh}$，该轨道的晶格基为 $K_o$。令 $b_o$ 为轨道系数块起点，$j(\alpha)$ 为 C 顺序张量分量编号，则每个贡献为

$$
(C_{\rm ac})_{(\text{prefix},\alpha),\,b_o+k}
\mathrel{+}= (T_{oh}K_o)_{j(\alpha),k}.
$$

每个前缀有 $3^p$ 个分量行，包含被求和位置的方向指标；求和只改变最后原子的身份与晶格平移。所有有序镜像都参与，不额外乘阶乘。合并同一行列的整数贡献，零和删除；列块依次使用该阶的轨道顺序。

物理转换在每个轨道上为 $W_o=[(A^{\mathsf T})^{\otimes p}K_o][J_o,:]$。将其组成分块对角 $W$，并将组装的整数方程交给[认证三角因子](advanced/exact-linear-algebra.md)，得到同核整数行矩阵 $U$、按消元建立顺序排列的主元 $P$，以及按原列号升序排列的自由列 $F$。

## 提升与伴随的递推

三角结构指第 $k$ 行在先前主元处为零，而不是要求主元列号递增。令主元为 $p_k$，非零对角为 $d_k=U_{k,p_k}$，将 $c_F=z$ 后，按逆主元顺序计算

$$
c_{p_k}=-\frac{1}{d_k}\sum_{j\ne p_k}U_{kj}c_j.
$$

随后逐轨道计算 $\theta_o=W_oc_o$。这就是 $L_{\rm ac}z$，不需要生成 $T$ 或 $L_{\rm ac}$ 的全部列。

伴随可以由行限制得到。给定物理观测行 $r$，先令 $v=rW$。按正主元顺序，对每个 $k$ 取 $a=v_{p_k}/d_k$，并对 $j\ne p_k$ 更新 $v_j\leftarrow v_j-aU_{kj}$；处理后读取自由列 $v_F$。由回代关系直接可知

$$
rL_{\rm ac}=(rW)T,
\qquad
L_{\rm ac}^{\mathsf T}r^{\mathsf T}=(rL_{\rm ac})^{\mathsf T}.
$$

这解释了 `restrict_rows` 与 `adjoint` 的关系。伴随不是提升的逆，迭代最小二乘只需要这两个方向的线性作用。浮点回代和物理转换都有舍入误差，整数认证不消除这些误差。

若已知物理参数满足 ASR，可以先逐块解 $W_oc_o=\theta_o$，取 $c_F$，再以提升恢复原向量并检查数值一致性；这是 `extract`，不是将任意模型投影到 ASR。零自由维数时提升唯一返回零向量；零约束秩时全部晶格系数自由。

旋转修正使用的物理法向来自 $(UW^{-1})^{\mathsf T}$ 的列空间，通过薄 QR 正交化；其列数使用已认证的约束秩，而非浮点 SVD 重新推断。它与自由坐标提升属于同一允许空间的两种表示，不能混同为一个正交基。

### 公式与工程对象

| 数学对象 | 代码 |
|---|---|
| $C_{\rm ac}$ 与各轨道 $W_o$ | `lattice_acoustic_equations` 的矩阵与映射返回值 |
| 整数三角 $U$ | `AcousticCoordinates.indptr/indices/values`，稀疏行存储 |
| 主元与自由列 | `AcousticCoordinates.pivots/free` |
| $W_o$ 分块 | `AcousticCoordinates.physical_maps` |
| $L_{\rm ac}z$ | `AcousticCoordinates.lift` |
| $RL_{\rm ac}$、$L_{\rm ac}^{\mathsf T}r$ | `restrict_rows`、`adjoint` |
| 满足约束的物理向量提取自由坐标 | `extract` |
| 物理参数欧氏度量下的法向 | `normal_basis` |
