# 谐波晶格动力学

二阶力常数描述原子位移引起的线性回复力。有了这些相互作用，就可以研究晶体中的集体振动：周期性允许把振动按波矢分解，而每个波矢下只需处理一个原胞大小的动力学矩阵。

这里从[周期力常数](periodic-force-constants.md)和[对称参数化](symmetry-parameterization.md)得到的 FC2 出发，说明它怎样进入声子频率。质量和傅里叶相位在这一步出现，二阶力常数本身仍是势能对笛卡尔位移的导数。

## 先保留每个原子对的相对位置

代表轨道的 FC2 张量由 $\Phi_o=B_o\theta_o$ 给出。通过空间群作用和指标置换，将它扩展到所有锚定原子对 $(i,j,\mathbf n)$，相应的笛卡尔张量记为 $\Phi_{i\alpha,j\beta}(\mathbf n)$。

这里 $\mathbf n$ 是第二个原子相对第一个原子的晶格平移。它不仅用于辨认相互作用，也决定振动传播到这个原子时的相位。仅有某个超胞的折叠原子对总和，通常无法恢复每项贡献的相对平移，因而不能据此重建任意 q 点的响应。这与[超胞折叠](supercells-identifiability.md)中的信息损失是同一个问题。

## 相位为什么包含原胞内部位置

不同原子除了位于不同晶格平移处，还各有原胞内的分数坐标 $\mathbf s_i$。两者之间的完整分数间距为

$$
\mathbf d_{ij}(\mathbf n)=\mathbf n+\mathbf s_j-\mathbf s_i.
$$

MLFCS 采用包含这个间距的傅里叶相位，称为位置规范（positional gauge）。以原胞倒格基中的分数行坐标 $q$ 表示波矢，则笛卡尔波矢为 $\mathbf k=2\pi qA^{-\mathsf T}$，并有 $\mathbf k\cdot(\mathbf dA)=2\pi q\cdot\mathbf d$。

将所有原子对的回复力按这个相位求和，再引入质量加权，就得到动力学矩阵：

$$
D_{i\alpha,j\beta}(q)
=\frac{1}{\sqrt{m_im_j}}
\sum_{\mathbf n}\Phi_{i\alpha,j\beta}(\mathbf n)
e^{2\pi i q\cdot\mathbf d_{ij}(\mathbf n)}.
$$

每一项的相位、质量和未折叠 FC2 对应同一组原子。FC2 的单位为 eV/Å$^2$，质量为 amu，因此矩阵数值的单位为 eV/(Å$^2$ amu)。分数 q 坐标可以相差一个倒格矢，但不能简单通过周期匹配抹去原胞内部位置的相位。

力常数满足原子对交换对称性时，$D(q)$ 是厄米矩阵。实现中最后取 $(D+D^\dagger)/2$，清理浮点舍入引入的非厄米部分；它不替代力常数应当满足的物理对称性。

## 质量与声学零模

质量不改变几何对称所允许的力常数空间，却会改变动力学矩阵和频率。例如，同位素质量的改变可以保持原有力常数参数，同时缩小质量加权振动问题的对称群。

在 $\Gamma$ 点，三个刚性平移在质量加权坐标中写成

$$
t^{(\gamma)}_{i\alpha}=\sqrt{m_i}\,\delta_{\alpha\gamma}.
$$

若 FC2 满足[声学求和规则](invariance-constraints.md)，这些方向就是三个零频模式。`Harmonic` 按给定 FC2 计算频率，不自动施加 ASR，也不强制把三个本征值设成零。

这些零模在频率计算中有明确物理意义，但在计算位移涨落时需要单独处理。后面的 SCPH 会从 $\Gamma$ 点协方差中排除刚性平移，以避免零频造成的发散。

## 本征值怎样换成频率

求解 $D(q)e_\nu=\lambda_\nu e_\nu$ 后，正本征值在换算到 SI 单位时对应角频率平方。实验和声子谱中常用的是普通频率 $f_\nu=\omega_\nu/(2\pi)$，而非角频率。

MLFCS 以 THz 返回带符号的普通频率：

$$
f_\nu(q)
=c_{\rm THz}\,\operatorname{sign}\lambda_\nu(q)
\sqrt{|\lambda_\nu(q)|},
$$

其中单位转换因子为

$$
c_{\rm THz}
=\frac{\sqrt{\mathrm{eV}/(\mathrm{Å}^2\mathrm{amu})}}
{2\pi\,10^{12}\,\mathrm{s}^{-1}}
$$

负返回值用来标记虚频模式：它表示参考构型沿该振动方向存在负曲率，不是负振动能量。计算不会将这些值裁剪为零。

每个 q 点的模式按本征值独立排序，不进行跨 q 点的分支追踪。因此，一条绘出的频率曲线未必始终对应同一个振动本征矢。

## 相同频率为何可以对应不同矩阵

频率随倒格矢周期重复，但位置规范下的动力学矩阵不一定逐元素重复。令整数倒格矢位移为 $G$，并令 $U_G$ 在原子 $i$ 的三个方向上乘 $e^{2\pi iG\cdot\mathbf s_i}$，则

$$
D(q+G)=U_G^\dagger D(q)U_G.
$$

这是一个酉相似变换，所以两边具有相同本征值，复矩阵元素却可能不同。对称性恢复频率只需要知道 q 点之间的等价关系；恢复矩阵还必须处理旋转、原子置换和这些位置相位。这个区别将在[倒空间网格与对称性](reciprocal-symmetry.md)中继续展开。

当前 `Harmonic` 从参数化原胞 FC2 计算动力学矩阵、频率、网格和能带路径。`DipoleEwald` 的折叠超胞张量不能直接作为任意 q 的解析长程模型接入，频率和能带路径计算也不自动添加非解析修正（NAC）。极性材料中的长程处理见[长程偶极作用](long-range-electrostatics.md)。

傅里叶几何与矩阵计算对应 `DynamicalTerms`、`prepare_dynamical_terms` 和 `accumulate_dynamical_matrices`；用户通过 `Harmonic` 使用这些结果。路径、网格及结果读取见[谐波 API](../harmonic-api.md)。

### 公式与工程对象

| 数学对象 | 代码 |
|---|---|
| 未折叠原子对张量与相对平移 | `LatticeForceConstants.sites/translations/tensors` |
| $\mathbf n+\mathbf s_j-\mathbf s_i$ | `DynamicalTerms.fractional_separations` |
| $1/\sqrt{m_im_j}$ | `DynamicalTerms.mass_weights` |
| 原胞原子对编号 | `DynamicalTerms.first_sites/second_sites` |
| 项的轨道和镜像来源 | `DynamicalTerms.orbit_indices/image_indices` |
| $D(q)$ | `accumulate_dynamical_matrices`、`Harmonic.dynamical_matrices` |
| 带符号普通频率 | `Harmonic.frequencies` |
| 排除质量加权刚性平移的正交补 | `translation_complement`，由 SCPH 协方差使用 |
