# 长程偶极作用与局域修正

前面的原子簇模型通过有限截断描述相互作用。极性晶体却有一类不能这样直接截断的谐波贡献：原子位移改变极化，产生由 Born 有效电荷决定的长程偶极作用。

MLFCS 将这部分作用单独求和。它的力可以从训练或差分数据中扣除，让有限原子簇模型恢复短程剩余；得到短程力常数后，再加回长程 FC2。这个分离还需要兼顾[平移不变性](invariance-constraints.md)和张量对称性，因此周期偶极求和之外会有一份有限局域修正。

## 位移怎样产生偶极响应

Born 有效电荷张量联系原子位移与极化变化：

$$
\delta P_\alpha
=\frac{e}{\Omega}\sum_{i\beta}Z^*_{i,\alpha\beta}u_{i\beta}.
$$

第一个笛卡尔指标表示极化方向，第二个表示位移方向。输入 $Z_i^*$ 按原胞原子顺序排列，数值以元电荷为单位。电子介电张量 $\epsilon_\infty$ 描述屏蔽，采用同一笛卡尔坐标系，必须对称、正定且与空间群相容。

Born 张量也必须满足空间群变换关系与电中性条件：

$$
\sum_iZ_i^*=0.
$$

MLFCS 检查这些物理条件，不自动校正输入。参考几何、Born 电荷和介电响应在模型中保持固定，所以得到的是谐波偶极力常数，而不是随每帧结构重新确定电荷的非线性点电荷势。

## 为什么采用 Ewald 求和

各向异性介电屏蔽下，格林函数的空间部分为

$$
G_\epsilon(r)
=\frac{1}{\sqrt{\det\epsilon_\infty}
\sqrt{r^{\mathsf T}\epsilon_\infty^{-1}r}}.
$$

对于不重合的原子对，将其负 Hessian 与 Born 张量收缩，就得到偶极 FC2：

$$
\Phi^{\rm dip}_{ij}
=-\frac{e^2}{4\pi\epsilon_0}
Z_i^{*\mathsf T}\nabla\nabla G_\epsilon(r_{ij})Z_j^*.
$$

要在三维周期体系中累加这些作用，Ewald 方法将求和分为实空间、倒空间和自作用三部分：

$$
\Phi^{\rm LR}_{\rm raw}
=\Phi^{\rm real}+\Phi^{\rm reciprocal}+\Phi^{\rm self}.
$$

实空间距离用 $\epsilon_\infty^{-1/2}$ 度量，倒空间距离用 $\epsilon_\infty^{1/2}$ 度量。分裂参数决定两部分的数值分工；收敛后，改变这个参数不应改变总相互作用。

当前求和省略倒空间零向量，对应零宏观电场边界。实空间与倒空间求和半径同时增大，直到连续两次变化满足绝对与相对容差。这是数值收敛检查，并非对剩余尾项的严格误差证明。

## 满足 ASR 还不够

通常可以通过把每一行的总和取负，补上同位置项来满足声学求和规则。然而，如果原始偶极行和不是对称的 $3\times3$ 矩阵，这样补出的同位置项就不符合 Hessian 对称性。

MLFCS 先在已有的非同位置 FC2 轨道空间中寻找局域修正 $\Delta\Phi^{\rm local}$，消去原始行和的反对称部分，再单独补上零平移同位置项：

$$
\Phi^{\rm onsite}_{ii}
=-\sum_{j,\mathbf n}
\left[\Phi^{\rm LR}_{\rm raw,ij}(\mathbf n)
+\Delta\Phi^{\rm local}_{ij}(\mathbf n)\right].
$$

这份补全是原始偶极作用之外额外添加的同位置贡献。非零平移的相互作用即使折叠到同一个超胞原子，仍属于原始周期求和，不能将它们当作同位置自作用删掉。

满足上述条件的修正可能不唯一。当前选择使展开后的非同位置笛卡尔张量范数最小：

$$
\min_{\Delta\theta}
\sum_{o,\,\text{image}}
\|B_{o,\text{image}}\Delta\theta_o\|_F^2
$$

相应度量为 $\sum_{\text{image}}B_{o,\text{image}}^{\mathsf T}B_{o,\text{image}}$，所以这里最小化的是张量变化，而非参数系数范数。`correction_norm` 测量这份局域修正，不包含随后确定的同位置补全。

修正后的长程作用为

$$
\Phi^{\rm LR}_{\rm corrected}
=\Phi^{\rm LR}_{\rm raw}
+\Delta\Phi^{\rm local}+\Phi^{\rm onsite}.
$$

局域修正可用的空间随 FC2 截断范围改变，但无限长程偶极尾部没有被这个截断限制。如果现有空间不足以满足条件，修正会失败，不通过放宽约束接受结果。当前恢复同位置对称性、ASR 以及相容的原子对交换和空间群对称性，不施加 Born–Huang 旋转约束。

长程分离的构造参考 [Zhou 等的方法](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.184309)，Phys. Rev. B 100, 184309 (2019), Sec. II；以上展开张量最小范数则明确了本项目采用的修正选择。

## 在超胞中扣除与加回长程作用

选定目标超胞后，重新进行周期 Ewald 求和，再加入同一原胞局域修正和同位置补全，得到

$$
\Phi^{\rm LR,S}\in\mathbb R^{N\times N_s\times3\times3}.
$$

这是第一原子轴采用原胞代表的紧凑 FC2。利用超胞平移对称性，可以生成完整 Hessian，并对映射原子顺序下的位移帧计算线性力：

$$
F^{\rm LR,S}(u)=-\Phi^{\rm LR,S}_{\rm full}u.
$$

将这份力显式从训练或有限差分数据中扣除，恢复的是短程剩余。总的超胞力常数再由两部分相加：

$$
F^{\rm SR}_{\rm target}=F_{\rm target}-F^{\rm LR,S},
\qquad
\Phi^{\rm total,S}=\Phi^{\rm SR,S}+\Phi^{\rm LR,S}.
$$

局域修正也参与了短程与长程的分配约定。改变修正范围后，只有配套进行扣除与加回，并且短程模型和数据足够充分，才能期待总模型一致。不同修正范围下的短程参数不能直接视为同一种分解。

## 超胞张量与非解析修正的区别

`DipoleEwald` 绑定一个超胞映射，预先计算这个超胞的折叠张量。后续计算力或导出周期张量可以复用它，不重复求和；改用另一超胞则需重新构造模型。

折叠 FC2 不保留各个长程晶格镜像，也不能单独确定任意 q 点的解析长程响应。极性晶体在 $q\to0$ 时的非解析修正（NAC）涉及方向相关的宏观电场，不能由给折叠数组加一份张量自动推出。

当前 `Harmonic` 没有解析长程或 NAC 入口。[NaCl 教程](../notebooks/nacl-long-range.ipynb)将相加后的紧凑 FC2 写成 phonopy 输入，再由 phonopy 处理 NAC。这个工作流与 MLFCS 原生计算任意 q 点电静力学仍有区别。

这里的长程模型、力扣除和张量组合分别对应 `DipoleEwald`、`ForceDataset.subtract_forces` 与 `CompactForceConstants`。具体输入和导出方式见[长程力与 Ewald](../ewald-api.md)。

## 三部分求和的具体表达

为固定归一化与符号，令 $E=\epsilon_\infty$、$v=E^{-1}r$、$\rho^2=r^{\mathsf T}E^{-1}r$、$d_E=\sqrt{\det E}$，分裂参数为 $\alpha>0$。将负 Hessian 的实空间部分写成

$$
\mathcal D^{\rm real}(r)=\frac{1}{d_E}
\left[E^{-1}a(\rho)-vv^{\mathsf T}b(\rho)\right],
$$

$$
a(\rho)=\frac{\operatorname{erfc}(\alpha\rho)}{\rho^3}
+\frac{2\alpha e^{-\alpha^2\rho^2}}{\sqrt\pi\rho^2},\qquad
b(\rho)=\frac{3\operatorname{erfc}(\alpha\rho)}{\rho^5}
+\frac{6\alpha e^{-\alpha^2\rho^2}}{\sqrt\pi\rho^4}
+\frac{4\alpha^3e^{-\alpha^2\rho^2}}{\sqrt\pi\rho^2}.
$$

对周期晶胞体积 $\Omega$ 和笛卡尔倒格矢 $k$，余下部分为

$$
\mathcal D^{\rm reciprocal}(r)=\frac{4\pi}{\Omega}
\sum_{k\ne0}\frac{kk^{\mathsf T}}{k^{\mathsf T}Ek}
\exp\!\left(-\frac{k^{\mathsf T}Ek}{4\alpha^2}\right)\cos(k\cdot r),
\qquad
\mathcal D^{\rm self}=-\frac{4\alpha^3}{3\sqrt\pi\,d_E}E^{-1}.
$$

实空间对 $r=r_j-r_i+nA$ 求和并排除重合位置；倒空间采用 $k=2\pi mA^{-\mathsf T}$。自作用只添加到当前周期晶胞中的同一原子条目。得到总 $\mathcal D$ 后，计算 $(e^2/4\pi\epsilon_0)Z_i^{*\mathsf T}\mathcal DZ_j^*$；长度以 Å 计、Born 数值以元电荷计，库仑因子需转换为 eV·Å，输出为 eV/Å$^2$。

当前 $\alpha=\sqrt\pi/(\Omega_{\rm primitive}/d_E)^{1/3}$。取 $t_0=\sqrt{-\log(\mathrm{rtol})}+1$，依次用 $t=t_0,\ldots,t_0+11$，实空间截断为 $\rho\le t/\alpha$，倒空间截断为 $\sqrt{k^{\mathsf T}Ek}\le2\alpha t$。构造实空间平移候选时要额外覆盖原子间位置差，再按实际 $\rho$ 筛选。代码排除 $\rho^2\le10^{-24}$ 的重合项。

每轮以全部 FC2 分量最大绝对变化检查 $\mathrm{atol}+\mathrm{rtol}\max|\Phi|$，必须连续两轮满足；十二轮仍不满足则报错。这是当前数值策略，不是 Ewald 恒等式本身。

### 局域修正的可执行最小范数问题

对每个非同位置轨道收集镜像物理基 $B_{oh}$，令

$$
G_o=\sum_hB_{oh}^{\mathsf T}B_{oh},\qquad
G_o=C_oC_o^{\mathsf T},\qquad J_{\rm white,o}=C_o^{-\mathsf T}.
$$

白化矩阵 $J_{\rm white,o}$ 将欧氏坐标映射到展开张量度量下的参数。将镜像基按第一原胞原子累积成行和，提取 $xy-yx,xz-zx,yz-zy$ 三个反对称分量，组成 $R_o$。令原始行和为 $Q_i$，目标为 $b=-\operatorname{antisym}(Q)$，则

$$
\min_y\|y\|_2\quad\text{满足}\quad
[\,R_oJ_{\rm white,o}\,]_o y=b,\qquad
\Delta\theta_o=J_{\rm white,o}y_o.
$$

因为 $J_{\rm white,o}^{\mathsf T}G_oJ_{\rm white,o}=I$，这个欧氏目标就是展开张量范数。当前用 SVD 最小二乘取最小范数解，再要求最大绝对约束残差不超过 $\mathrm{atol}+\mathrm{rtol}\max|Q|$；无可用列且目标非零时同样失败。之后补同位置项，并核对它的对称性。

> **算法：指定超胞的修正偶极模型**
>
> 1. 验证 Born、电中性、介电正定与空间群相容性。
> 2. 在原胞周期边界下求原始偶极 FC2 行和，求白化局域修正和同位置补全。
> 3. 在目标超胞重新求周期 Ewald FC2；映射并累加相同局域镜像修正，再把同位置补全加入每个原胞代表对应的超胞锚点。
> 4. 核对完整周期 Hessian 的 ASR 和原子对交换对称性，保存紧凑张量。计算力时按平移对称性扩展并收缩，不重新求和。

### 公式与工程对象

| 数学对象 | 代码 |
|---|---|
| 三部分屏蔽 Hessian | `_sum_dipoles` |
| 半径收敛序列 | `_raw_fc2`、`_converged_fc2` |
| $G_o,J_{\rm white,o},R_o$ | `_local_correction` 中的 `tensor_metric/whitener/antisymmetric` |
| 展开修正范数 | `DipoleEwald.correction_norm` |
| 指定超胞的修正 FC2 | `DipoleEwald.force_constants()` 返回的 `CompactForceConstants` |
| $-\Phi_{\rm full}u$ | `DipoleEwald.forces` |
