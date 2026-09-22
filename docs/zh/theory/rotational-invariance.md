# 旋转不变约束：原理、数学与实现

本文解释 MLFCS 中"施加旋转不变规则"这件事的物理依据、数学形式，以及 `src/mlfcs/constraints/` 里的实现方式与数值策略。所有结论都对应到具体代码位置；教程里的数字是 `tutorial/rotational-sum-rules/` 已提交的实测结果（MLFCS 4.0.0a6）。

## 0. 代码地图

| 位置 | 角色 |
|---|---|
| `src/mlfcs/constraints/translational.py` | 平移（声学求和规则 ASR）约束矩阵与参数空间投影：`build_translational_constraints`、`project_parameters`、`project_acoustic_sum_rule`、`maximum_acoustic_sum_rule_drift`、`maximum_constraint_residual` |
| `src/mlfcs/constraints/rotational.py` | FC2 后处理投影：`enforce_rotational_sum_rules`、`RotationalSumRuleResult`，三个约束矩阵 `_asr_matrix`、`_born_huang_matrix`、`_huang_matrix`，投影内核 `_metric_project`、`_null_asr_constraint_correction`、`_spectral_pinv`，写回 `_replace_fc2` |
| `src/mlfcs/fitting/constraints.py`、`src/mlfcs/fitting/fitter.py` | 拟合阶段在 orbit 参数空间施加 ASR（显式约束参数化 + 预处理） |
| `src/mlfcs/finite_difference/reconstruction.py` | 有限差分重建阶段投影 ASR 并报告漂移 |
| `tutorial/rotational-sum-rules/` | 两个独立拟合任务 `asr/` 与 `born-huang-huang/`，以及对比声子谱的 `plot.py` |
| `tests/test_rotational_sum_rules.py` | 严格投影、`strength=0`、FC3/FC4 不被改动的契约测试 |

## 1. 物理起点

### 1.1 势能展开与力常数

设晶体在简谐近似下的位移场为 $u^\alpha_{i\mathbf R}$（$i$ 为 primitive site 编号，$\mathbf R$ 为整数晶格平移，$\alpha$ 为 Cartesian 分量）。把势能写成

$$
U = U_0 + \frac12 \sum_{i,j,\mathbf R} u^\alpha_{i}\, \Phi^{\alpha\beta}_{ij}(\mathbf R)\, u^\beta_{j\mathbf R} + O(u^3),
$$

其中

$$
\Phi^{\alpha\beta}_{ij}(\mathbf R) \;=\; \frac{\partial^2 U}{\partial u^\alpha_{i\mathbf 0}\,\partial u^\beta_{j\mathbf R}}
$$

就是 MLFCS 里"带晶格标签的稀疏 FC2"：`SparseOrderForceConstants(order=2)` 的三元组 `sites=(i, j)`、`translations=(\mathbf R,)`、`tensors=(\Phi^{\alpha\beta}_{ij}(\mathbf R))`，见 `src/mlfcs/force_constants/representation.py`。每个稀疏行都显式携带自己的整数平移 $\mathbf R$，因此

$$
\mathbf r_{ij\mathbf R} \;=\; \mathbf p_j - \mathbf p_i + \mathbf R \cdot \mathbf A
$$

是不依赖任何"最小像折叠"的真实 bond 向量（$\mathbf A$ 为 primitive cell 矩阵，按行排列；代码中用 `primitive.positions[second] - primitive.positions[first] + translation @ cell` 计算，见 `_physical_fc2_values`）。

### 1.2 为什么力常数必须满足求和规则

$U$ 是**构型**的函数：把整块晶体刚体平移或刚体旋转，物理构型不变，$U$ 必须不变。把这个要求代入上式的二次型，就得到对 $\Phi$ 的线性约束 —— 这就是求和规则。它们不是"数值技巧"，而是势能必须满足的对称性；不满足意味着拟合/差分给出的力常数在刚体自由度上"产生了能量"，表现为：

- 声学支在 $\Gamma$ 点不归零、$q\to0$ 出现虚假频移；
- 长波极限下弹性常数与内应变张量出现不符合旋转不变性的伪贡献；
- 参考结构带应力时，应力平衡条件与力常数不自洽。

三条约束对应两种不变性：

| 约束 | 来源的不变性 | 物理含义 |
|---|---|---|
| ASR（平移/声学求和规则） | 刚体平移 | 均匀位移不产生净力；声学支 $\omega(\Gamma)=0$ |
| Born–Huang（一阶矩） | 刚体旋转 | 一阶矩在末两指标上对称；长波弹性响应满足旋转不变性 |
| Huang（二阶矩） | 刚体旋转（二阶矩） | 等价于参考态零应力条件（仓库说明见下） |

仓库既有的用户级说明在 `docs/zh/theory/constraints.md`：*"Huang 是零应力条件，只适用于无应力参考结构；它不替代长程静电或 NAC 处理"*，以及"Born-Huang 与 Huang 是不同语义的 FC2 物理后处理"。本文补充其数学与实现细节。

### 1.3 与拟合阶段 ASR 的分工

ASR 在**拟合/有限差分阶段**就已经施加。拟合侧是精确的：`ForceConstantFitter` 把 ASR 写成 orbit 参数空间里的线性方程组，直接做约束参数化（日志实例：graphene 54 个参数 → 52 个独立参数、2 个约束块、"18 ASR before compression"、"Maximum joint constraint residual: 1.070650e-15"；MoS2 为 80 个独立参数、"27 ASR before compression"、残差 5.66e-16）。有限差分侧走 `project_acoustic_sum_rule`，用 LSMR 把 pivot 参数投影到 $A p = 0$ 上，并报告投影前后的漂移。

Born–Huang 与 Huang 则**不在拟合里施加**：它们是 FC2 的独立后处理（`enforce_rotational_sum_rules`），因此可以分别开关、可以用 `strength` 做部分修正、可以保留原始结果做对比。函数文档明确：它"不把约束重新塞回原拟合 null space，也不改变 cutoff"，且只改 order 2，FC3 及以上逐项复制。

### 1.4 直觉图像：三条规则各管一件事

| 规则 | 一句话 | 它在求和什么 |
|---|---|---|
| ASR | 整块平移不产生力 | 零阶矩 $\sum_{j\mathbf R}\Phi^{\alpha\beta}_{ij}$ |
| Born–Huang | 整块旋转不产生力矩 | 一阶矩 $\sum_{j\mathbf R} r^\gamma_{ij\mathbf R}\,\Phi^{\alpha\beta}_{ij}$（力臂 × 劲度） |
| Huang | 整块旋转不产生应力 | 二阶矩 $\sum_{j\mathbf R} r^\gamma r^\delta\,\Phi^{\alpha\beta}_{ij}$（力臂² × 劲度） |

**一阶矩就是力矩。** 每根键对原子 $i$ 的贡献是"力臂 × 劲度"。如果原子 $i$ 的邻居一侧硬、一侧软（或只在一个方向有邻居），这些力矩就不会互相抵消：把整块晶体刚体转过一个小角 $\theta$，原子 $i$ 会收到一个凭空出现的力矩，势能随 $\theta$ 变成 $\propto\theta^2\ne 0$ —— 等于说"转一下晶体就改变能量"，物理上不可能。纯中心力（$\Phi\propto\hat r\otimes\hat r$）自动满足这条，所以 Born–Huang 实际约束的是**非中心（弯曲/扭转）劲度必须成对抵消**。

**二阶矩就是应力。** 刚体转动在位移梯度里是"反对称部分"，也就是**无应变**的运动。要让它不与应变耦合，就需要二阶矩平衡。Huang 条件经长度尺度还原后的量纲正是能量（应力 × 体积，见 §3.5 的单位表），所以它读作"参考构型的内应力为零"：参考结构确实无应力时这条应当成立，不成立就是拟合/差分误差；参考结构本身带应力（固定实验晶格常数、未做应力弛豫、人为施加应变）时，真实的二阶力常数就不满足它 —— 这时强行投影，等于把残余应力的信息当作误差抹掉。"Huang 只对无应力参考结构有意义"就是这个意思。

**三条互不包含**（并且：纯中心力会自动满足全部三条，所以真正被约束的是非中心劲度，见 §2.4）。用本节的三条公式逐项算一个一维玩具（键长 $a=1$、横向劲度 $b=0.7$、$\pm x$ 处各一个邻居、onsite 项由 ASR 定出），最大残差为：

| 玩具构型 | ASR | Born–Huang | Huang |
|---|---|---|---|
| 纯中心力块 $-\hat r\otimes\hat r$，$\pm x$ 对称邻居 | 0 | 0 | 0 |
| 横向弹簧（两侧等硬）$+\ \text{onsite}$ | 0 | 0 | 1.4 |
| 横向弹簧（一侧 $1.3$、一侧 $0.35$） | 0 | 0.95 | 1.65 |
| 只有 $+x$ 一个邻居 | 0 | 0.70 | 0.70 |

第二行是关键：**Born–Huang 满足而 Huang 不满足**——横向劲度对称配对时"转动不产生力矩"，但它仍然携带应力型响应。教程里的真实体系正是这种情形（graphene 的 `born_huang_before` $1.06\times10^{-14}$ 但 `huang_before` $1.0357$）。第四行说明不对称环境连力矩平衡都破掉了。

## 2. 数学

记号约定：$\Phi^{\alpha\beta}_{ij}(\mathbf R)$ 的第一个下标 $i$ 固定在原点胞，$\mathbf R$ 指第二个原子的相对平移；所有求和都对该 $i$ 的全部 $(j,\mathbf R)$ 行（含 onsite 行 $(j=i,\mathbf R=0)$）进行。$\epsilon^{\alpha\mu\nu}$ 为 Levi-Civita 符号。

### 2.1 平移不变 → ASR

取 $u^\alpha_{i\mathbf R} = a^\alpha$（任意常量），二次型成为 $\frac12 a^\alpha a^\beta \sum_i \sum_{j\mathbf R}\Phi^{\alpha\beta}_{ij}(\mathbf R)$。对任意 $\mathbf a$ 必须为零，于是

$$
\boxed{\;\sum_{j\mathbf R} \Phi^{\alpha\beta}_{ij}(\mathbf R) = 0 \qquad \forall\, i,\alpha,\beta\;}
$$

即每个 primitive atom 的力常数对"所有邻居 + 自身"求和为零。等价说法：均匀位移下每个原子受力为零；也等价于 $q=0$ 的动力学矩阵具有三个零特征值（$\omega_{\text{acoustic}}(\Gamma)=0$）。

### 2.2 旋转不变 → 矩条件

取刚体旋转位移场 $u^\alpha_{i\mathbf R} = \epsilon^{\alpha\mu\nu}\omega^\mu r^\nu_{i\mathbf R}$（$\boldsymbol\omega$ 为无穷小旋转矢量，$r$ 为绝对坐标）。代入二次型：

$$
U_2 = \frac12\,\omega^\mu\omega^\rho\,
\epsilon^{\alpha\mu\nu}\epsilon^{\beta\rho\sigma}
\sum_{i,j\mathbf R} r^\nu_{i\mathbf 0}\, \Phi^{\alpha\beta}_{ij}(\mathbf R)\, r^\sigma_{j\mathbf R}.
$$

要求 $U_2$ 对任意 $\boldsymbol\omega$ 为零。用 ASR 可以把 $r^\nu_{i\mathbf 0}$ 平移成 $-r^\nu_{ij\mathbf R}$、把 $r^\sigma_{j\mathbf R}$ 平移成 $r^\sigma_{ij\mathbf R}$（ASR 允许在求和中加减与求和指标无关的量），于是条件只依赖 bond 向量：

$$
\epsilon^{\alpha\mu\nu}\epsilon^{\beta\rho\sigma}
\sum_{j\mathbf R} r^\nu_{ij\mathbf R}\, r^\sigma_{ij\mathbf R}\,
\Phi^{\alpha\beta}_{ij}(\mathbf R) \;=\; 0
\qquad (\text{对 }\mu,\rho\text{ 对称化后}).
$$

也就是说：旋转不变性对**二阶矩** $\sum_{j\mathbf R} r^\gamma r^\delta \Phi^{\alpha\beta}$ 施加条件。文献把这一族条件按矩的阶数拆成两支，MLFCS 把两支分别实现为线性约束、可独立开关：

**Born–Huang（一阶矩）**：对每个 $i$ 与 $\alpha$，要求一阶矩在末两指标上对称，等价于反对称部分为零：

$$
\boxed{\;\sum_{j\mathbf R}\Bigl(r^{\gamma}_{ij\mathbf R}\,\Phi^{\alpha\beta}_{ij}(\mathbf R)
- r^{\beta}_{ij\mathbf R}\,\Phi^{\alpha\gamma}_{ij}(\mathbf R)\Bigr) = 0
\qquad \forall i,\alpha,\ \beta<\gamma\;}
$$

代码只写 $(\beta,\gamma)\in\{(0,1),(0,2),(1,2)\}$ 三个独立分量（`_born_huang_matrix`：同一行上给 $\Phi^{\alpha\beta}$ 系数 $+r^\gamma$、给 $\Phi^{\alpha\gamma}$ 系数 $-r^\beta$）。

**Huang（二阶矩）**：要求二阶矩在交换两个指标对时对称：

$$
\boxed{\;\sum_{j\mathbf R}\Bigl(r^{\gamma}_{ij\mathbf R} r^{\delta}_{ij\mathbf R}\,\Phi^{\alpha\beta}_{ij}(\mathbf R)
- r^{\alpha}_{ij\mathbf R} r^{\beta}_{ij\mathbf R}\,\Phi^{\gamma\delta}_{ij}(\mathbf R)\Bigr) = 0
\qquad \forall i,\alpha,\beta,\gamma,\delta\;}
$$

代码对全部 $3^4=81$ 个 $(\alpha,\beta,\gamma,\delta)$ 组合建行（`_huang_matrix`：$\Phi^{\alpha\beta}$ 系数 $+r^\gamma r^\delta$，$\Phi^{\gamma\delta}$ 系数 $-r^\alpha r^\beta$）。这些行是冗余的（含 $(\alpha\beta\gamma\delta)\leftrightarrow(\gamma\delta\alpha\beta)$ 的相反行与恒等式行），冗余由谱伪逆处理，见 §3.3。

物理上，这两支条件保证由 $\Phi$ 导出的长波弹性/内应变响应满足旋转不变性所要求的对称性（标准文献结果，本文不展开指标等价性证明）；Huang 这一支与参考构型零应力等价，因此只对无应力参考结构有意义。ASR、Born–Huang、Huang 是三条**独立**约束：ASR 管平移，另两条管旋转的不同矩阶；只做 ASR 并不蕴含后两者（graphene 教程实测：ASR 与 Born–Huang 残差已在 $10^{-14}$ 量级，而 Huang 残差是 $1.04$，见 §5）。

### 2.3 约束规模

设 primitive 原子数为 $N_p$、参与约束的 bond 键数为 $B$（`_physical_fc2_values` 归并后的键数）：

| 约束 | 行数 | 列数 | 每键非零 |
|---|---|---|---|
| ASR | $9N_p$ | $9B$ | 该键的 9 个分量各进一行 |
| Born–Huang | $9N_p$ | $9B$ | 每行 2 个 |
| Huang | $81N_p$ | $9B$ | 每行 2 个 |

### 2.4 为什么是"二阶"：矩的阶数 = 位移场里力臂的幂次

三条规则是同一件事（势能在刚体运动与均匀变形下不变）在三类位移场上的读数，而"矩的阶数"由位移场里位置 $\mathbf r$ 的幂次直接决定：

| 运动 | 位移场 | 位移里的力臂幂次 | 代进二次型后出现的矩 |
|---|---|---|---|
| 刚体平移 | $u^\alpha = a^\alpha$ | 0 | 零阶矩 $\sum_{j\mathbf R}\Phi^{\alpha\beta}_{ij}$ → ASR |
| 刚体转动 | $u^\alpha = (\boldsymbol\omega\times\mathbf r_i)^\alpha$ | 1 | 二阶矩 $\sum_{j\mathbf R} r^\gamma r^\delta\,\Phi^{\alpha\beta}_{ij}$ → Huang |
| 均匀应变 | $u^\alpha = \varepsilon^{\alpha\beta} r^\beta_i$ | 1 | 同一个二阶矩 → 弹性常数 |

要点：二次型 $u^\alpha\Phi^{\alpha\beta}u^\beta$ 里有两个 $u$。**当位移场本身是位置的一次函数时，位置在二次型里出现两次**，于是条件落在二阶矩上。转动场与应变场都是位置的一次函数，唯一区别是位移梯度的对称性：应变取对称部分（有能量代价 → 弹性/应力），转动取反对称部分（必须零能量 → Huang）。这也回答了"为什么二阶矩同时是旋转条件和应力载体"——它们本来就是同一个矩上的两种对称性读法。

**"二阶矩 = 应力"的代数链条**：把位移写成 $\varepsilon\mathbf r$，键上的力就是 $\Phi\varepsilon\mathbf r$；再按 virial 定义对键求和

$$
\Omega\,\sigma^{\alpha\beta} \;=\; \sum_{\text{bonds}} r^\alpha\, f^\beta ,
$$

**又一次**乘上力臂，总共两次。于是应力—应变的系数（弹性常数）必然是二阶矩：

$$
C^{\alpha\beta\gamma\delta} \;\propto\; \sum_{j\mathbf R} r^\beta_{ij\mathbf R}\, r^\delta_{ij\mathbf R}\,\Phi^{\alpha\gamma}_{ij}(\mathbf R),
$$

与 `_huang_matrix` 使用的 $\sum r^\gamma r^\delta \Phi^{\alpha\beta}$ 是同一族量。应力的量纲是能量/体积，所以 Huang 残差（能量）除以参考体积就是应力量级；本仓不做这步换算，残差只作违约程度指标。

**一个必须知道的推论：纯中心力自动满足全部三条。** 当 $\Phi_{ij}\propto \hat r\otimes\hat r$ 时，二阶矩是四个 $\hat r$ 的乘积，交换指标对显然对称；数值验证见 §1.4 玩具表第一行（0/0/0），以及"各向同性横向劲度"那一行（环境仍对称，Born–Huang 仍为 0，Huang 变成 2.4）。所以 Huang 条件实际约束的是**非中心（有方向性）劲度**；它在真实拟合里会被违反，是因为拟合出的 IFC 不必是"某个旋转不变势的 Hessian" —— 多出来的那部分方向性劲度就是残差。这也解释了 §1.4 里 graphene 的图景：Born–Huang 已满足（$10^{-14}$）而 Huang 违约 1.04 eV，说明违约来自非中心劲度的二阶矩失配，而不是一阶力矩失配。

## 3. 实现

### 3.1 输入契约与前置校验

`enforce_rotational_sum_rules(force_constants, *, born_huang=False, huang=False, strength=1.0, tolerance=1e-8)`：

- 至少选择一个条件，否则 `ValueError("select born_huang=True and/or huang=True")`；
- `strength` 必须在 $[0,1]$，`tolerance` 必须为正；
- 必须有已验证的 `StructureRelation`（否则 `ValueError`）—— 因为约束要用 primitive 晶格向量；
- 必须有 lattice-labelled 稀疏 FC2（`sparse[2]`），否则 `ValueError`；
- 若 FC2 只有 onsite 项（非零键长为 0 的键为空），报 `ValueError("Born--Huang/Huang constraints require a non-onsite FC2 pair")`。

### 3.2 数据整理

`_physical_fc2_values` 做三件事：

1. 按 key $(i, j, R_x, R_y, R_z)$ 把稀疏行归并；同一 key 的多行取**算术平均**作为该键的物理张量；
2. 计算 bond 向量 $\mathbf r_{ij\mathbf R}$、二阶外积 $\mathbf r\otimes\mathbf r$、键长；
3. 取所有非零键长的**中位数**作为长度尺度 $L$（`length_scale`）。graphene 教程的 $L=5.9513\ \text{Å}$，MoS2 为 $5.4539\ \text{Å}$。

注意当前实现里 `multiplicities` 恒为 $1.0$，即每条键等权（权重接口 `weights`/`inverse_weights` 已按 9 分量/键铺开，只是取值为 1）。因此修正量是 9 维/键空间里的**欧氏最小范数**解，不做简并度或不确定度加权。

### 3.3 度量投影（minimum-norm correction）

设 $v\in\mathbb R^{9B}$ 是展平的 FC2 值、$A$ 为 ASR 矩阵、$C$ 为选中的 Born–Huang/Huang 矩阵、$W^{-1}$ 为权重对角阵（当前为单位阵）。把约束当成"把 $v$ 拉回 $\ker A\cap\ker C$"，并选择**在 $W$ 度量下改动最小**的那一点：

$$
v' = \operatorname*{arg\,min}_{Av' = 0}\ \lVert v'-v\rVert_W .
$$

单组约束 $A$ 的解由 $\ker A$ 上的加权正交投影给出：

$$
v' = v - W^{-1}A^{\mathsf T}\bigl(AW^{-1}A^{\mathsf T}\bigr)^{+} A\,v,
$$

这正是 `_metric_project`：先算 Gram 矩阵 $AW^{-1}A^{\mathsf T}$，再乘伪逆、再用 $W^{-1}$ 映射回参数空间。伪逆由 `_spectral_pinv` 实现：对称化后用 `eigh` 做谱分解，保留 $\lambda_i > \tau\max(1,\lambda_{\max})$ 的方向并取倒数，返回保留方向数 `retained_rank`。求按 $L$ 归一化后再做谱截断，所以 `tolerance` 是**无量纲**的（`docs/zh/reference/units-and-parameters.md` 也是这么定义的）。

### 3.4 两级投影：先 ASR，再在 ASR 的零空间里做旋转修正

关键点：Born–Huang/Huang 的修正**不能破坏已经满足的 ASR**。代码因此分两步：

1. **ASR 严格投影**：`asr_projected = _metric_project(asr, initial, ...)`。
2. **只在 $\ker A$ 里修正**：`_null_asr_constraint_correction` 用 Schur 补把 ASR 从约束系统中消去：

$$
G_{\text{null}} = C W^{-1} C^{\mathsf T} - \bigl(CW^{-1}A^{\mathsf T}\bigr)\bigl(AW^{-1}A^{\mathsf T}\bigr)^{+}\bigl(CW^{-1}A^{\mathsf T}\bigr)^{\mathsf T},
$$

对残差 $\operatorname{res}=C\,v$ 求乘子 $\Lambda = G_{\text{null}}^{+}\,\operatorname{res}$，得到落在 $\ker A$ 内的最小范数修正

$$
\delta = W^{-1}C^{\mathsf T}\Lambda - W^{-1}A^{\mathsf T}\Bigl[\bigl(AW^{-1}A^{\mathsf T}\bigr)^{+}\bigl(CW^{-1}A^{\mathsf T}\bigr)^{\mathsf T}\Lambda\Bigr].
$$

这是"先投影 ASR、再只做不移动 ASR 的旋转修正"的等价一次求解，代价是一次 $9B\times 9B$ 级别的稠密 Gram 运算。

3. **`strength` 缩放**：`projected = asr_projected - strength * correction`。`strength=1` 是严格投影；$[0,1)$ 只保留该比例的旋转修正（ASR 仍然精确）；`strength=0` 得到"只做 ASR"的结果。
4. **浮点清尾**：再跑一次 ASR 投影，只去掉舍入残差（注释原文：*"Algebraically the second correction is in ASR's null space. Repeat the exact ASR projection to remove only floating-point round-off."*）。

### 3.5 残差诊断与写回

`RotationalSumRuleResult` 同时给出修正前后的最大残差，且都还原成带物理单位的量：

| 字段 | 定义 | 单位 |
|---|---|---|
| `acoustic_*` | $\max\lvert Av\rvert$ | eV/Å² |
| `born_huang_*` | $\max\lvert A_{\text{BH}}(v/L)\rvert \cdot L$ | eV/Å |
| `huang_*` | $\max\lvert A_{\text{H}}(v/L^2)\rvert \cdot L^2$ | eV |
| `relative_fc2_correction` | $\lVert\Delta v\rVert_2/\lVert v\rVert_2$ | 无量纲 |
| `maximum_fc2_correction` | $\max\lvert\Delta v\rvert$ | eV/Å² |
| `retained_rank` | 谱截断后保留的约束方向数 | — |

未启用的条件对应字段为 `None`。写回由 `_replace_fc2` 完成：把修正值按 key 广播回**所有**同 key 的原始稀疏行（保持行序、`sites`/`translations` 不变），保留 `supercell`、`metadata`（追加 `harmonic_constraints` 子字典）、`relation` 与其它阶，返回新的 `ForceConstants`。原始对象不被修改。

## 4. 使用方式

```python
from mlfcs import enforce_rotational_sum_rules, write_force_constants

corrected = enforce_rotational_sum_rules(
    result.force_constants,          # 需要 relation + lattice-labelled sparse FC2
    born_huang=True,
    huang=True,
    strength=1.0,                    # 0 表示只保留严格 ASR
    tolerance=1e-8,                  # 无量纲谱截断
)
print(corrected.acoustic_before, corrected.huang_before, corrected.huang_after)
write_force_constants(corrected.force_constants, "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
```

## 5. 教程实测（仓库内已提交的数字）

`tutorial/rotational-sum-rules/{graphene,MoS2-monolayer}` 各有两个独立任务：`asr/fit.py`（只做 ASR 拟合）与 `born-huang-huang/fit.py`（拟合后再施加两个旋转条件）。后者的 `metrics.json` 记录：

| 量 | graphene | MoS2-monolayer |
|---|---|---|
| `length_scale` (Å) | 5.951295 | 5.453878 |
| `acoustic_before` → `after` (eV/Å²) | $3.20\times10^{-14} \to 7.11\times10^{-15}$ | $1.95\times10^{-14} \to 3.00\times10^{-15}$ |
| `born_huang_before` → `after` (eV/Å) | $1.06\times10^{-14} \to 2.64\times10^{-15}$ | $6.37\times10^{-1} \to 9.99\times10^{-15}$ |
| `huang_before` → `after` (eV) | $1.0357 \to 2.91\times10^{-13}$ | $7.893 \to 2.60\times10^{-13}$ |
| `relative_fc2_correction` | $1.22\times10^{-4}$ | $4.43\times10^{-3}$ |
| `maximum_fc2_correction` (eV/Å²) | $2.43\times10^{-3}$ | $3.02\times10^{-2}$ |
| `retained_rank` | 60 | 126 |

两种结构给出的图景一致且很有代表性：拟合阶段的 ASR 已经是机器精度，graphene 的 Born–Huang 也几乎满足（$10^{-14}$），**唯一显著被违反的是 Huang 条件**（graphene $1.04\ \text{eV}$、MoS2 $7.89\ \text{eV}$）；投影后残差降到 $10^{-13}$，而 FC2 只被改动 $1.2\times10^{-4}$ / $4.4\times10^{-3}$ 的相对量。MoS2 的 Born–Huang 也明显被违反（$0.637\ \text{eV/Å}$），说明多层/非中心力成分更强的体系里两支条件都需要。

`plot.py` 把 `asr/` 与 `born-huang-huang/` 的 `FORCE_CONSTANTS_2ND` 分别喂给 phonopy 画同一路径上的声子谱（图见同目录 `phonon-bands.png`），用于直观比较两支约束对长波声子的影响。

## 6. 限制与注意事项

- **只作用于 FC2**，且是后处理；不会回灌拟合的 null space，也不改 cutoff（`docs/zh/reference/constraints-api.md`）。
- **Huang 只对无应力参考结构有意义**；它不替代长程静电/NAC 处理（长程库仑在短程 FC 投影里被截断，需另行处理，见 `docs/zh/theory/long-range-electrostatics.md`）。
- **必须用未折叠的 bond 向量**：每个稀疏行携带自己的整数平移，因此代码不需要 Wigner–Seitz 最小像重建（docstring 明确：*"No Wigner--Seitz image reconstruction is needed because every sparse FC2 entry carries its physical integer translation explicitly."*）。对偏斜胞，折叠后的 $\mathbf r$ 会让矩条件写错。
- **简并/重复行**：同 key 多行先取平均再约束，修正值再广播回所有行；因此同 key 的行永远保持相等。
- **谱截断是唯一的"数值自由度"**：`tolerance` 太小会保留数值噪声方向（残差震荡），太大则丢掉真实约束方向（`retained_rank` 下降、条件无法满足）。约束矩阵的行本身冗余（尤其 Huang 的 81 组），秩由 `retained_rank` 报告。
- **当前等权**：`multiplicities` 恒为 1，修正量是欧氏最小范数；若将来要按简并度或拟合不确定度加权，入口已在 `weights`/`inverse_weights` 上预留。

## 7. 如何自行验证

```bash
# 契约测试：严格投影、strength=0 只做 ASR、FC3/FC4 不变
uv run pytest tests/test_rotational_sum_rules.py

# 端到端：两个任务的日志（fit.log，含 ASR 约束残差）与指标（metrics.json）
cd tutorial/rotational-sum-rules/graphene && uv run python asr/fit.py && uv run python born-huang-huang/fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

教程的每个拟合任务都会把完整 stdout/stderr/traceback 覆盖写入自己的 `fit.log`（仓库规则要求）。
