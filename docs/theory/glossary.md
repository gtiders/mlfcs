# 术语索引

这是一份中英文对照，供阅读正文或源码时查阅。概念的推导和例子在所链接的章节中，不需要先记住这些名称。

## 相互作用与对称性

| 中文 | 英文或代码名称 | 含义 |
|---|---|---|
| 力常数阶数 | force-constant order / tensor order | 能量对位移求导的次数；不指矩阵的秩 |
| 体数 | body order | 相互作用涉及的不同周期原子位置数 |
| 周期原子位置 | periodic lattice site / `LatticeSite` | 原胞原子编号与晶格平移共同指定的位置 |
| 原子簇 | cluster / `Cluster` | 按张量指标顺序排列的位置，允许重复，按共同平移等价锚定 |
| 对称轨道 | orbit / `Orbit` | 对称作用连接的等价原子簇 |
| 代表簇与轨道成员 | cluster representative / orbit image | 选定的代表，以及作用产生的其他等价簇 |
| 稳定子群 | stabilizer | 将锚定代表簇带回自身的作用集合 |
| 晶格镜像 | periodic image | 原子或相互作用的晶格平移副本，与轨道成员的含义不同 |

[周期力常数](periodic-force-constants.md)解释阶数、体数和原子簇，[对称性与独立参数](symmetry-parameterization.md)解释轨道和稳定子群。

## 张量分量与参数

| 中文 | 英文或代码名称 | 含义 |
|---|---|---|
| 晶格坐标与笛卡尔坐标 | lattice coordinates / Cartesian coordinates | 位置的分数晶格坐标与实际空间坐标；张量的晶格表示须结合所在推导解释 |
| 晶格基与分量基 | lattice basis / component basis | 整数晶格张量生成元，以及以选定笛卡尔分量为参数的基 |
| 物理参数坐标 | physical parameter coordinates | 选定的代表张量分量 $\theta$ |
| 观测行 | observation rows / `observation_rows` | 代表张量中选定的分量行，与原子力设计矩阵的行不同 |
| 完整晶格系数 | full lattice coordinates | 晶格基的系数 $c$，经 $W$ 转为物理参数 |
| ASR 自由坐标 | free acoustic coordinates | 满足声学方程所需的自由变量 $z$ |
| 拟合坐标 | fitting coordinates | 拟合求解的变量 $\eta$；关闭 ASR 时为 $\theta$，开启时为 $z$ |
| 声学方程与零空间 | acoustic equations / acoustic nullspace | 平移不变性的齐次方程，以及它们允许的完整实数子空间 |

这些坐标的关系见[对称参数化](symmetry-parameterization.md)和[平移与旋转不变性](invariance-constraints.md)。`LatticeForceConstants` 的名称中 lattice 指周期地址，输出张量分量仍为笛卡尔分量。

## 超胞与数据恢复

| 中文 | 英文或代码名称 | 含义 |
|---|---|---|
| 超胞映射 | supercell mapping / `ClusterMap` | 原胞周期位置与给定超胞原子的关系 |
| 平移商类 | quotient class | 模超胞平移的等价类 |
| 折叠 | folding | 映射到同一超胞原子组的镜像贡献相加 |
| 混叠 | aliasing | 不同镜像落到同一有序超胞原子组 |
| 结构可辨识性 | structural identifiability | 折叠是否保留全部独立参数方向 |
| 位移键与步长 | displacement key / step length | 需对哪些原子方向求导的序列，以及采样用的正标量长度 |
| 混合中央差分 | mixed central derivative | 用正负位移组合估计混合导数 |
| 零步长外推 | zero-step extrapolation | 利用偶次误差展开外推到零步长 |

相关解释见[超胞与可辨识性](supercells-identifiability.md)和[从原子力恢复力常数](reconstruction.md)。

## 声子与长程作用

| 中文 | 英文或代码名称 | 含义 |
|---|---|---|
| 动力学矩阵 | dynamical matrix | 质量加权的二阶力常数傅里叶矩阵 |
| 分数倒空间坐标 | fractional reciprocal coordinates | 原胞倒格基中的 q 坐标 |
| q 网格 | q-grid / `QGrid` | 与整数超胞对应的有限倒空间网格 |
| 不可约代表与星成员 | irreducible representative / star member | 一组对称等价 q 点中选定的代表及其他成员 |
| 位置规范 | positional gauge | 傅里叶相位包含原子内部位置差的约定 |
| 裸二阶项、有效二阶项 | bare FC2 / effective FC2 | SCPH 的固定输入与迭代后的二阶力常数 |
| 四阶环图修正 | quartic-loop correction | 四阶力常数与位移协方差收缩产生的二阶修正 |
| 位移协方差 | displacement covariance | $\langle uu\rangle$；实际位移与质量加权坐标的协方差须区分 |
| 长程偶极项 | long-range dipole contribution | 屏蔽偶极作用产生的谐波贡献 |
| 局域修正、同位置 ASR 项 | local correction / onsite ASR term | 有限非同位置修正，以及另外确定的同位置 ASR 补全 |

这些概念在[谐波动力学](harmonic-dynamics.md)、[倒空间对称性](reciprocal-symmetry.md)、[自洽声子](advanced/self-consistent-phonons.md)和[长程作用](long-range-electrostatics.md)中使用。

源码中的 array dimensionality 是数组轴数；rank 是线性映射的秩；order 是力常数的求导次数。三个量各有用途，不互相代替。
