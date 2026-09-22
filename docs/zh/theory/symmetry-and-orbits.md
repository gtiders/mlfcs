---
title: 对称性与轨道
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# 对称性与轨道

## 动机

空间群对 primitive site 和整数平移的作用是仿射的：$g: (\mathbf s, \mathbf t) \mapsto
(R\mathbf s + \boldsymbol\tau,\ \mathbf t R^{\mathsf T} + \boldsymbol\delta)$。它是否精确可判定，取决于
在哪个参考系里写下旋转矩阵。

* 在 **lattice（scaled）参考系**里，spglib 给出的旋转矩阵 $R$ 对任何晶胞都是**整数**矩阵；
* 在 Cartesian 参考系里，$R_{\mathrm{cart}} = A^{-\mathsf T} R A^{\mathsf T}$ 只在晶胞与坐标轴对齐
  （正交、四方、立方）时才是整数，fcc primitive $60^\circ$ 胞、六方与菱方胞都含 $\sqrt3/2$ 等无理数。

因此"哪些张量分量被对称性允许"必须用整数判定，而"有限差分观测到了什么、拟合交付了什么"是
Cartesian 物理量。两者不能混成同一个 `basis`。

## 数学对象

记用户 primitive 晶胞为 $A_s$（行向量为晶格矢量），规约后的代数晶胞为 $A$，两者通过唯一的
整数矩阵 $U$ 联系：

$$
A = U A_s, \qquad \lvert\det U\rvert = 1, \qquad U, U^{-1} \in \mathbb Z^{3\times 3}.
$$

同一物理点、同一个整数平移与同一次旋转在两个参考系中的坐标分别为

$$
\mathbf f_A = \mathbf f_s U^{-1}, \qquad \mathbf t_A = \mathbf t_s U^{-1}, \qquad
R_A = U^{-\mathsf T} R_s U^{\mathsf T}.
$$

$n$ 阶张量从 lattice 分量到 Cartesian 分量的映射只有一个入口：

$$
K_n = (A^{\mathsf T})^{\otimes n}.
$$

每个 orbit 的锚定 cluster 有一组稳定子 $S_a$。以 0/1 指标基 $L$（把同类原子互换下的等价分量并成
一列）表示时，不变子空间是

$$
V_{\mathbb Z} = \{\, x : (S_a L - L)x = 0 \ \ \forall a \,\} \subset \mathbb Z^{3^n},
$$

它由**饱和整数基** $B_{\mathbb Z}$ 张成。物理张量子空间则由同一次基变换给出 $C = K_n B_{\mathbb Z}$，
再取对角元为正的 QR 分解

$$
C = QR, \qquad Q^{\mathsf T}Q = I .
$$

轨道因此携带两套基，用途严格分开：

| 名称 | 含义 | 用途 |
|---|---|---|
| `exact_lattice_basis` | 整数基 $B_{\mathbb Z}$ | 群代数、秩证书、不变量维数、来源追踪 |
| `cartesian_basis` | 正交基 $Q$ | 拟合、有限差分重建、ASR 与其他约束、力常数展开 |
| `coefficient_transform` | 上三角 $R$ | 精确系数与数值系数之间的换算 $c = R^{-1}\theta$ |

拟合参数 $\theta$ 只有一种坐标含义：它是 $Q$ 的系数，代表张量为 $Q\theta$；精确整数系数
$c = R^{-1}\theta$ 只是同一物理张量在 lattice 参考系里的记录。

有限差分计划需要知道"测哪些分量"。这些行记作 `observation_rows`，取自 $Q$（按体积最大化的贪心选择），
观测矩阵为 $Q_{\mathrm{obs}} = Q[\text{rows}]$，重建显式求解

$$
Q_{\mathrm{obs}}\,\theta = y_{\mathrm{obs}},
$$

并记录 $Q_{\mathrm{obs}}$ 的 2-范数条件数。它们是**观测行**，不是参数 pivot：观测分量并不等于参数。

## 在 MLFCS 中的实现

* `LatticeFrame` 保存 $A_s$、$A$、$U$、$U^{-1}$、motif 映射与 $K_n$，并给出分数坐标、整数平移、
  旋转矩阵在两个参考系间的精确换算。规约采用 Minkowski reduction，并在等长、符号与轴置换退化时用
  "晶胞 + motif" 稳定键在有限个 signed permutation 候选中选出唯一规范表示，因此等价的
  unimodular 写法得到**同一套整数**。
* `interactions/algebra/exact.py` 提供模素数秩证书与饱和整数核：模 $p$ 秩是 $\mathbb Q$ 秩的下界，
  某个素数上满秩即为证明；秩亏则由"已用素数乘积超过最大子式的 Hadamard 上界"证书判定。核由
  Smith 标准形构造，因此是饱和的整数核，而不是逐列除以最大公约数。
* `interactions/algebra/invariants.py` 堆叠稳定子残差约束，用证书给出维数，返回整数基并**精确验证**
  约束矩阵乘基为零。
* orbit 遍历在代数参考系中进行，轨道成员在写回时通过 `LatticeFrame.source_labels` 精确映射回用户晶胞的
  site 与整数平移，因此 reference 超胞索引仍然指向同一个物理 interaction。
* Cartesian 渲染只有一个入口 `interactions/algebra/rendering.py`：`frame @ basis` 不再出现在任何消费者里。
  拟合设计核、有限差分重建、ASR 与力常数展开都消费 $Q$。
* realization identifiability 直接对整数 realization 矩阵做秩证书，不再有系数阈值或秩容差。

## 数值注意事项

* 精确的部分与浮点的部分被刻意分开：秩、核、稳定子不变性在整数域判定；只有 Cartesian 渲染、QR 与
  观测行选择使用浮点，且都作用在 $O(1)$ 的数值基上。
* 条件数随 orbit 一起保存（`observation_condition`）。由于等价表示得到逐位相同的整数代数，它不会随原始
  晶胞的 shear 恶化；四阶问题实测不超过约 5。
* lattice 整数进入张量运算前会做显式的 int64 范围检查，超出范围抛出 `IntegerRangeError` 并指出是哪个量，
  而不是让 NumPy 静默截断或在中途抛出含混的转换错误。
* 秩证书的素数流是**按需**的：SymPy `prevprime` 提供不同素数，累计乘积与 Hadamard 上界比较后终止。它不是
  数学意义上的无限素数枚举；在可用素数内无法终止时抛出 `RankCertificateError`。任意精度只覆盖秩与核
  这一层，端到端并不承诺 arbitrary precision。

## 验证方式

* 同一晶体在 adjacent、crossed 与 large shear 下的等价表示（立方、fcc primitive、六方、倾斜晶胞，二至四阶）
  必须给出相同的整数基、Cartesian 子空间、观测行、条件数、展开后的力常数与有限差分重建结果。
* 稳定子约束在源码参考系中独立枚举（用浮点秩作 oracle）交叉验证不变子空间。
* 精确秩以 SymPy 为 oracle，覆盖零矩阵、秩亏、坏素数、$2^{63}$ 与 $10^{30}$ 量级整数。
* 解析 Morse 势的 FC4 oracle 与教程案例固定物理内容：重整换基不得改变交付的力常数。

## 原胞、显式超胞与单一精度

轨道与相互作用代数在**用户提供的**原胞上进行，参考超胞必须由用户显式给出，主线不会根据 cutoff 猜测或
隐式构造。原胞与超胞之间的整数复制关系、超胞原子到「原胞原子 + 整数格矢」的映射、以及 spglib 的原胞
对称性识别，统一使用唯一的长度精度 `symprec`（单位 Å）；无量纲矩阵元素、分数坐标差或角度都不与它比较。
`mlfcs.tools.supercell.build_supercell` 只是可选的便利构造工具，主线不依赖它。
