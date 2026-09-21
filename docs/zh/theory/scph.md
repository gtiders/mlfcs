---
title: Loop-SCPH
audience:
  - advanced
status: experimental
code_verified: 4.0.0a6
---

# Loop-SCPH

`LoopSCPH` 对 FC2 应用静态四阶 loop 修正，返回与温度相关的有效 FC2。这里的 SCPH 是
静态的最低阶四阶 loop 近似，不包含频率相关的三阶 bubble 自能，也不等同于 SSCHA 的
变分自由能优化。FC2 和 FC4 是两个独立的 `ForceConstants` 对象，但必须描述同一个
primitive/reference frame。

单个温度的计算保持标量接口：

```python
result = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=600,
    interpolation_multiplier=1, scph_multiplier=2,
    mixing=0.1, tolerance=1e-10, max_iterations=100,
).run()
```

`result.force_constants` 是一个只含 FC2 的标准 `ForceConstants` 对象。因此无需
SCPH 专用转换，直接使用普通 writer 即可写出：

```python
write_force_constants(result.force_constants, "scph.h5", format="hdf5")
write_force_constants(result.force_constants, "FORCE_CONSTANTS_SCPH", format="phonopy")
write_force_constants(result.force_constants, "force_constants.xml", format="alamode")
```

q 点网格由 reference 超胞矩阵的整数倍自动生成：
`interpolation_multiplier` 控制输出频率网格，`scph_multiplier` 控制 loop 协方差积分网格；后者必须是前者的整数倍。

网格本身仍然是精确的有限平移群商群，但**对角化只发生在不可约代表点上**。空间群中保持该网格的操作把
网格分成星（star），每个星由一个代表点和若干成员组成；成员与其代表点由空间群表示
$U_g(q)$ 联系：

$$
 D(gq)=U_g(q)\,D(q)\,U_g(q)^\dagger,\qquad
 W(gq_s)=U_g\,W(q_s)\,U_g^\dagger,
$$

其中 $W(q)=V\operatorname{diag}(\sigma^2)V^\dagger$ 是基底无关的协方差矩阵；成员若通过时间反演到达，
则先取复共轭。频率与任何谱量在整个星上相同，因此完整网格由代表点数据**精确展开**得到，
展开过程不会再调用任何对角化器。位置规范把代表点的**未约化**像与成员自身标签之间相差的原始倒格矢平移
变成一个对角因子 $\Gamma_G=\operatorname{diag}(\exp[2\pi i\,G\cdot\tau_a])$，协方差展开必须同时应用它，
否则矩阵与 Fourier 相位会在标签确实被约化的成员上相差一个量级为 1 的因子。

## 1. 从晶格哈密顿量到 loop 方程

以原胞中的原子 site 和晶格平移为标签，位移记为
`u_{a\alpha}(R)`。MLFCS 的泰勒力常数采用如下约定：

$$
 U = U_0 + \frac{1}{2!}\sum_{1,2}\Phi^{(2)}_{1,2}u_1u_2
     + \frac{1}{4!}\sum_{1,2,3,4}\Phi^{(4)}_{1,2,3,4}u_1u_2u_3u_4+\cdots .
$$

数字 `1=(a,\alpha,R)` 同时包含原子 site、笛卡尔分量和晶格平移。对参考谐振子
进行高斯平均时，四阶项的二阶导数为

$$
 \frac{\partial^2}{\partial u_1\partial u_2}
 \left(\frac{1}{4!}\Phi^{(4)}_{1,2,3,4}u_1u_2u_3u_4\right)
 = \frac{1}{2}\Phi^{(4)}_{1,2,3,4}u_3u_4 .
$$

因此静态 loop 的有效二阶力常数满足固定点方程

$$
 \widetilde\Phi^{(2)}_{1,2}(T)
 = \Phi^{(2)}_{1,2}
 + \frac{1}{2}\sum_{3,4}
 \Phi^{(4)}_{1,2,3,4}\,C_{3,4}(T;\widetilde\Phi^{(2)}),
$$

其中 `C=<u u>` 是由当前有效二阶力常数自洽计算的位移协方差。这个 `1/2` 来自
`4!` 的组合因子，不是经验参数。

## 2. 协方差和动力学矩阵

由当前的有效 FC2 构造质量归一化动力学矩阵：

$$
 D_{a\alpha,b\beta}(q)=
 \frac{1}{\sqrt{m_am_b}}
 \sum_R \widetilde\Phi^{(2)}_{a\alpha,b\beta}(R)
 e^{2\pi i q\cdot(R+r_b-r_a)} .
$$

对每个 q 点对 `D(q)` 对角化，令本征值为 `\lambda_{q\nu}`，本征矢为
`e_{a\alpha,q\nu}`。代码内部的 `mode_sigma` 返回质量归一化模坐标的均方根，满足

$$
 \sigma^2_{q\nu}=
 \begin{cases}
 \dfrac{\hbar}{2\omega_{q\nu}}, & T=0,\\[6pt]
 \dfrac{\hbar}{2\omega_{q\nu}}
 \coth\!\left(\dfrac{\hbar\omega_{q\nu}}{2k_BT}\right), & T>0
 \end{cases}
$$

量子统计下使用上式，经典统计下使用 `k_BT/\omega^2`。于是实空间协方差为

$$
 C_{a\alpha,b\beta}(R)=\frac{1}{N_q}\sum_q\sum_\nu
 \frac{e_{a\alpha,q\nu}e^*_{b\beta,q\nu}\sigma^2_{q\nu}}
 {\sqrt{m_am_b}}
 e^{2\pi i q\cdot(r_a-r_b+R)} .
$$

零频模由 `frequency_cutoff_thz` 排除；负本征值只作为不稳定性的诊断，当前实现对其使用
`|\lambda|` 计算模幅，因此“出现虚频”不会被误当作固定点收敛条件。

## 3. Fourier 变换和相位约定

设 primitive 的三个晶格矢量按行组成矩阵 `A`，site `a` 在 primitive 中的笛卡尔位置为
`r_a`。代码中的 q 是倒格子的分数坐标，因此对应的笛卡尔波矢为

$$
 k(q)=2\pi A^{-T}q,
 \qquad
 e^{i k\cdot d}=e^{2\pi i q\cdot d_f},
$$

其中 `d_f` 是位移 `d` 在 primitive 基底中的分数坐标。MLFCS 的 FC2 物理条目写成

$$
 \Phi^{(2)}_{a\alpha,b\beta}(R),
 \qquad
 d_{ab}(R)=r_b-r_a+R A,
$$

这里 `R` 是从 site `a` 所在原胞指向 site `b` 所在原胞的整数平移。实空间到 q 空间的
约定为

$$
 D_{a\alpha,b\beta}(q)=
 \frac{1}{\sqrt{m_am_b}}
 \sum_R \Phi^{(2)}_{a\alpha,b\beta}(R)
 e^{+2\pi i q\cdot d_{ab,f}(R)} .
$$

在相同的有限 q 网格上，反变换的符号相反：

$$
 \Phi^{(2)}_{a\alpha,b\beta}(R)
 =\frac{\sqrt{m_am_b}}{N_q}\sum_q D_{a\alpha,b\beta}(q)
 e^{-2\pi i q\cdot d_{ab,f}(R)} .
$$

实际的 `D(q)` 是 Hermitian 化后的矩阵，因此采用整体相反的 Fourier 符号不会改变频率，
但正变换、协方差和导出必须使用同一套约定。当前代码在
`_fourier_terms` 中计算 `d_f`，在 `_dynamical`/`_dynamical_batch` 中使用
`exp(+2j*pi*(q @ d_f))`；协方差采用与其相容的
`r_a-r_b+R` 相位。

## 4. 有限超胞的周期性

若 reference supercell 矩阵为整数矩阵 `S`，则超胞中的平移属于商群

$$
 R\sim R+nS,\qquad n\in\mathbb Z^3 .
$$

有限超胞 realization 时，`PeriodicIndex` 使用 `(primitive_site, residue)` 做 O(1) 原子查找；
但 canonical IFC 保存的是 exact primitive translation，而不是 residue。`_fourier_terms` 直接计算

```text
R_exact = translations[row]
d_f = (r_site - r_first) @ A^{-1} + R_exact
```

因此 Fourier 相位不依赖原子编号、reference 排列或商群代表元。训练超胞只决定哪些 exact
interaction 可被辨识；拟合完成后的 primitive 实空间 IFC 可在任意 q 点使用。

协方差使用同一周期约定。FC4 条目的内部两个腿为 `(s3,R3)` 和 `(s4,R4)` 时，代码只需
计算相对平移 `R3-R4`：

$$
 C_{s_3s_4}(R_3-R_4)=\frac1{N_q}\sum_q
 \sum_\nu \frac{e_{s_3,q\nu}e^*_{s_4,q\nu}\sigma^2_{q\nu}}
 {\sqrt{m_{s_3}m_{s_4}}}
 e^{2\pi i q\cdot(r_{s_3}-r_{s_4}+R_3-R_4)} .
$$

这使得 loop 收缩只依赖 lattice-labelled 物理键，而不依赖 FC4 条目在 reference 中的
存储顺序。

## 5. 与 MLFCS 稀疏数据结构的对应

每个 FC4 稀疏条目保存

```text
sites = (s1, s2, s3, s4)
translations = (R2, R3, R4)
tensors = Phi4(s1, 0; s2, R2; s3, R3; s4, R4)
```

第一个原子是零平移锚点。对一个 FC4 条目，代码取

```text
C = covariance[(s3, s4, R3 - R4)]
DeltaPhi2 = 0.5 * einsum("abcd,cd->ab", Phi4, C)
```

并把结果写入 `(s1, s2, R2)` 对应的 FC2 支撑。`PeriodicIndex.atom(site, R)` 只负责把物理
site/平移标签映射回 reference supercell 原子号；它不改变用户 reference 的原子顺序。
因此，稀疏物理标签、计算用超胞索引和最终导出顺序彼此分离。

协方差的 q 点和为 `1/N_q`，`scph_multiplier` 是协方差积分网格相对 reference 超胞的倍数，`interpolation_multiplier` 仅用于
频率变化的收敛诊断和结果频率输出。两者可以不同，但当前要求 `scph_multiplier` 是
`interpolation_multiplier` 的整数倍。

## 6. 自洽迭代

从裸 FC2（或用户提供的 `warm_start`）开始：

1. 用当前 FC2 在**不可约代表点**上构造 `D(q_s)`，对角化并求 `C(T)`；协方差按第 5 节的星展开
   求和到完整网格；
2. 对所有 FC4 条目执行四阶张量与协方差的 loop 收缩；
3. 得到新的目标 `Phi2_target = Phi2_bare + DeltaPhi2`；
4. 用星权重下的完整网格频率 RMS
   $\Delta\omega=\sqrt{\frac{1}{N_qN_b}\sum_s w_s\lVert\omega_s^{(n)}-\omega_s^{(n-1)}\rVert_2^2}$
   判断固定点是否收敛；它与「先展开到完整网格再取均方根」等价。

星权重 $w_s$ 是星的大小，$\sum_s w_s=N_q$。协方差求和是**真实的完整求和**：星权重只用于统计量
（自由能、模态计数、收敛范数），绝不会用来缩放代表点的矩阵，因为星成员之间相差的是旋转而不是标量。

`mixing` 对连续两次协方差做欠松弛：

$$
 C^{\mathrm{used}}_n = \eta C_n+(1-\eta)C_{n-1},
 \qquad \eta=\texttt{mixing}.
$$

它只改变固定点迭代的数值路径，不改变最终方程；`mixing=1` 是无欠松弛的直接固定点
迭代。多温度调用 `temperature=range(T_start, T_stop, step)` 时，前一温度得到的有效 FC2
作为下一温度的 `warm_start`，从而实现温度延续。

## 7. 性能与验证

`scripts/benchmark_reciprocal_reduction.py` 可复现下面的对比（对角化次数按**矩阵个数**统计，
不是调用次数；一次批量 `eigh` 是 `N_irr` 次对角化）。体系为各向同性弹簧网络 FC2 加 onsite 四次项，
参考超胞见下表，`interpolation_multiplier=2`：

| 体系 | $N_q$ | $N_{\mathrm{irr}}$ | 约化比 | 频率对角化 | 单轮 SCPH 对角化 |
|---|---|---|---|---|---|
| diamond 2×2×2 | 64 | 8 | 8.00 | 8（全网格 64） | 9（全网格 24） |
| hcp 2×2×2 | 64 | 12 | 5.33 | 12（全网格 64） | 12（全网格 24） |
| GaAs 3×2×2 | 96 | 34 | 2.82 | 34（全网格 96） | 18（全网格 36） |

合并门槛是「对角化次数严格等于 $N_{\mathrm{irr}}$」，而不是墙钟时间：小体系的首次调用还要支付一次性的
对称性分析，因此可能比全网格更慢；低对称体系可能几乎没有加速，这是正确结果。

## 8. 适用范围和诊断

该实现输出一个静态、温度依赖的有效 FC2，可以继续交给 MLFCS 的 writer 或声子工具。
它不生成频率依赖的自能，也不把 FC3 bubble 自动加入 FC2。若结果未收敛，应优先检查
`result.history`、最终频率变化、负频率数量以及 FC2/FC4 是否来自严格相同的
primitive/reference 关系。

```python
results = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=[300, 600, 900],
    interpolation_multiplier=1, scph_multiplier=2,
    max_iterations=100,
).run()
```


