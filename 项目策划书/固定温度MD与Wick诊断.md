# 固定温度 MD、Wick 基底与跨阶耦合诊断

## 1. 固定温度下的位移分布为什么接近 Gaussian

考虑一个包含 $N$ 个原子的晶体，其位移向量为：

$$
\mathbf u\in\mathbb R^{3N}.
$$

在固定温度 $T$ 的正则系综中，位移构型的概率分布满足：

$$
P(\mathbf u)\propto\exp[-\beta U(\mathbf u)],
\qquad
\beta=\frac{1}{k_B T}.
$$

如果处于理想谐波极限，势能为：

$$
U(\mathbf u)=\frac12\mathbf u^{\mathsf T}\Phi^{(2)}\mathbf u.
$$

于是：

$$
P(\mathbf u)\propto
\exp\left[-\frac{\beta}{2}
\mathbf u^{\mathsf T}\Phi^{(2)}\mathbf u\right],
$$

这正是多元 Gaussian 分布：

$$
\mathbf u\sim\mathcal N(0,C),
\qquad
C=\langle\mathbf u\mathbf u^{\mathsf T}\rangle.
$$

经典谐波极限下，在去除整体平移等零频模后，近似有：

$$
C\approx k_B T\left(\Phi^{(2)}\right)^{-1}.
$$

因此，在固定温度、单势阱、弱非谐和充分遍历的条件下，MD 位移分布通常可以很好地近似为多元 Gaussian。这里需要注意：严格的 NVT 采样对应正则分布；NVE 轨迹只有在体系足够大、混合充分且能量对应的温度波动较小时，才可以把时间平均近似为固定温度的统计平均。

## 2. 真实非谐体系不严格服从 Gaussian

真实势能通常为：

$$
U(\mathbf u)=
\frac12\Phi^{(2)}uu
+\frac1{3!}\Phi^{(3)}uuu
+\frac1{4!}\Phi^{(4)}uuuu+\cdots.
$$

因此：

$$
P(\mathbf u)\propto e^{-\beta U(\mathbf u)}
$$

一般不再严格是 Gaussian。典型偏离包括：

- FC3 使分布产生偏斜；
- 强 FC4 改变分布峰度；
- 软模造成宽分布和长尾；
- 双势阱造成双峰分布；
- 相变附近出现强烈的 mode-mode correlation。

所以 Gaussian 是谐波极限的严格结果，是弱非谐体系的近似，而不是所有晶体的普遍定理。

## 3. Wick 基底为什么适合固定温度 MD

对于参考 Gaussian measure：

$$
\mathbf u\sim\mathcal N(0,C),
$$

Wick polynomial，也可看作与该 covariance 匹配的多元 Hermite polynomial，是自然的正交多项式体系。二阶和三阶例子为：

$$
:u_i u_j:=u_i u_j-C_{ij},
$$

$$
:u_i u_j u_k:
=u_i u_j u_k
-C_{ij}u_k-C_{ik}u_j-C_{jk}u_i.
$$

Wick 变换做的事情可以概括为：把普通 Taylor monomial 中由 Gaussian covariance 自动产生的低阶 contraction 剥离出来。

如果 MD 数据接近 $\mathcal N(0,C)$，用 MD 数据本身估计的 $C$ 构造 Wick basis，就相当于选择了一套与训练数据统计分布匹配的多项式坐标系。

## 4. Wick 没有消灭高阶对低阶的影响

一维情况下：

$$
:u^3:=u^3-3\sigma^2u,
\qquad
u^3=:u^3:+3\sigma^2u.
$$

同样：

$$
:u^4:=u^4-6\sigma^2u^2+3\sigma^4,
$$

$$
u^4=:u^4:+6\sigma^2:u^2:+3\sigma^4.
$$

因此高阶 Wick coefficient 在变回 Taylor 表示时，会通过 covariance contraction 回流到低阶。例如多维情况下会出现：

$$
C_{ij}\Phi^{(4)}_{ijkl}
$$

形式的收缩，也就是 $\mathrm{FC4}\rightarrow\mathrm{FC2}$ 的贡献。Wick 没有让高阶和低阶真正互不影响，而是把普通 Taylor 基底中隐藏的跨阶竞争，转化为显式、可计算的 covariance contraction。

## 5. FC4 回流到 FC2 不等于分布很不 Gaussian

即使 MD 位移分布完全 Gaussian，只要 covariance $C$ 较大、四阶力常数 $\Phi^{(4)}$ 较强，收缩 $C:\Phi^{(4)}$ 仍然可能很大。因此：

$$
\text{large }\mathrm{FC4}\rightarrow\mathrm{FC2}
\ne
\text{strong non-Gaussianity}.
$$

它可能只说明温度较高、位移涨落较强、四阶非谐性较大，因而有效二阶刚度发生明显重整化。这与有限温度声子理论中的结构相同：

$$
\Phi^{(2)}_{\mathrm{eff}}
\sim
\Phi^{(2)}+\Phi^{(4)}\langle uu\rangle+\cdots.
$$

因此 Wick 到 Taylor 的高阶向低阶回流，更准确地反映的是高阶非线性通过当前温度下的涨落 covariance 对低阶响应产生了多强的重整化。

## 6. 真正的非 Gaussian 信息在哪里

如果位移分布严格 Gaussian，四阶矩满足 Wick theorem：

$$
\langle u_i u_j u_k u_l\rangle
=C_{ij}C_{kl}+C_{ik}C_{jl}+C_{il}C_{jk}.
$$

因此四阶 connected cumulant 为零。实际诊断量定义为：

$$
\kappa^{(4)}_{ijkl}
=
\langle u_i u_j u_k u_l\rangle
-C_{ij}C_{kl}-C_{ik}C_{jl}-C_{il}C_{jk}.
$$

如果 $\kappa^{(4)}\ne0$，说明只靠 covariance 已经不能完全描述四阶位移统计。高阶统计因此可以分成两部分：

$$
\text{Gaussian contraction}
+
\text{connected non-Gaussian fluctuation}.
$$

Wick transformation 能明确处理前者；后者会表现为 Wick 化后仍然残留的高阶统计结构。类似地，可以继续计算 $\kappa^{(6)}$ 等更高阶 cumulant。

## 7. Wick 后残余跨阶相关如何观测

将 Wick 设计矩阵按阶分块：

$$
X_{\mathrm W}=\left[
X_{\mathrm W}^{(2)}\mid
X_{\mathrm W}^{(3)}\mid
X_{\mathrm W}^{(4)}\mid\cdots
\right].
$$

Gram 矩阵为：

$$
G_{\mathrm W}=X_{\mathrm W}^{\mathsf T}X_{\mathrm W}
=
\begin{pmatrix}
G_{22}&G_{23}&G_{24}&\cdots\\
G_{32}&G_{33}&G_{34}&\cdots\\
G_{42}&G_{43}&G_{44}&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{pmatrix}.
$$

其中：

$$
G_{24}=(X_{\mathrm W}^{(2)})^{\mathsf T}X_{\mathrm W}^{(4)}
$$

就是 FC2 与 FC4 Wick feature 之间的残余 cross-order correlation。如果 Wick reference measure 与实际训练数据非常匹配，理想情况下 $G_{24}\approx0$；若仍明显非零，说明 Wick feature subspace 仍未真正解耦。

这种诊断适合 Gram-based solver，因为可以使用 streaming Gram accumulation，不需要显式保存完整设计矩阵。

## 8. Frobenius 归一化的残余指标

为了避免不同 block 的数值尺度直接决定诊断结果，可以定义：

$$
\rho^{\mathrm F}_{pq}
=
\frac{\lVert G_{pq}\rVert_F}
{\sqrt{\lVert G_{pp}\rVert_F\lVert G_{qq}\rVert_F}}.
$$

例如：

$$
\rho^{\mathrm F}_{24}
=
\frac{\lVert G_{24}\rVert_F}
{\sqrt{\lVert G_{22}\rVert_F\lVert G_{44}\rVert_F}}.
$$

如果 $\rho^{\mathrm F}_{24}\approx0$，说明二阶和四阶 Wick feature 基本解耦；如果该值较大，则仍存在明显跨阶相关。应同时报告 block 的规模和有效秩，避免只看一个归一化数值。

## 9. 白化后的 canonical correlation

Frobenius 范数仍会受到 block 内部不同方向尺度的影响。更严格的指标是对白化后的跨阶 block 做奇异值分析：

$$
R_{pq}=G_{pp}^{-1/2}G_{pq}G_{qq}^{-1/2}.
$$

在奇异或近奇异 block 上，应使用由有效奇异值构造的伪逆平方根，而不是无条件求逆。定义最大残余 canonical correlation：

$$
\rho^{\max}_{pq}=\sigma_{\max}(R_{pq}).
$$

它回答的是：所有 $p$ 阶 Wick feature 线性组合和所有 $q$ 阶 Wick feature 线性组合之间，最强的残余相关有多大。

- $\rho^{\max}_{pq}=0$ 表示两个 feature subspace 完全正交；
- $\rho^{\max}_{pq}\rightarrow1$ 表示存在几乎线性相关的一对组合。

这就是 residual cross-order canonical correlation。对于基于 Gram 系统的 MLFCS，它是比单纯 Frobenius 范数更严格的诊断量，但需要先报告 block 的有效秩和截断容差。

## 10. Taylor 与 Wick 的前后对照

可以分别计算 Taylor 和 Wick basis 下的跨阶指标：

$$
\rho^{\mathrm T}_{pq},\qquad \rho^{\mathrm W}_{pq}.
$$

其中 $\rho^{\mathrm T}_{pq}$ 是 Taylor basis 的相关性，$\rho^{\mathrm W}_{pq}$ 是 Wick basis 的剩余相关性。可以定义 decorrelation efficiency：

$$
\eta_{pq}=1-\frac{\rho^{\mathrm W}_{pq}}{\rho^{\mathrm T}_{pq}}.
$$

当 $\rho^{\mathrm T}_{pq}$ 非零且 $\eta_{pq}\approx1$ 时，表示 Wick 对这两阶的 feature correlation 改善明显；$\eta_{pq}\approx0$ 表示基本没有改善。若 Taylor 指标本身接近零，则效率比值不稳定，应直接报告两组原始指标而不强行解释 $\eta_{pq}$。

## 11. Cumulant 与 Gram residual 不是同一个问题

四阶 cumulant $\kappa^{(4)}$ 回答：位移概率分布偏离 Gaussian 的程度有多大？

Wick Gram cross block $G_{24}^{\mathrm W}$ 回答：这种统计结构最终有没有在实际拟合特征中产生 FC2 与 FC4 的残余相关？

因此：

$$
\text{cumulant}=\text{distribution diagnostic},
$$

$$
\text{Wick Gram residual}=\text{fitting diagnostic}.
$$

一个分布可以有较大的非 Gaussian cumulant，但对应的拟合特征残余相关不一定最大；反过来，有限样本、模型截断或特征尺度问题也可能造成明显 Gram residual，而不代表概率分布本身强烈非 Gaussian。两类指标应同时使用，不能互相替代。

## 12. 高阶向低阶的重整化强度

Wick 到 Taylor 逆变换时，可以直接测量高阶 contribution 对低阶 IFC 的回流。例如定义：

$$
R_{4\rightarrow2}
=
\frac{\lVert\Delta\Phi^{(2)}_{\leftarrow4}\rVert}
{\lVert\Phi^{(2)}_{\mathrm{Taylor}}\rVert}.
$$

它衡量当前 covariance 下，四阶信息有多大比例回代进二阶 Taylor IFC。为了与非 Gaussian 程度区分，还可以定义归一化非 Gaussian 指标：

$$
\eta_4
=
\frac{\lVert\kappa^{(4)}\rVert}
{\lVert C\otimes C\rVert}.
$$

这两个量可以区分几种情况：

### 情况一：重整化强但接近 Gaussian

$$
R_{4\rightarrow2}\gg0,\qquad \eta_4\ll1.
$$

说明分布仍接近 Gaussian，但四阶非谐性和热涨落都很强，存在明显有限温度重整化。

### 情况二：重整化强且 Gaussian reference 失效

$$
R_{4\rightarrow2}\gg0,\qquad \eta_4\gg0.
$$

说明高阶重整化很强，同时 Gaussian reference 本身已经明显失效。

### 情况三：弱非谐、接近 Gaussian

$$
R_{4\rightarrow2}\ll1,\qquad \eta_4\ll1.
$$

说明高阶向低阶的回流和非 Gaussian 偏离都较弱。

这些不是自动判定材料性质的硬阈值，而是需要结合温度、采样量、单位、有效秩和误差条一起解释的诊断坐标。

## 13. Wick 使隐藏矛盾变得可观测

普通 Taylor basis 中，$u$、$u^3$、$u^5$ 等特征可能高度相关，导致跨阶 IFC 在最小二乘中竞争解释同一部分 force。这种问题隐藏在：

$$
X_{\mathrm T}^{\mathsf T}X_{\mathrm T}
$$

的非对角 block 中。Wick transformation 通过 covariance contraction 重新组织这种关系：

$$
\text{Taylor：跨阶关系隐藏在设计矩阵相关性中},
$$

$$
\text{Wick：跨阶关系显式进入 covariance-dependent transformation}.
$$

因此 Wick 并没有让不同阶真正彼此无关，而是把不可控的隐式 mixing 转化为可计算、可追踪的显式 mixing。这样才有可能区分：

- Gaussian contraction；
- connected non-Gaussian residual；
- cluster 或 body cutoff 造成的模型截断；
- 数据量不足和位移方向不可辨识；
- feature conditioning 问题。

## 14. 对 MLFCS 的诊断体系

如果把这套思想落实到 MLFCS，至少应报告以下四类量：

1. 位移 covariance：

   $$
   C=\langle\mathbf u\mathbf u^{\mathsf T}\rangle;
   $$

2. Taylor 和 Wick basis 下的跨阶指标：

   $$
   \rho^{\mathrm T}_{pq},\qquad \rho^{\mathrm W}_{pq};
   $$

3. Wick 到 Taylor 逆变换中的高阶回流强度：

   $$
   R_{q\rightarrow p};
   $$

4. 判断采样分布偏离 Gaussian 程度的 cumulant：

   $$
   \kappa^{(4)},\qquad \kappa^{(6)},\ldots.
   $$

这样一句模糊的“FC2 和 FC4 好像互相污染得很严重”，就可以拆成几个独立问题：

1. Taylor basis 本身有多相关？
2. Wick 去掉了多少统计相关？
3. Wick 后剩余跨阶相关有多少？
4. 高阶向低阶的 covariance renormalization 有多强？
5. 实际位移分布有多偏离 Gaussian？
6. 这些指标的不确定性是否只是有限样本或数值秩判定造成的？

## 总结

固定温度 MD 为 Wick 基底提供自然统计背景：

$$
\text{谐波极限}\Rightarrow\text{Gaussian displacement distribution}.
$$

Wick polynomial 是 Gaussian measure 下自然的正交多项式。它不消灭非谐性，也不让 FC2、FC3、FC4 真正互不影响，而是先剥离 Gaussian covariance 能解释的 contraction，把剩余结构暴露出来。

高阶向低阶的 Wick 到 Taylor 回流 $\mathrm{FC4}\rightarrow\mathrm{FC2}$ 主要反映 covariance-controlled renormalization；Wick 后仍然存在的 $G_{24}^{\mathrm W}\ne0$ 与 $\kappa^{(4)}\ne0$，则分别用于判断残余拟合耦合和非 Gaussian 统计。

整套设计可以概括为：

> Wick 不消灭跨阶矛盾，而是把原本隐藏在 Taylor 拟合病态性中的矛盾，分解为可解释的 covariance contraction 和可测量的 residual correlation。正因为这些耦合能够被显式观测，求解器才有机会得到更好的条件数、更稳定的参数识别和更清楚的非谐性诊断。
