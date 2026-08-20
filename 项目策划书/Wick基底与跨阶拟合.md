# Wick 基底与高阶力常数的跨阶拟合

## 1. 为什么需要 Wick 基底

普通 Taylor 拟合直接使用位移单项式作为特征：

$$
u,\qquad uu,\qquad uuu,\qquad uuuu,\ldots
$$

这种写法物理含义直观，但有限训练数据中的不同阶特征通常并不正交。以一维位移为例，$u$ 和 $u^3$ 的统计相关性为：

$$
\langle u\,u^3\rangle=\langle u^4\rangle\ne0.
$$

因此同时拟合多个 Taylor 阶数时，不同阶参数可能竞争解释同一部分力信号，表现为：

- 不同阶 IFC 参数相互漂移；
- 高阶项吸收低阶误差；
- 低阶项补偿未充分采样的高阶贡献；
- Gram 矩阵出现明显的跨阶非对角块；
- 设计矩阵条件数恶化；
- 结果对噪声、采样分布和正则化参数更加敏感。

这种现象可以称为跨阶混合或跨阶污染。它不是不同物理阶数真的互相矛盾，而是有限采样下不同阶特征在数值上高度相关。

## 2. Wick 没有删除物理耦合

在固定训练数据、固定 covariance、固定最高展开阶数，并且基底变换可逆的条件下，Taylor 基底和 Wick 基底表示的是同一个多项式空间。设 Taylor 设计矩阵为 $X_{\mathrm T}$，Wick 设计矩阵为 $X_{\mathrm W}$，则：

$$
X_{\mathrm W}=X_{\mathrm T}T,
$$

其中 $T$ 是由 covariance 决定的基底变换矩阵。存在相应的参数变换，使得：

$$
X_{\mathrm T}\theta_{\mathrm T}
=
X_{\mathrm W}\theta_{\mathrm W}.
$$

因此 Wick 基底不增加新的物理信息，也不删除原有的跨阶物理耦合；它改变的是同一模型空间的坐标表示。只有在有限阶截断、协方差不准确或变换矩阵数值退化时，Taylor 与 Wick 的实际拟合空间才可能出现差异，这属于模型或数据条件造成的差异，而不是 Wick 基底凭空改变了势能面。

## 3. Wick 多项式如何显式表示 contraction

设位移均值为零，协方差为：

$$
C_{ij}=\langle u_i u_j\rangle.
$$

三阶 Wick 多项式定义为：

$$
:u_i u_j u_k:
=
u_i u_j u_k
-C_{ij}u_k
-C_{ik}u_j
-C_{jk}u_i.
$$

普通三阶单项式因此可以写成：

$$
u_i u_j u_k
=
:u_i u_j u_k:
+C_{ij}u_k+C_{ik}u_j+C_{jk}u_i.
$$

右侧包含两部分：一个真正的三阶 fluctuation，以及 covariance 与一阶特征的 contraction。Wick 基底并不是说三阶和一阶从此互不影响，而是把这种影响写成了明确的 covariance contraction。

一维情况下：

$$
:u^3:=u^3-3\sigma^2u,
$$

所以：

$$
u^3=:u^3:+3\sigma^2u.
$$

如果力模型为：

$$
F(u)=-ku-\frac{g}{6}u^3,
$$

转换到 Wick 基底后得到：

$$
F(u)=
-\left(k+\frac{g}{2}\sigma^2\right)u
-\frac{g}{6}:u^3:.
$$

高阶系数 $g$ 对低阶响应的影响没有消失，而是通过 $\sigma^2$ 显式进入低阶系数。因此更准确的说法是：Wick 重新定位跨阶耦合，而不是消灭跨阶耦合。

## 4. 从隐藏参数竞争到可观测统计结构

Taylor 基底下，如果 FC2 和 FC4 同时变化，很难仅凭最终参数判断这种变化来自真实的高阶重整化，还是来自设计矩阵列相关导致的参数补偿。例如可能观察到：

$$
\theta_2\uparrow,\qquad \theta_4\downarrow.
$$

Wick 变换后，可以直接检查 covariance $C_{ij}$，以及它如何把高阶特征 contraction 到低阶子空间。不同阶之间的 Gram block 也可以定量诊断：

$$
G=X^{\mathsf T}X.
$$

设 $G_{pq}$ 是 $p$ 阶和 $q$ 阶特征之间的 Gram block，可以定义归一化耦合强度：

$$
\rho_{pq}=
\frac{\lVert G_{pq}\rVert_F}
{\sqrt{\lVert G_{pp}\rVert_F\lVert G_{qq}\rVert_F}}.
$$

Taylor 基底中可能有较大的 $\rho_{24}$；Wick 化后，若 covariance 与采样分布匹配，$\rho_{24}$ 通常会显著降低。这样“跨阶污染有多强”就从一个只能猜测的参数漂移问题，变成了可以测量的统计结构。

## 5. Wick 如何改善最小二乘的数值稳定性

线性拟合问题为：

$$
\min_\theta\lVert X\theta-y\rVert_2^2.
$$

如果不同列近似线性相关，例如：

$$
X_i\approx aX_j,
$$

设计矩阵会出现很小的奇异值，条件数变大：

$$
\kappa(X)\gg1.
$$

这时很小的力噪声就可能导致 IFC 参数出现很大变化。普通 Taylor 特征中的 $u$、$u^3$、$u^5$ 在有限位移分布上容易相关，正是这种病态的常见来源。

Wick 多项式通过减去低阶 contraction，使不同阶 polynomial feature 在目标统计测度下更接近正交。理想 Gaussian 情况下：

$$
\langle :u^m::u^n:\rangle=0,\qquad m\ne n.
$$

于是 Wick Gram 矩阵更接近按阶分块的形式：

$$
G_{\mathrm W}=X_{\mathrm W}^{\mathsf T}X_{\mathrm W}
\approx
\begin{pmatrix}
G_1&0&0&\cdots\\
0&G_2&0&\cdots\\
0&0&G_3&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{pmatrix}.
$$

相比之下，Taylor Gram 矩阵通常包含明显的跨阶 block：

$$
G_{\mathrm T}\approx
\begin{pmatrix}
G_1&G_{12}&G_{13}&\cdots\\
G_{21}&G_2&G_{23}&\cdots\\
G_{31}&G_{32}&G_3&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{pmatrix}.
$$

因此 Wick 的数值优势主要来自 feature decorrelation 或统计正交化，而不是来自某种简单的矩阵三角形式。

## 6. 三角变换为什么有用

一维 Wick 多项式的前几项为：

$$
:u:=u,
$$

$$
:u^2:=u^2-\sigma^2,
$$

$$
:u^3:=u^3-3\sigma^2u,
$$

$$
:u^4:=u^4-6\sigma^2u^2+3\sigma^4.
$$

第 $n$ 阶 Wick 多项式只包含 $n,n-2,n-4,\ldots$ 阶 Taylor 项。因此按阶排列时，变换具有类似下三角的结构：

$$
T=
\begin{pmatrix}
1&0&0&\cdots\\
*&1&0&\cdots\\
*&*&1&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{pmatrix}.
$$

三角结构带来四个工程优势：基底变换可以递归构造、在满足条件时可逆、跨阶关系不会无规则扩散，并且高阶信息如何进入低阶可以逐项追踪。但三角结构本身不保证条件数良好；真正的稳定性仍来自与采样 covariance 相匹配的 decorrelation。

## 7. Wick 是一种统计感知的 feature-space 预条件

Taylor 拟合可以写成：

$$
y=X_{\mathrm T}\theta_{\mathrm T}.
$$

经过 Wick 基底变换后：

$$
y=X_{\mathrm W}\theta_{\mathrm W},
\qquad X_{\mathrm W}=X_{\mathrm T}T(C).
$$

如果 $T(C)$ 使 $X_{\mathrm W}^{\mathsf T}X_{\mathrm W}$ 更接近分块对角，那么 Wick 就相当于在 feature space 中进行了一次与物理统计分布相关的预条件：

$$
\text{Wick}=\text{physics-aware polynomial preconditioner}.
$$

它不改变拟合目标的物理含义，只改变求解坐标系。拟合完成后，通过逆变换可以恢复传统 Taylor IFC，供通用 HDF5、phonopy、ShengBTE 和 ALAMODE writer 使用。

## 8. 跨阶问题被转移到了 covariance 阶段

Wick 变换依赖位移分布的统计量：

$$
C_{ij}=\langle u_i u_j\rangle,
\qquad T=T(C).
$$

如果 covariance 发生变化：

$$
C\rightarrow C',
$$

Wick 基底本身也会变化。因此跨阶混合并没有从数学上消失，而是从 Taylor Gram 矩阵中的隐式列相关，转移成 $T(C)$ 中显式的 covariance dependence。

这种转移是有价值的，因为 covariance 可以直接计算、检查收敛、比较不同数据集、分析温度依赖和比较不同采样策略。换句话说，问题从“参数如何互相污染”变成了“统计测度如何选择以及 contraction 有多大”，后者是可以记录和诊断的。

## 9. Wick 的适用边界

Wick 基底不能弥补物理模型或数据本身的缺陷。以下问题不可能仅靠基底变换解决：

- 训练数据不足或位移方向没有激发全部参数；
- cluster cutoff 太短；
- body-order truncation 过于激进；
- Taylor order 太低；
- DFT 或 MLIP 力噪声过大；
- covariance 估计不准确；
- 训练分布与目标应用分布差别过大。

特别是：

$$
C_{\mathrm{train}}\ne C_{\mathrm{target}}
$$

时，针对训练分布建立的正交性在目标分布下未必成立。因此 Wick 不能把不完备的物理模型变成完备模型，它解决的是同一个模型空间在有限数据下如何更稳定、更可解释地识别。

## 10. 对 MLFCS 的设计意义

Wick 基底在 MLFCS 中的价值可以概括为三点：

1. **可逆**：在固定 covariance 和完整模型空间下，Taylor 与 Wick 可以确定性互转，最终仍然输出传统 Taylor IFC；
2. **稳定**：降低不同 polynomial order feature 的统计相关性，改善 Gram 矩阵和线性求解的条件；
3. **可诊断**：把原本隐藏在参数漂移中的 FC2 与 FC4、FC3 与 FC5 等跨阶耦合显式表示成 covariance contraction。

因此 Wick 不是“让高阶和低阶互不影响”，而是把不可控的互相影响转化为结构化、可测量、可逆的互相影响。

## 总结

Taylor 基底把跨阶竞争隐藏在设计矩阵相关性中；Wick 基底把这部分结构显式放入 covariance contraction：

$$
\text{隐式跨阶竞争}
\longrightarrow
\text{显式跨阶映射}.
$$

Wick 的核心价值不是消除高阶力常数之间的耦合，而是承认耦合始终存在，并通过 covariance-dependent 的正交化基底使它可观测、可诊断和可逆，从而减少病态参数竞争并提高高阶 IFC 拟合的稳定性与可解释性。
