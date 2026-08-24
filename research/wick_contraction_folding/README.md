# Wick 跨阶收缩中的有限超胞 folding

## 状态

```text
状态：Known limitation / accepted assumption
当前行为：保持不变
正式实现：不增加 residual guard，不保存 source-only contraction residual
```

MLFCS 当前假设：训练 reference 足以使 Wick→Taylor 跨阶收缩得到的低阶响应可以由现有
transferable exact-$R$ IFC 表示。即使有限 reference 中存在 translation folding，本阶段也将其
数值误差视为可接受，不改变 fitter、HDF5、validation 或 writer 行为。

该决定不是数学上证明问题不存在，而是明确接受以下近似：

$$
K_C\operatorname{im}(M_n)
\subseteq
\operatorname{im}(M_{n-2p}).
$$

其中 $M_n$ 是 $n$ 阶 primitive exact-$R$ 参数到 source supercell observable 的 realization，
$K_C$ 是由有限超胞 covariance 定义的 Wick contraction。

## 一般形式

这个问题不只存在于 FC4→FC2。任意同奇偶性的 Wick→Taylor 转换都包含 contraction：

$$
\mathrm{FC}n
\rightarrow
\mathrm{FC}(n-2),
\mathrm{FC}(n-4),
\ldots.
$$

例如：

- FC3→FC1；
- FC4→FC2、FC4→FC0；
- FC5→FC3、FC5→FC1；
- FC6→FC4、FC6→FC2、FC6→FC0。

对 $p$ 次 covariance contraction，一般项为

$$
\Delta\Phi^{(n-2p)}
\propto
\Phi^{(n)}:\underbrace{C:\cdots:C}_{p\text{ 次}}.
$$

若 contraction 使用的是有限 reference covariance，则 covariance 严格拥有的是 quotient label
$[R]$，而不一定拥有唯一的 infinite-lattice exact translation $R$。不同 exact translations 在
source quotient 中 folding 时，低阶结果可能包含

$$
\Phi_{\mathrm{contracted}}^{(n-2p),\mathrm{source}}
=
\Phi_{\mathrm{transferable}}^{(n-2p)}
+
\Phi_{\mathrm{finite\ residual}}^{(n-2p),\mathrm{source}}.
$$

当前转换只保留第一种表示所能承载的结果，不另行建模第二项。

## 已验证反例与安全对照

在一个 $4\times1\times1$ Ar reference、4.1 Å cutoff 的 FC2+FC3+FC4 原型中，$y,z$
方向只有一个 primitive cell，$R=0$ 与 $R=\pm1$ folding。FC4→FC2 后 Wick 与 Taylor force
的相对差异约为

$$
2.73\times10^{-3}.
$$

使用 $3\times3\times3$ reference，使 cutoff 支撑内相关 translations 可区分后，相对差异为

$$
6.30\times10^{-15}.
$$

显式枚举 FC4 的六种轴对 contraction 与当前 intertwiner map 一致，说明该反例不是 factorial、
固定轴或组合系数错误。

## 与现有检查的关系

现有检查分别回答：

- cutoff：哪些 exact-$R$ interactions 属于模型支撑；
- realization rank：原始 $n$ 阶参数在 source reference 中是否可辨识；
- missing exact contraction：收缩是否产生配置支撑外的 target key。

它们不检查：

$$
K_C\operatorname{im}(M_n)
\subseteq
\operatorname{im}(M_{n-2p}).
$$

因此原始高阶参数满秩，并不自动证明 contraction 后的低阶 observable 完全 transferable。

## 当前锁定处理

当前版本采取以下处理：

1. 不增加跨阶 contraction span/rank residual guard；
2. 不因为该问题拒绝 fitting 或 HDF5 写出；
3. 不引入 source-only `FiniteHarmonicResponse` 正式功能；
4. 不改变现有 Wick→Taylor intertwiner；
5. 将误差视为 reference/cutoff 选择下可接受的建模误差；
6. 文档和测试不得宣称有限胞 folding 情况下一定达到 FP64 严格等价。

## 对 FC1 与 ASE Calculator 的影响

FC1 数据表示和 ASE Calculator 设计与本问题绑定，当前一并暂缓实施。

原因不是 FC1 本身无法表示。primitive 周期性的 Taylor FC1 可以写成

$$
\Phi^{(1)}_{a\alpha},
$$

并按 primitive site 映射重复到任意合法 target supercell。真正尚未锁定的是它的来源语义：
FC3、FC5 等奇数阶 Wick 项产生 FC1 时，同样经过有限 covariance contraction，因而可能包含
transferable primitive FC1 无法完整代表的 source-only finite response。

同理，若 ASE Calculator 直接读取当前 Wick 拟合后保存的 Taylor IFC，它在一般 folding 情况下
不保证逐点重现训练期 Wick predictor。项目既然暂时接受该差异，就不应同时发布一个暗示严格
等价的公共 Calculator。

因此当前决定为：

1. 不向 HDF5 增加 FC1；
2. 不修改 `ForceConstants` 数据结构以支持 `order=1`；
3. 不发布 `ForceConstantPotential` 或 `MLFCSCalculator`；
4. 保留 `research/ase_calculator/` 中的数学设计和独立原型；
5. 若未来重新启动，必须先决定 Calculator 表示的是 canonical transferable Taylor potential，
   还是要求严格复现 source-reference Wick predictor。

若未来重新开启该问题，候选方案是：构造实际 finite contraction response，投影到目标
transferable span，并将 span 外 residual 作为诊断或 source-only companion；在此之前不改变
生产路径。

相关原型和数值结果位于：

- `research/ase_calculator/prototype.py`；
- `research/ase_calculator/results.json`；
- `research/fc2_observable_closure/`。
