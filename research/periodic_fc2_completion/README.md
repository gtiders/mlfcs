# Exact-$R$ 与 Periodic FC2 Completion 实现报告

## 结论

MLFCS 现在提供默认关闭的 `periodic_fc2_completion=True` 拟合模式。它保留 canonical
exact-$R$ IFC，并在当前唯一训练超胞上增加一个 source-bound periodic FC2 sidecar。补空间
首先是满足 Hessian 对称、空间群对称和平移声学求和规则的二阶势能空间：

$$
E_C(\mathbf u)=\frac12\mathbf u^T\Phi_C\mathbf u,
\qquad
\mathbf F_C=-\Phi_C\mathbf u.
$$

它不是由 force residual 定义的补偿器。

## 当前计算链与插入位置

```text
exact-R interaction/orbit → exact design ───────────────┐
                                                       ├→ streamed Gram → solver
finite pair group orbit → ASR basis → exact-span complement → periodic FC2 design ┘
```

FC3、FC4、Taylor/Wick backend、exact-$R$ expansion 和 alias rejection 都没有改变。只有显式
开启 completion 时，fitter 才构造 `SupercellHessianSpace` 并追加 FC2 columns。

## `SupercellHessianSpace`

有限空间先用 pair label $(a,b,[R])$ 表示 translation-reduced FC2 block。生成元包含：

1. 与 reference 相容的有限空间群作用；
2. Hessian transpose $(a,b,[R],T)\sim(b,a,[-R],T^T)$；
3. compact FC2 上的 ASR null space。

每个 pair orbit 只在 $3\times3$ tensor 空间累计 stabilizer Gram，随后把 invariant basis 沿
orbit 传播；因此不构造 raw dense projector。对 NaCl 512 原子 source，完整对称 Hessian有
1,180,416 个上三角自由度，translation-reduced raw space 为 9,216 维，空间群约化后 168
维，ASR 后 166 维。

## exact 映射与补空间

exact FC2 的 ASR 参数映射到同一 compact finite Hessian 坐标：

$$
M=B_{\rm SC}^TB_E R_{\rm ASR}.
$$

若 $\ker M\ne\{0\}$，继续抛出 `InteractionAliasingError`。只有 $M$ 满列秩时才用一次完整
SVD 构造左正交补：

$$
B_C=B_{\rm SC}U_{[:,r:]}.
$$

因此 exact columns 优先保留，且

$$
\operatorname{im}B_E\cap\operatorname{im}B_C=\{0\},
\qquad
\operatorname{span}[B_E,B_C]=\mathcal H_{\rm SC}^{\rm ASR}.
$$

这就是不 double count 的原因。completion 不需要 exact $R$，因为它的物理对象是某一有限
translation quotient 上的 Hessian，而不是无限晶格 interaction 的某个 lift。

## Design、Gram 与内存

completion design 是严格的 $-\Phi u$ contraction。实现保存 compact periodic basis
$(P,n_p,N,3,3)$ 和 cell-translation lookup，不展开 $(P,N,N,3,3)$ full basis。NaCl 中该
改动把原本约 1.7 GiB 的潜在 full-basis 临时量降为 6.47 MiB completion basis，加上
11.67 MiB ASR observable basis。

显式 design 继续进入现有 streamed Gram。小型 NaCl $2^3$、100 帧基准中，moment/direct
Gram 与显式 design 的相对误差分别为 $2.89\times10^{-16}$ 和
$1.68\times10^{-15}$；单次计时为 0.00060 s 对 0.00203 s，且 moment 为 18 KiB、显式
completion design 为 338 KiB。当前没有接入 Direct-Gram，因为联合 exact/completion cross
block、GPU/CPU 双路径和现有 streamed cache 会增加第二套 Gram 实现；当前 periodic kernel
在真实 NaCl 的两帧上只增加约 0.22 s Gram 时间。该方向保留为后续 benchmark candidate。

## symfc 对照

symfc 1.7.3 的 FC2 路径按 lattice-translation compression、permutation projector、space-group
coset projector 和 sum-rule projector 构造 basis。MLFCS 的 source mapping 与 exact span 已经
存在，因此没有把 symfc 的原子排序和 compression object 引入运行时；但用其公共 FC2 basis
进行了独立验证：

| case | MLFCS symmetry/ASR | symfc compressed/ASR | 最大子空间夹角 |
|---|---:|---:|---:|
| NaCl primitive $2^3$ | 13/11 | 13/11 | $1.48\times10^{-15}$ rad |
| Si 128 atoms | 45/44 | 45/44 | 维数一致 |
| NaCl 512 atoms | 168/166 | 168/166 | 维数一致 |

这表明 completion space 与成熟 periodic FC2 algebra 一致，同时避免新增运行时依赖和第二套
structure mapping。

## 验证结果

合成 NaCl $2^3$ 测试得到 exact 2 维、completion 9 维、hybrid 11 维。随机
$\Phi\in\mathcal H_{\rm SC}^{\rm ASR}$ 的 force fit RMSE 在数值舍入下为零，FC2 相对恢复误差
约 $6.4\times10^{-12}$；纯 exact target 的 completion coefficient 小于 $10^{-12}$。

| case | exact RMSE | hybrid RMSE | exact→phonopy FC2 | hybrid→phonopy FC2 | exact/hybrid phonon RMS |
|---|---:|---:|---:|---:|---:|
| Si，128 atoms，3 Gaussian frames | $4.996\times10^{-3}$ | $4.820\times10^{-3}$ | 0.00574 | 0.01073 | 0.0404 / 0.0937 THz |
| NaCl，512 atoms，2 official DFT frames | $2.652\times10^{-5}$ | $1.315\times10^{-5}$ | 0.04054 | 0.00882 | 0.1273 / 0.000328 THz |

Si 的训练误差略降但对独立 finite-displacement FC2 更差，说明三帧数据不足以稳定估计新增 17
维，completion 不是无条件改善结果的正则化。NaCl 则将 phonopy FC2 差异降低约 4.6 倍，符合
它补齐 finite-supercell harmonic response 的设计目标。本比较不开 NAC。

## Reconstruction、IO 与边界

- `ForceConstants.sparse[2]` 始终只保存 transferable exact-$R$ FC2。
- `ForceConstants.periodic_fc2_completion` 保存 source-bound compact Hessian。
- `materialize(2)`、phonopy text/HDF5 与 `MLFCSCalculator` 使用两者之和。
- native HDF5 v3 可选保存 source reference、rank report 和 periodic compact Hessian。
- 相同 translation sublattice 的 atom reorder 可转换；不同大小 target 明确拒绝。
- ALAMODE 的 FC2 sparse export 和 rotational postprocessor当前拒绝带 completion 的结果。
- completion 只用于 FC2；FC3/FC4 的 finite periodic tensor 空间按 $N^3/N^4$ 增长，且失去当前
  exact interaction 的可迁移物理语义，不在本机制范围内。

未来 Gonze/Ewald 项与本机制不冲突。解析长程项可以先从 total force 中扣除并作为独立已知
$\Phi_{\rm LR}$ 恢复；periodic completion 只拟合剩余的、当前 exact FC2 span 未覆盖的 source
harmonic directions。

## 复现

```bash
uv run python research/periodic_fc2_completion/benchmark.py
uv run pytest -q -s tests/test_periodic_fc2_completion.py
```

数值摘要保存在 `results.json`。
