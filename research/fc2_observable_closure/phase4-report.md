# FC2 Observable Closure 第四阶段：Minimal Architecture Prototype

## 1. 结论

第四阶段只在 `research/fc2_observable_closure/` 中实现架构 prototype，没有修改
`src/mlfcs`。结论升级为：

$$
\boxed{\text{Architectural GO}}
$$

Production GO 仍明确为 false。该结论表示 source-specific finite harmonic residual 可以在
不污染 canonical exact-$R$ IFC 的条件下，以局部改动接入当前 fitting architecture；它
不表示功能已经进入正式 API。

## 2. `FiniteObservableSpace`

prototype 中的 `FiniteObservableSpace` 拥有：

- source `StructureRelation` 与 atom-order-independent fingerprint；
- ASR-constrained finite observable basis；
- transferable realization map；
- closure projector factor；
- 临时 SVD closure basis；
- 构造与内存诊断。

稳定语义是 source identity、observable definition、closure projector 和重建后的 Hessian。
SVD basis 与 coefficient $\eta$ 只是拟合 cache，不作为 physical parameter 序列化。

## 3. 不构造 dense projector 的 observable builder

新 builder 在逻辑 pair label

$$
(a,b,[R])
$$

上执行有限群作用：

1. 使用 reference-compatible space-group operations 和 Hessian transpose 生成有限 pair
   orbits；
2. 对每个 pair orbit 选择一个 representative；
3. 仅在 representative 的 9 维 Cartesian tensor space 中累计 stabilizer constraint
   Gram；
4. 使用现有小 Gram null-space kernel 得到 invariant tensor basis；
5. 将 basis 沿 orbit actions 稀疏传播，并按 orbit size 归一化；
6. 在已生成的 symmetry-allowed coordinates 中施加 ASR；
7. realization transferable map 后构造 closure complement。

它不形成 raw compact projector，也不构造 $O(D_{\rm raw}^2)$ 矩阵。stabilizer constraints
只累计一个 $9\times9$ Gram；早期 prototype 曾错误堆叠 constraints 并执行 full SVD，
$4^3$ 临时内存达到约 822 MiB，现已删除该路径。

### 正确性

| reference | 旧 dense dim | pair-orbit dim | 最大 principal angle | projector 相对差异 |
|---|---:|---:|---:|---:|
| $2^3$ | 13 | 13 | $8.35\times10^{-16}$ | $8.84\times10^{-16}$ |
| $3^3$ | 24 | 24 | $1.22\times10^{-15}$ | $8.82\times10^{-16}$ |

### 性能与内存

| reference | pair orbits | observable dim | basis nnz | builder | total observable+closure | 自身临时峰值 |
|---|---:|---:|---:|---:|---:|---:|
| $2^3$ | 8 | 13 | 120 | 0.22 s | 0.24 s | 0.11 MiB |
| $3^3$ | 12 | 24 | 612 | 0.79 s | 0.82 s | 0.31 MiB |
| $4^3$ | 22 | 52 | 2376 | 1.92 s | 1.99 s | 0.79 MiB |

进程 peak RSS 在三个构造之后约从 263 MiB 增至 268 MiB；这包含 Python、JAX、phonopy
和 PolyMLP 已加载运行时，不能解释为 builder 独占内存。`tracemalloc` 隔离的 builder
临时峰值更能反映增量。

$4^3$ 具有 128 个原子和 2304 维 raw compact space，但新算法可以直接构造 52 维
observable basis，不再受到第三阶段 dense projector 的限制。

## 4. Design-block protocol

prototype 定义最薄的内部协议：

```text
DesignBlock
    n_parameters
    build_batch(displacements) -> design columns
```

两个实现为：

- `OrbitDesignBlock`：包装现有 `force_design_batch()`、`OrderParameterization` 和 Wick 路径；
- `FiniteHarmonicDesignBlock`：只执行
  $$
  X_{s,ia,p}=-\sum_{jb}\Phi^{(p)}_{ia,jb}u_{s,jb}.
  $$

closure block 没有复制 Wick、batch runtime、Gram、column scaling、preconditioner 或 solver。
prototype 将它包装成一个额外的 `DesignKernelGroup`，与现有 orbit group 在同一 batch 中
拼接 physical columns，再通过 block-diagonal parameter map 同时施加 transferable ASR：

$$
R_{\rm joint}=\operatorname{diag}(R_{\rm ASR},I_C).
$$

所有 columns 一次进入现有 `_StreamingGramSystem` 和现有 solver。不存在先拟合
transferable、再拟合 residual 的顺序依赖。

## 5. End-to-end KCl

KCl $2^3$、100 个 $0.01$ Å COM-removed snapshots 的结果：

| 指标 | 数值 |
|---|---:|
| joint rank | 11/11 |
| transferable reduced parameters | 2 |
| closure fitting coordinates | 9 |
| streamed solver stop code | 0 |
| streamed/direct parameter relative difference | $8.96\times10^{-13}$ |
| Gram relative difference | $4.30\times10^{-16}$ |
| rhs relative difference | $5.48\times10^{-16}$ |
| force RMSE | $6.4602774\times10^{-4}$ eV/Å |
| total Hessian ASR maximum | $1.87\times10^{-15}$ |
| total Hessian vs Phase 3 dense result | $4.26\times10^{-13}$ |
| closure source view round trip | 0 |
| streamed Gram | 0.28 s |
| 100-frame closure design | $3.6\times10^{-4}$ s |

差异由现有 iterative solver tolerance 决定；将 tolerance 收紧到 $10^{-12}$ 后，Hessian
和 parameter 均达到 FP64 数值等价范围。

## 6. `FiniteHarmonicResponse` source ownership

prototype 对象只拥有：

- source relation 和 translation-sublattice fingerprint；
- compact finite Hessian；
- optional full source view；
- symmetry、ASR 和 source-only metadata。

它不拥有 exact-$R$ rows、orbit 或 transferable parameter identity。验证结果为：

- 相同 source 的任意 atom reorder 可以通过 `(primitive site, translation residue)` 精确
  重映射，误差为 0；
- 同一 translation sublattice 的整数幺模 supercell basis representation 可以验证并往返，
  norm 与逐元素 round-trip 误差均为 0；
- $3^3$ 等不同 source supercell 被明确拒绝；
- 不调用 nearest-image，也不选择 arbitrary exact-$R$ lift。

source 上总谐响应只有一个显式组合：

$$
\Phi_{\rm total}^{\rm source}
=
\operatorname{realize}(\Phi_{\rm transferable},\mathrm{source})
+
\Phi_{\rm finite\ residual}^{\rm source}.
$$

closure 不会静默写入 `SparseOrderForceConstants`。

## 7. 退化与拒绝路径

- closure dimension = 0：不创建 `FiniteHarmonicDesignBlock`，参数数为 0；
- transferable alias：原有 `InteractionAliasingError` 仍先于 closure；
- joint dataset rank deficient：在 solve 前拒绝，不使用 regularization；
- source fingerprint 或 translation sublattice 不匹配：拒绝应用 response。

## 8. 正式实现的最小文件边界

若下一阶段获批，正式源码只需局部涉及：

1. 新增 `force_constants/finite_harmonic.py`：source-owned response 与 finite observable
   builder；
2. `fitting/design.py`：内部 design-block protocol 和 finite harmonic block；
3. `fitting/gram.py`：从“只遍历 orbit groups”泛化为遍历 design blocks，Gram 数值核不变；
4. `fitting/fitter.py`：显式 opt-in 构造与 result companion field；
5. 新增 builder、joint Gram、ownership 和拒绝路径测试。

以下保持完全不变：

- `PrimitiveInteractionSpace`、`InteractionSpace`、`OrderParameterization`；
- exact-$R$ expansion 与 `SparseOrderForceConstants`；
- canonical `ForceConstants` 与 HDF5 schema；
- `realize_force_constants()`；
- FC3/FC4、Wick、solver、external writers。

## 9. 最终判定

第四阶段的全部验收条件成立：无 dense full projector、joint design/Gram、无数值引擎复制、
canonical IFC 不变、source ownership 严格、退化与 alias 路径明确、$4^3$ 构造可接受，且
KCl end-to-end 与原始研究结果 FP64 等价。

因此结论升级为 **Architectural GO**。这只批准未来制定正式实现计划，不是
Production GO，也没有改变当前公共 API。
