# FC2 transferable map 与有限超胞 observable closure

本目录研究当前 primitive/orbit FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

在 KCl 的 $2\times2\times2$ reference 上是否存在可辨识且严格互补的 finite-supercell
closure。`prototype.py` 是独立研究原型，不修改或绕过正式实现。

运行：

```bash
uv run --with phonopy --with pypolymlp python research/fc2_observable_closure/prototype.py
```

第一阶段在未施加 ASR 的 13 维 observable space 中得到 12/13 的去质心数据秩，因此
结论为 No-Go。第二阶段把 ASR 直接定义进表示空间：

$$
\mathcal H_{\mathrm{SC}}^{\mathrm{ASR}}=\ker C_{\mathrm{ASR}}.
$$

结果为：observable space 从 13 维降到 11 维；生产 transferable 参数从 4 维降到
2 维并保持零 kernel；重新构造的 closure 为 9 维；联合表示和 100 帧去质心数据的
design 均为 11/11 满列秩，condition number 为 4.02。第一阶段唯一的 null direction
投影到 ASR 子空间的相对范数仅为 $1.31\times10^{-15}$，证明它属于 ASR 禁止的均匀
平移响应。

因此第二阶段的最小可行性结论为 **GO**：ASR-constrained observable closure 在该 KCl
案例中数学上闭合、数值上可分离且能由给定数据辨识。这只是允许进入后续架构讨论，
不是正式功能接入。

这里的 closure 始终只表示：

> finite-supercell harmonic response not represented by the current transferable FC2 basis

它不是 long-range FC2，也不对应唯一的无限晶格 interaction。完整矩阵、null vector、
奇异值和拟合诊断保存在 `results.json`。

第三阶段进一步研究维数结构、metric、reference/cutoff dependence、真实 PolyMLP Hessian、
数据稳健性和正式架构边界：

```bash
uv run --with phonopy --with pypolymlp python research/fc2_observable_closure/phase3.py
```

完整结论见 `phase3-report.md`，机器结果见 `results-phase3.json`。第三阶段确认
**Mathematical GO + Prototype Recommended**，但没有给出 Production GO，也没有把 closure
提升为 transferable exact-$R$ IFC。

## 第四阶段：最小架构原型

运行：

```bash
uv run --with phonopy --with pypolymlp python research/fc2_observable_closure/architecture_prototype.py
```

第四阶段用有限 pair-label 群轨道和 $3\times3$ tensor stabilizer invariant 直接构造
`FiniteObservableSpace`，不再构造 dense full projector。$2^3$、$3^3$、$4^3$ KCl
reference 的 observable dimension 分别为 13、24、52；$4^3$ 构造约需 1.9 秒，原型新增
Python 内存峰值约 0.8 MiB。

原型以统一 `DesignBlock` 接口把 transferable orbit columns 和 source-only finite harmonic
columns 同时送入现有 streamed Gram 与 solver。KCl $2^3$ 的联合 design 为 11/11 满列秩，
Gram、RHS 与显式参考的相对误差分别为 $4.30\times10^{-16}$ 和
$5.48\times10^{-16}$，总 Hessian 与第三阶段 dense 参考的相对差异为
$4.26\times10^{-13}$。

`FiniteHarmonicResponse` 严格绑定 source primitive 与 translation sublattice：允许 reference
原子重排和同一子晶格的整数幺模换基，拒绝不同 source supercell；它不会写入 canonical
exact-$R$ IFC、HDF5 或外部格式。

第四阶段结论为 **Architectural GO**，但 `production_go=false`。完整说明见
`phase4-report.md`，机器结果见 `results-phase4.json`。本目录仍是独立研究原型，没有修改
`src/mlfcs` 或公共 API。
