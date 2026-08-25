---
title: Exact-R 与 Periodic FC2 Completion
audience:
  - advanced
  - developer
status: experimental
code_verified: 4.0.0a6
---

# Exact-$R$ 与 Periodic FC2 Completion

该实验功能在 canonical exact-$R$ FC2 旁增加一个只属于当前 source supercell 的 periodic
harmonic Hessian。它先定义

$$
E_C=\frac12\mathbf u^T\Phi_C\mathbf u,
\qquad
\mathbf F_C=-\Phi_C\mathbf u,
$$

再生成 design columns，因此不是 arbitrary force residual。

完整 finite FC2 space 同时满足 Hessian symmetry、reference-compatible space-group symmetry
和 ASR。exact FC2 映射到同一空间后，completion 取其正交补：

$$
\mathcal H_{\rm SC}^{\rm ASR}=\mathcal H_E\oplus\mathcal H_C.
$$

```python
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2,),
    cutoffs={2: 4.5},
    periodic_fc2_completion=True,
)
```

该功能默认关闭。completion 只能随相同 source translation sublattice 的原子重排，不能导出
到不同大小超胞。Si 三帧测试显示新增自由度可能过拟合；NaCl 两帧官方 DFT 数据中，hybrid
相对 phonopy FC2 的差异从 0.0405 降至 0.00882，且不开 NAC 的声子频率 RMS 差异从
0.1273 THz 降至 0.000328 THz。

完整推导、symfc 对照、内存分析和复现脚本位于
`research/periodic_fc2_completion/`。
