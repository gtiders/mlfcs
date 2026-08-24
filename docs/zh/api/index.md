---
title: API 导览
audience:
  - beginner
  - advanced
status: stable
code_verified: 4.0.0a4
---

# API 导览

`mlfcs` 的全部顶层导出共 29 个符号，按功能分组如下。每组对应 `api/`
下的一个页面，介绍各导出的用途、主要参数与最小用法示例；更完整的
签名细节见[参考手册](../reference/index.md)。

| 分组页 | 覆盖的导出 |
| --- | --- |
| [计算入口](calculation.md) | `ForceConstantCalculation`, `Calculation`, `CentralDifferenceStencil` |
| [超胞构建](supercell.md) | `build_supercell`, `PeriodicIndex` |
| [结构与对齐](structure-relation.md) | `StructureRelation`, `align_structures` |
| [力常数拟合](fitting.md) | `FitDataset`, `ForceConstantFitter`, `FittingResult`, `FittingDiagnostics` |
| [力常数对象](force-constants.md) | `ForceConstants`, `SparseOrderForceConstants`, `realize_force_constants` |
| [约束与求和规则](constraints.md) | `enforce_rotational_sum_rules`, `RotationalSumRuleResult`, `RotationalSumRuleDiagnostics` |
| [读写与序列化](io.md) | `write_force_constants`, `read_hdf5` |
| [SCPH 无序计算](scph.md) | `LoopSCPH`, `LoopSCPHResult`, `harmonic_frequencies`, `TemperatureSeriesResult` |
| [SSCHA 方法](sscha.md) | `SSCHA`, `SSCHAIteration`, `SSCHAResult`, `HarmonicEnsemble`, `EnsembleDiagnostics`, `perturb_structures` |

## 典型调用顺序

1. 用 `build_supercell` 构造超胞，得到 `PeriodicIndex`。
2. 用 `ForceConstantCalculation` / `Calculation` 组织有限差分位移。
3. 用 `FitDataset` 收集力数据，交给 `ForceConstantFitter` 拟合出
   `ForceConstants`。
4. 需要时用 `enforce_rotational_sum_rules` 施加声学求和规则。
5. 用 `write_force_constants` / `read_hdf5` 保存或复用结果。
6. 温度相关性质走 `LoopSCPH`（SCPH）或 `SSCHA` 流程。
