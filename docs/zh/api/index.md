---
title: API 导览
audience:
  - beginner
  - advanced
status: stable
code_verified: 4.0.0a6
---

# API 导览

`mlfcs` 当前提供 12 个顶层公共入口。常用工作流可以只依赖这些名称；结果类型与高级对象从
各自 canonical 子模块导入。完整签名、参数、单位和异常见[参考手册](../reference/index.md)。

| 分组页 | 覆盖的导出 |
| --- | --- |
| [计算入口](calculation.md) | `FiniteDifferenceCalculation`、`perturb_structures` |
| [超胞构建](supercell.md) | `build_supercell` |
| [力常数拟合](fitting.md) | `ForceConstantFitter` |
| [力常数对象](force-constants.md) | `ForceConstants`、`realize_force_constants` |
| [ASE Calculator](../reference/calculator-api.md) | `MLFCSCalculator` |
| [约束与求和规则](constraints.md) | `enforce_rotational_sum_rules` |
| [读写与序列化](io.md) | `write_force_constants`, `read_hdf5` |
| [SCPH](scph.md) | `LoopSCPH` |
| [SSCHA](sscha.md) | `SSCHA` |

## 典型调用顺序

1. 用 `build_supercell` 构造显式超胞。
2. 用 `FiniteDifferenceCalculation` 组织有限差分位移。
3. 将带力的 ASE 结构交给 `ForceConstantFitter` 拟合出 `ForceConstants`。
4. 需要时用 `enforce_rotational_sum_rules` 施加声学求和规则。
5. 用 `write_force_constants` / `read_hdf5` 保存或复用结果。
6. 温度相关性质走 `LoopSCPH`（SCPH）或 `SSCHA` 流程。
7. 需要把 Taylor IFC 作为势能使用时，构造 `MLFCSCalculator` 计算相对能量与原子力。
