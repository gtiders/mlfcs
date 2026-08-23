---
title: 约束与求和规则
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# 约束与求和规则

本页介绍声学求和规则相关的顶层导出。完整签名见
[约束 API 参考](../reference/constraints-api.md)。

## enforce_rotational_sum_rules

对力常数施加旋转不变性 / 声学求和规则修正（待补充支持的规则种类与迭代参数）。

### 主要参数（待核对）

- 目标力常数
- 规则选择（Huang 等）
- 迭代次数与收敛容差

## RotationalSumRuleResult

施加约束后的结果容器：修正后的力常数与误差下降情况（待补充字段）。

```python
from mlfcs import RotationalSumRuleResult
```

## RotationalSumRuleDiagnostics

约束前后求和规则违反度的诊断信息（待补充指标含义）。

```python
from mlfcs import RotationalSumRuleDiagnostics
```
