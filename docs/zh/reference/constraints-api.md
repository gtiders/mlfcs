---
title: 约束 API
audience:
  - developer
status: stable
code_verified: 4.0.0a5
---

# 约束 API

记录平移约束构造和公共 FC2 旋转修正操作。

~~~python
enforce_rotational_sum_rules(
    force_constants: ForceConstants,
    *,
    born_huang: bool = False,
    huang: bool = False,
    strength: float = 1.0,
    tolerance: float = 1e-8,
) -> RotationalSumRuleResult
~~~

该操作只修正 FC2，并始终保持平移 ASR。`strength=1.0` 是严格默认值；$[0,1]$ 内的值缩放 Born–Huang/Huang 修正，诊断对象报告实际残差。
