---
title: SSCHA 方法
audience:
  - advanced
status: experimental
code_verified: 4.0.0a5
---

# SSCHA 方法

本页介绍随机自洽谐波近似（SSCHA）的顶层导出。完整签名见
[SSCHA API 参考](../reference/sscha-api.md)；方法背景见
[原生 SSCHA 模块理论](../theory/sscha.md)。

## SSCHA

SSCHA 自洽循环入口（待补充主要参数：温度、采样数、收敛准则）。

```python
from mlfcs import SSCHA
```

## SSCHAIteration

单次迭代的记录：有效谐波模型与采样统计（待补充字段）。

```python
from mlfcs.physics.sscha import SSCHAIteration
```

## SSCHAResult

自洽收敛后的最终结果（待补充字段）。

```python
from mlfcs.physics.sscha import SSCHAResult
```

## perturb_structures

按谐波系综对结构施加随机扰动，生成用于外层计算的构型（待补充参数）。

```python
from mlfcs import perturb_structures
```
