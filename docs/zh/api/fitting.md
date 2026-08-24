---
title: 力常数拟合
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# 力常数拟合

本页介绍线性力拟合流程的四个顶层导出。完整签名见
[拟合 API 参考](../reference/fitting-api.md)。

## FitDataset

力/能量数据集容器，把位移模式与对应的受力组织成可拟合形式（待补充构造方式）。

```python
from mlfcs import FitDataset
```

## ForceConstantFitter

最小二乘拟合器，从 `FitDataset` 拟合出指定阶数的力常数（待补充正则化与加权参数）。

```python
from mlfcs import ForceConstantFitter
```

## FittingResult

拟合结果：力常数、残差与统计指标（待补充字段）。

```python
from mlfcs import FittingResult
```

## FittingDiagnostics

拟合质量诊断：条件数、交叉验证等（待补充指标含义）。

```python
from mlfcs import FittingDiagnostics
```
