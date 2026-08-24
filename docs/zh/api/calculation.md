---
title: 计算入口
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# 计算入口

本页介绍组织有限差分计算的三个顶层导出。完整签名见
[力常数 API 参考](../reference/force-constants-api.md)。

## FiniteDifferenceCalculation

有限差分力常数计算的高层入口，负责生成位移模式、收集力数据并拟合 FC2/FC3 等。

### 主要参数（待核对）

- 结构与超胞设置
- 位移幅度与位移模式选择
- 拟合阶数

### 最小示例（占位）

```python
from mlfcs import FiniteDifferenceCalculation
```

## CentralDifferenceStencil

中心差分格式描述（待补充：位移方向、步长与前向/中心差分的关系）。

```python
from mlfcs.finite_difference.stencil import CentralDifferenceStencil
```
