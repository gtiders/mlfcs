---
title: 力常数对象
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# 力常数对象

本页介绍承载力常数数据的三个顶层导出。完整签名见
[力常数 API 参考](../reference/force-constants-api.md)。

## ForceConstants

核心力常数容器，支持 FC2 及更高阶（待补充索引约定与常用方法）。

```python
from mlfcs import ForceConstants
```

## SparseOrderForceConstants

单阶稀疏存储的力常数，适合高阶与大超胞（待补充构造方式）。

```python
from mlfcs import SparseOrderForceConstants
```

## realize_force_constants

把稀疏/压缩表示展开成完整数组形式（待补充参数与返回类型）。

```python
from mlfcs import realize_force_constants
```
