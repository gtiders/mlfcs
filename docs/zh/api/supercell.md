---
title: 超胞构建
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# 超胞构建

本页介绍超胞构造相关的顶层导出。完整签名见
[结构 API 参考](../reference/structures-api.md)。

## build_supercell

从原胞构造 $n_1 \times n_2 \times n_3$ 超胞，并返回周期索引信息（待补充具体签名与返回值）。

### 主要参数（待核对）

- 原胞结构
- 超胞尺寸
- 是否返回映射关系

### 最小示例（占位）

```python
from mlfcs import build_supercell
```

## PeriodicIndex

超胞的周期索引对象，用于在原胞索引与超胞原子/格点之间换算（待补充字段与方法）。

```python
from mlfcs import PeriodicIndex
```
