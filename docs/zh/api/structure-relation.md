---
title: 结构与对齐
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# 结构与对齐

本页介绍结构关系与对齐相关的顶层导出。完整签名见
[结构 API 参考](../reference/structures-api.md)。

## StructureRelation

描述两个结构之间的对应关系（原子映射、晶胞变换），是把力常数从一个超胞
转换到另一个超胞的基础（待补充字段与方法）。

```python
from mlfcs import StructureRelation
```

## align_structures

把一个结构旋转/平移对齐到另一个结构，返回对齐信息（待补充参数与返回值）。

### 主要参数（待核对）

- 源结构与目标结构
- 是否容差匹配

### 最小示例（占位）

```python
from mlfcs import align_structures
```
