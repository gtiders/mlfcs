---
title: 读写与序列化
audience:
  - advanced
status: stable
code_verified: 4.0.0a5
---

# 读写与序列化

本页介绍力常数的持久化顶层导出。完整签名见
[I/O API 参考](../reference/io-api.md)。

## write_force_constants

把力常数写入 HDF5 文件（待补充格式版本与元数据字段）。

```python
from mlfcs import write_force_constants
```

## read_hdf5

从 HDF5 文件读取力常数（待补充返回类型与向后兼容说明）。

```python
from mlfcs import read_hdf5
```
