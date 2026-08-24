---
title: 版本与兼容范围
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 版本与兼容范围

当前版本可由 `mlfcs.__version__` 查询。项目仍处于 alpha 阶段，公共接口会在版本说明中明确记录破坏性变化。

稳定边界包括顶层 `mlfcs.__all__`、本 Reference 明确列出的高级结果对象、HDF5 v3 schema 和公开格式
语义。`mlfcs.interactions`、JAX design packing、约束内部矩阵与以下划线开头的名称属于内部实现。

当前读取器只接受 HDF5 v3，不保留 v1/v2 兼容分支。文件应记录创建时的 MLFCS 版本；若未来 schema
改变，将通过显式迁移工具或明确拒绝处理，而不是静默猜测旧字段。

建议每次计算保存：版本、primitive/reference、每阶 order/cutoff/body order、`symprec`、拟合基、
ASR、batch size、随机种子、外部计算器版本、完整日志和关键 JSON 指标。

文档 front matter 的 `code_verified` 表示页面签名最后核对的源码版本，不表示未来版本自动兼容。
