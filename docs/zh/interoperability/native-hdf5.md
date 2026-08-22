---
title: 原生 HDF5 v3
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 原生 HDF5 v3

原生 HDF5 v3 保存 primitive 结构以及 exact 实空间 IFC：primitive site、primitive 晶格整数平移和
笛卡尔张量。文件不保存 source supercell 映射；读取后可将同一组 IFC 展开到任意经过验证的整数超胞。

旧 schema 会明确报告不支持；没有会猜测旧原子语义的迁移 reader。
