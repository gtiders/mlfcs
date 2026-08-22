---
title: 格式
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 格式

原生 HDF5 是规范交换格式，保存 primitive/reference 结构、一般超胞矩阵、原子映射、稀疏 IFC、单位、
约束和生成 metadata。

导出是经过验证的 view 操作。允许原子重排、周期原点平移和整数幺模换基，但必须保持 primitive 和
supercell 的平移晶格；writer 不会放大、缩小、应变或重新定义原胞。

| 使用者 | 格式 |
|---|---|
| MLFCS 和高阶工作流 | [原生 HDF5](native-hdf5.md) |
| phonopy | [FC2 文本](phonopy-text.md) |
| phono3py | [phonopy 与 phono3py HDF5](phonopy-hdf5.md) |
| ShengBTE | [FC3/FC4 文本](shengbte.md) |
| ALAMODE | [FCSXML](alamode.md) |
