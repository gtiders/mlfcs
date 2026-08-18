# 周期几何与 IFC

共享的 `StructureRelation` 和 `PeriodicIndex` 使用 primitive site 与平移 residue 建立映射，不依赖对角
`repeats` 或 cell-major 原子顺序。`PeriodicGeometry` 使用一般 MIC、Minkowski 约化搜索和全部等距最近像。

原生稀疏模型保存 lattice-labelled 条目：

```text
sites                        (K, order)
translation_representatives  (K, order - 1, 3)
tensors                      (K, 3, ..., 3)
```

residue 是有限超胞中的物理标签。任意 q 点 Fourier 插值时，SCPH 会解析全部简并 Wigner-Seitz 镜像并
等权处理相位。writer 不会自行猜测原子顺序或周期几何。
