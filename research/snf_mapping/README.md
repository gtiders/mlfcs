# 完整 SNF 映射研究原型

该目录研究完整 Smith 分解 $D=USV$ 能否替代 MLFCS 当前以 HNF 为中心的
有限超胞平移映射。它不属于 `mlfcs` 运行时，也不会修改正式的 supercell、
IFC、拟合、SSCHA 或 SCPH 路径。

在仓库根目录运行：

```text
uv run python research/snf_mapping/prototype.py
```

原型会验证：

- 完整分解、双侧 unimodular transformation 和 invariant factors；
- SNF 群坐标与 HNF/residue 是否定义同一个 quotient partition；
- SNF direct-space representatives 的 round trip；
- SNF 与现有实现是否产生相同的 commensurate q-point 集合；
- 对角、非对角和高剪切矩阵的大批量 lookup 时间；
- 等价 supercell 行基变化后 SNF transformation 是否保持 canonical。

数学推导和架构结论见 `docs/zh/开发/研究/完整SNF能否统一有限超胞平移映射.md`。
