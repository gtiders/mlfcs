# 工作流

所有工作流共享结构关系、稀疏 IFC 模型、约束和 writer。先决定力的来源，再单独决定导出格式。

| 力来源 | 工作流 |
|---|---|
| ASE calculator | [有限差分](finite-difference.md) |
| VASP/QE/其他外部程序 | [外部计算器](external-calculators.md) |
| 已有结构快照和力 | [仅力数据拟合](fitting.md) |
| 谐波采样 | [SSCHA](sscha.md) |
| FC2 + FC4 loop 修正 | [SCPH](scph.md) |
