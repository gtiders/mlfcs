---
title: 工作流
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 工作流

所有工作流共享结构关系、稀疏 IFC 模型、约束和 writer。先决定力的来源，再单独决定导出格式。

| 力来源 | 工作流 |
|---|---|
| ASE calculator | [有限差分](finite-difference-workflow.md) |
| VASP/QE/其他外部程序 | [外部计算器](external-calculator.md) |
| 已有结构快照和力 | [仅力数据拟合](first-fc2-fitting.md) |
| 谐波采样 | [SSCHA](sscha-workflow.md) |
| FC2 + FC4 loop 修正 | [SCPH](scph-workflow.md) |
