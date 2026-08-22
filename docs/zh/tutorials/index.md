---
title: 工作流
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 工作流

## 目标

所有工作流共享结构关系、稀疏 IFC 模型、约束和 writer。先决定力的来源，再单独决定导出格式。 | 力来源 | 工作流 | |---|---| | ASE calculator | [有限差分](finite-difference-workflow.md) | | VASP/QE/其他外部程序 | [外部计算器](external-calculator.md) | | 已有结构快照和力 | [仅力数据拟合](first-fc2-fitting.md) | | 谐波采样 | [SSCHA](sscha-workflow.md) | | FC2 + FC4 loop 修正 | [SCPH](scph-workflow.md) |

## 前置条件

准备经过检查的 primitive、reference supercell 和力来源，并确认环境能够从 `mlfcs` 顶层导入所需 API。教程默认所有快照保持 reference 的原子标签和顺序。

## 工作流

依次完成输入检查、计算或拟合、结果诊断、IFC 保存和目标软件验证。每一步使用独立脚本，使失败可以定位，也避免重跑已经完成的昂贵力计算。

## 预期结果

成功不只意味着生成文件，还应得到合理的参数或位移数量、可接受残差，以及在目标表示下连续且可解释的物理结果。

## 常见问题

若结果异常，先排除原子顺序、单位、cutoff、有限超胞和未收敛原子力，再检查约束和算法设置。不要通过覆盖 baseline 隐藏变化。
