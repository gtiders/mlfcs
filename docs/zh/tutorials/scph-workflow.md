---
title: SCPH 工作流
audience:
  - advanced
status: experimental
code_verified: 4.0.0a6
---

# SCPH 工作流

## 目标

在温度序列上运行 FC4 loop 自洽计算，并检查迭代残差与有效 FC2 输出。

## 前置条件

准备经过检查的 primitive、reference supercell 和力来源，并确认环境能够从 `mlfcs` 顶层导入所需 API。教程默认所有快照保持 reference 的原子标签和顺序。

## 工作流

依次完成输入检查、计算或拟合、结果诊断、IFC 保存和目标软件验证。每一步使用独立脚本，使失败可以定位，也避免重跑已经完成的昂贵力计算。

## 预期结果

成功不只意味着生成文件，还应得到合理的参数或位移数量、可接受残差，以及在目标表示下连续且可解释的物理结果。

## 常见问题

若结果异常，先排除原子顺序、单位、cutoff、有限超胞和未收敛原子力，再检查约束和算法设置。不要通过覆盖 baseline 隐藏变化。
