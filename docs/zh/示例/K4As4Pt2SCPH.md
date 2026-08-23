---
title: K4As4Pt2 SCPH
audience:
  - user
status: stable
code_verified: 4.0.0a4
examples:
  - examples/scph/K4As4Pt2
---

# K4As4Pt2 SCPH

## 案例目标

FC4 loop 自洽、收敛诊断与温度相关声子谱。

## 输入与来源

案例目录保留可复现所需的结构、配置、脚本和允许进入版本库的参考数据。第三方数据应标明来源；运行产物和缓存按目录 README 的政策处理。

## 运行顺序

先阅读案例目录 README，再按编号或任务名称分别运行准备、计算/拟合、分析和绘图脚本。脚本不跨案例复用，以保证案例能够独立复现。

## 应检查的结果

记录 orbit/参数或位移数量、拟合误差、约束残差、IFC 文件和最终图像。声子谱应检查 NaN、异常尺度、不连续分支和非预期虚频。

## 回归意义

该案例用于验证当前代码的物理结果，而不是只测试文件能否生成。内部 ordering 可以变化，但同一结构表示下的 IFC、预测力和派生物理量应保持在数值容差内。

## 案例目录

[examples/scph/K4As4Pt2](https://github.com/gtiders/mlfcs/tree/dev/examples/scph/K4As4Pt2) 是本案例的权威目录，包含输入、独立脚本、保留的参考数据和生成产物策略。
