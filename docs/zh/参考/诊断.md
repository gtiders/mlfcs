---
title: 诊断
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 诊断

## 用途

出现异常结果时，先检查结构关系：primitive 原子数、参考原子顺序、超胞矩阵和最大映射残差。然后检查 cutoff 壳层、力的单位、ASR 残差以及目标 writer 所需的超胞。 对于 SCPH，检查 `result.history`、最后的频率变化 RMS 和负的频率平方。未收敛的有效 FC2 只能作为诊断 输出，不能作为生产结果。

## 稳定性约定

Reference 只描述当前公开行为。参数默认值、单位、返回对象和异常必须与已验证版本一致；planned 或 research 功能不会出现在稳定接口中。

## 诊断顺序

先阅读异常消息和同一调用产生的诊断，再检查输入结构与数据形状。若问题涉及物理近似，应转到 Theory；若涉及具体流程，应转到 How-to。

## 版本与兼容性

内部模块路径不属于公共兼容承诺。用户代码应从 `mlfcs` 顶层导入公开对象，并在保存结果时记录 MLFCS 版本和 schema。
