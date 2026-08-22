---
title: 诊断
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 诊断

出现异常结果时，先检查结构关系：primitive 原子数、参考原子顺序、超胞矩阵和最大映射残差。然后检查
cutoff 壳层、力的单位、ASR 残差以及目标 writer 所需的超胞。

对于 SCPH，检查 `result.history`、最后的频率变化 RMS 和负的频率平方。未收敛的有效 FC2 只能作为诊断
输出，不能作为生产结果。
