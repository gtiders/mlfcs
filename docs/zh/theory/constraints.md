---
title: 求和规则
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# 求和规则

MLFCS 在有限差分和拟合中默认以 `acoustic_sum_rule=True` 施加 ASR。两条路径都先得到无约束物理 orbit
参数 $\theta_0$，再共用同一个逐阶欧氏投影器投到 $A\theta=0$。因此 ASR 是物理后处理，不是缩减后的
拟合坐标系，也不是受约束最小二乘估计量。

这个区别是有意的。拟合 Gram 只描述力观测，因此可在开启或关闭 ASR 时复用。后投影寻找离无约束解最近的
ASR 可行参数，而不是训练损失最小的 ASR 可行参数，所以训练残差可能增加；结果会同时报告投影前后数值。

Born-Huang 与 Huang 是不同语义的 FC2 物理后处理：在力常数生成或从原生 HDF5 读取后
显式调用。

```python
from mlfcs import read_hdf5

result = read_hdf5("mlfcs.h5")
constrained = enforce_rotational_sum_rules(result, 
    born_huang=True,
    huang=True,
)
fc2 = constrained.force_constants
print(constrained)
```

默认 `strength=1.0` 是保留数值秩上的严格投影。`[0, 1]` 内的值只缩放
Born-Huang/Huang 的修正，ASR 始终重新严格满足。`tolerance` 是以中位非零最近像距离
无量纲化后的谱截断。

投影器要求经过验证的 `StructureRelation` 和带晶格标签的稀疏 FC2。简并最近像等权：
Born-Huang 使用向量平均，Huang 使用二阶外积平均。它返回新结果，保留原始结果，且
不改动 FC3、FC4 或任何其他阶。

Huang 是零应力条件，只适用于无应力参考结构；它不替代长程静电或 NAC 处理。
