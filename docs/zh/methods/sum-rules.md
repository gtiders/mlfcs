# 求和规则

MLFCS 在有限差分和拟合中默认以 `acoustic_sum_rule=True` 施加 ASR。它是每一阶各自的
orbit 参数空间约束。

Born-Huang 与 Huang 是不同语义的 FC2 物理后处理：在力常数生成或从原生 HDF5 读取后
显式调用。

```python
from mlfcs import read_hdf5

result = read_hdf5("mlfcs.h5")
constrained = result.enforce_rotational_sum_rules(
    born_huang=True,
    huang=True,
)
fc2 = constrained.force_constants
print(constrained.diagnostics)
```

默认 `strength=1.0` 是保留数值秩上的严格投影。`[0, 1]` 内的值只缩放
Born-Huang/Huang 的修正，ASR 始终重新严格满足。`tolerance` 是以中位非零最近像距离
无量纲化后的谱截断。

投影器要求经过验证的 `StructureRelation` 和带晶格标签的稀疏 FC2。简并最近像等权：
Born-Huang 使用向量平均，Huang 使用二阶外积平均。它返回新结果，保留原始结果，且
不改动 FC3、FC4 或任何其他阶。

Huang 是零应力条件，只适用于无应力参考结构；它不替代长程静电或 NAC 处理。
