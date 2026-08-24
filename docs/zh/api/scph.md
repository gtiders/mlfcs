---
title: SCPH 无序计算
audience:
  - advanced
status: experimental
code_verified: 4.0.0a4
---

# SCPH 无序计算

本页介绍 Loop-SCPH 温度相关计算的顶层导出。完整签名见
[SCPH API 参考](../reference/scph-api.md)；方法背景见
[Loop-SCPH 理论](../theory/scph.md)。

## LoopSCPH

对 FC2 应用静态四阶 loop 修正，得到温度相关的有效 FC2（待补充主要参数）。

```python
from mlfcs import LoopSCPH
```

## LoopSCPHResult

SCPH 迭代结果：各温度下的有效 FC2 与收敛信息（待补充字段）。

```python
from mlfcs import LoopSCPHResult
```

## harmonic_frequencies

从力常数计算谐波频率（待补充 q 点网格参数与虚频表示）。

```python
from mlfcs import harmonic_frequencies
```

## TemperatureSeriesResult

温度序列结果容器（待补充与 `LoopSCPHResult` 的关系）。

```python
from mlfcs import TemperatureSeriesResult
```
