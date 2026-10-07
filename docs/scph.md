# SCPH

`SCPH` 执行静态四阶环图自洽声子计算。它从同时含 FC2 和 FC4 的 `ForceConstants` 模型开始，根据所选网格构造倒空间不可约星，并逐温度迭代有效 FC2。模型可以同时包含 FC3，但静态环修正不使用 FC3。

## 构造 `SCPH`

```python
from mlfcs.phonon import SCPH

scph = SCPH(model, mesh, statistics="quantum", time_reversal=True)
```

| 参数 | 含义 |
| --- | --- |
| `model` | 同一簇空间上的 `ForceConstants`，必须包含二阶和四阶。FC4 块提供环修正，FC2 块提供裸谐性模型。 |
| `mesh` | `QGrid`、正整数对角网格（如 `(4, 4, 6)`），或非奇异的 3×3 整数超胞矩阵。按原胞对称性的质量保持子群约化。 |
| `statistics` | 模态协方差统计方式：`"quantum"`（默认）或 `"classical"`。 |
| `time_reversal` | 倒空间星是否包含时间反演配对；默认 `True`。 |

对象提供 `model`、`stars` 和 `statistics`。倒空间网格基于精确整数数据；频率和协方差计算在星代表点上执行。

质量直接来自只读的 `model.cluster_space.masses`。结构等价站点可以使用不同质量，只有保持质量的对称操作参与倒空间约化。如何无须重建 cluster space 就替换质量，见[谐波频率与原子质量](harmonic-api.md)。

## 运行一个温度

```python
result = scph.run(
    300.0,
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)
print(result.converged, result.iterations, result.minimum_mode_thz)
```

`run(temperature, *, start=None, mixing=0.2, tolerance=1e-9, max_iterations=200)` 参数如下：

| 参数 | 含义 |
| --- | --- |
| `temperature` | 非负温度，单位 K。 |
| `start` | 可选的含 FC2 的 `ForceConstants`。用户负责参数布局和质量配套，程序不检查。省略时从模型的裸 FC2 开始。 |
| `mixing` | 每次 FC2 更新使用的线性混合比例，范围 `(0, 1]`。 |
| `tolerance` | 星加权 RMS 频率变化的收敛阈值，单位 THz。 |
| `max_iterations` | 自洽更新的最大次数。 |

每轮用当前 FC2 计算 FC4 环修正，再混合目标更新并比较有符号频率。`SCPHResult` 包含 `temperature`、有效 `fc2`、代表点上的 `frequencies`、`stars`、迭代 `history`、`converged` 和 `minimum_mode_thz`；`iterations` 与 `has_imaginary_modes` 是便捷属性。虚频以负的有符号频率报告，本身不表示迭代失败。协方差计算使用负曲率的绝对值；非平移的严格零模会导致谐性协方差发散并报错。

每个 `SCPHStep` 记录 `index`、`frequency_change_thz` 和 `minimum_frequency_thz`。

## 计算温度序列

```python
results = scph.run_many(
    [0, 100, 200, 300],
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)
```

`run_many(temperatures, **kwargs)` 要求温度序列非空、非负且严格递增。实际按高温到低温执行，返回的 `SCPHResult` 列表仍按输入升序排列。已收敛的高温 FC2 会作为下一个低温的初值；未收敛结果不会传递给下一次运行。虚频诊断与数值收敛状态分开报告。

教程 [K4As4Pt2：FC2–FC4 拟合、热导率与 SCPH](notebooks/k4as4pt2.ipynb) 展示了温度序列、高温到低温的初值传递，以及使用各温度 FC2 绘制声子谱。
