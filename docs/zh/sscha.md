# SSCHA

`SSCHA` 在固定晶胞和质心下进行随机自洽谐性力匹配。它从试探谐性 FC2 生成成对位移，用 ASE 计算器重新计算力，拟合 FC2、对每次拟合更新施加 ASR，并迭代试探模型。

## 构造 `SSCHA`

```python
from mlfcs.reciprocal import SSCHA

solver = SSCHA(
    mapping,
    mapping.supercell.matrix,
    initial=fc2,
    statistics="quantum",
    seed=7,
)
```

| 参数 | 含义 |
| --- | --- |
| `mapping` | FC2-only `ClusterSpace` 对应的 `ClusterMap`；其超胞必须能结构性辨识 FC2。 |
| `mesh` | `QGrid`、长度为 3 的对角网格或 3×3 整数矩阵。其矩阵必须等于声明超胞的矩阵。 |
| `initial` | 可选的 FC2-only `ForceConstants`，簇空间和质量必须与 mapping 一致。省略时，SSCHA 从随机笛卡尔位移生成初始试探模型。 |
| `statistics` | `"quantum"`（默认）或 `"classical"` 模态统计。经典统计要求正温度。 |
| `seed` | 可选的整数随机种子，范围 `[0, 2**64)`。省略时自动生成，并记录在结果中。 |
| `bootstrap_displacement` | 可选的正笛卡尔位移尺度，单位 Å；用于初值缺失或不稳定时构造试探模型。若给定试探模型不稳定且该值为 `None`，则不启用 bootstrap。 |

求解器保存 `mapping`、不可约 `stars`、`initial`、`statistics`、`seed` 和 `bootstrap_displacement`。网格必须与声明超胞匹配，因此倒空间采样与实空间力数据使用同一周期晶胞。

## 运行一个温度

```python
result = solver.run(
    300.0,
    calculator,
    pairs=128,
    max_pairs=512,
    max_iterations=20,
    tol_thz=0.01,
    mixing=0.5,
)
print(result.status, result.fc2.orders, len(result.history))
```

`run(temperature, calculator, *, pairs=128, max_pairs=None, max_iterations=20, tol_thz=0.01, mixing=0.5, max_backtracks=8, energy=False, start=None)` 参数如下：

| 参数 | 含义 |
| --- | --- |
| `temperature` | 有限的非负温度，单位 K。零温度使用量子统计。 |
| `calculator` | 用于重新计算力的 ASE `Calculator`；`energy=True` 时也计算能量。 |
| `pairs` | 独立位移对的初始数量，至少为 2。每对计算正、负位移。 |
| `max_pairs` | 最大样本对数。`None` 时等于 `pairs`；不确定度驱动的采样可增加到该上限。 |
| `max_iterations` | 最大自洽更新次数。 |
| `tol_thz` | 频率变化和采样不确定度共同使用的目标值，单位 THz。 |
| `mixing` | 初始接受的 FC2 更新比例，范围 `(0, 1]`。 |
| `max_backtracks` | 若更新产生不稳定试探模型，最多尝试将更新比例连续减半的次数。 |
| `energy` | 是否计算势能并估计非谐自由能；默认 `False`。 |
| `start` | 可选的 FC2-only 初值，簇空间和质量必须与 mapping 一致；优先于构造函数中的 `initial`。 |

若未提供试探模型，算法会用成对随机笛卡尔位移进行 bootstrap。若试探模型不稳定，提供正的 `bootstrap_displacement` 即可启用相同恢复路径。采样不会静默将负曲率改为正曲率：不稳定试探会被报告，候选更新会逐步回退，直到找到稳定模型或达到设定上限。

`SSCHAResult` 包含 `temperature`、`fc2`、`history`、`status`、`n_qpoints`、`n_irreducible`、`mapping_fingerprint`、`seed`、`message` 和 `bootstrap_displacement`；`converged` 由 `status` 得出。状态包括 `converged`、`max_iterations`、`insufficient_samples` 和 `unstable`。

每个 `SSCHAStep` 记录样本对数、频率变化和不确定度、最低模、实际接受的混合比例、拟合 RMSE、ASR 残差、平均力，以及可选的自由能估计及误差。

## 计算温度序列

```python
results = solver.run_many(
    [300, 400, 500],
    calculator,
    pairs=128,
    max_pairs=512,
)
```

`run_many(temperatures, calculator, **kwargs)` 要求温度严格递增、有限且非负。实际按高温到低温执行，使较低温从前一温度的 FC2 热启动；返回结果按输入升序排列。若某个结果未收敛，序列会在计算更低温之前停止并抛出 `SSCHAContinuationError`；已完成结果和停止时的结果都可从异常对象读取。

## 质量与对称性

SSCHA 从 `ClusterMap` 开始；质量取自 `mapping.space.primitive.masses`，并传给采样生成的 ASE 结构。当前构造函数没有独立的质量覆盖参数。若需使用指定质量，应在构造 `ClusterSpace` 和 mapping 前设置原胞 ASE `Atoms` 的质量。谐性质量加权必须与倒空间星约化相容；若质量不满足模型原胞对称性，谐性路径会拒绝计算。
