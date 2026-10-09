# SCPH：计算温度相关的有效力常数

谐波声子使用固定的二阶力常数。当四阶非谐作用不可忽略时，原子的热涨落会反过来改变有效回复力，声子频率也随温度变化。若已经恢复了 FC2 和 FC4，可以用自洽声子计算来描述这部分重整化。

MLFCS 的 `SCPH` 实现静态四阶环图近似。它反复用当前有效 FC2 计算位移涨落，再用 FC4 更新有效 FC2，直到频率变化足够小。最终得到的仍是一个二阶力常数模型，可以交给[谐波声子接口](harmonic-api.md)计算指定 q 点、网格或能带路径。

本章从含 FC2 和 FC4 的模型开始，介绍如何准备计算、选择迭代设置、判断结果并处理多个温度。四阶环图与协方差的推导见[自洽声子理论](theory/advanced/self-consistent-phonons.md)。

## 1. 准备模型和积分网格

```python
from mlfcs import ForceConstants
from mlfcs.phonon import SCPH

model = ForceConstants.load("fc2-fc4.mlfcs")
scph = SCPH(model, (4, 4, 4), statistics="quantum", time_reversal=True)
```

构造的两个位置参数是 `model` 和 `mesh`。`model` 必须是同一模型空间中同时含 FC2 和 FC4 的 `ForceConstants`，缺少任何一阶或输入类型不对都会抛出 `ValueError`。FC2 提供裸谐波回复力，FC4 在迭代期间保持固定。模型可以同时包含 FC3 等阶数，但它们不进入这项修正。

这里的“裸 FC2”指输入模型的二阶系数；“有效 FC2”指加入四阶修正后的结果：

$$
\Phi^{(2)}_{\rm eff}
=\Phi^{(2)}_{\rm bare}
+\frac12\Phi^{(4)}:\langle uu\rangle.
$$

冒号表示对两个位移指标的收缩。位移协方差 $\langle uu\rangle$ 又取决于有效 FC2，因此不能只计算一次修正就结束。

`mesh` 决定位移涨落的倒空间平均。它接受正整数三元组、非奇异整数 `(3, 3)` 行晶格矩阵，或已有 `QGrid`，与 `Harmonic.mesh()` 的网格含义一致。网格包含 Γ 点，不是用来绘图的高对称路径，也不需要为它构造原子超胞。无效网格会被拒绝，不支持的整数范围会显式失败。

初步计算可以从较小网格开始，但最终有效力常数和频率需要做网格收敛比较。增密绘图路径无法替代增密 SCPH 的积分网格。

### 量子还是经典统计

关键字 `statistics` 默认 `"quantum"`，也可选 `"classical"`；其他值抛出 `ValueError`。

量子统计包含零点涨落，因此即使温度为 0 K，也可能有四阶修正。经典统计使用等分配极限，不包含零点运动；在没有其他奇异性的情况下，0 K 的经典协方差为零。应根据希望描述的统计近似选择，而不是把两者当作不同的求解器。

### 质量与倒空间对称性

质量来自 `model.cluster_space.masses`，按照模型参考胞位点顺序排列。需要改变质量时，应先建立新质量的模型，再构造 `SCPH`，做法见[谐波接口中的质量替换](harmonic-api.md)。传入的参考胞不必是最小原胞，网格与 q 坐标都相对于它定义。

`time_reversal` 默认 `True`，将时间反演相关的 q 与 -q 纳入同一对称组。约化同时只使用保持质量分配、且与网格兼容的空间群操作。若同位素质量打破某些结构对称性，可用操作会减少。

构造后可查看以下公开属性，通常无需自己处理对称性变换：

| 属性 | 含义 |
|---|---|
| `scph.model` | 输入的裸 FC2 与固定 FC4 模型。 |
| `scph.statistics` | 本次使用的统计方式。 |
| `scph.stars` | 网格及其不可约分组；`points` 给出代表点坐标，`weights` 是各组完整点数，`grid` 可查看完整网格。 |
| `scph.plan` | 用于将代表点矩阵变换到完整网格成员的 `StarPlan`。协方差不是标量，计算中需要旋转、位点置换与相位变换。 |

这些属性用于解释与查看已准备的问题。换模型、网格或统计方式时应构造新对象，避免修改属性后与已准备的数据不一致。

## 2. 运行一个温度

```python
result = scph.run(
    300.0,
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)
print(result.converged, result.iterations)
print(result.minimum_mode_thz)
```

`run(temperature, *, start=None, mixing=0.2, tolerance=1e-9, max_iterations=200)` 计算一个温度，返回 `SCPHResult`。`temperature` 的单位为 K，必须有限且非负。

省略 `start` 时从输入模型的裸 FC2 开始。每一步计算四阶修正，再混合更新：

$$
\Phi^{(2)}_{n+1}
=(1-\alpha)\Phi^{(2)}_n
+\alpha\left(\Phi^{(2)}_{\rm bare}
+\Delta\Phi^{(2)}[\Phi^{(2)}_n]\right),
\qquad \alpha=\texttt{mixing}.
$$

混合控制的是参数更新幅度，不是温度或四阶物理作用的强弱。裸 FC2 和 FC4 在整个计算中不变。

若关联空间启用了 `asr=True`，初始化时先将裸 FC2 投影到已有声学子空间；显式初值也先投影。每轮将“裸 FC2 + 四阶修正”的目标投影后再混合，因此初值、混合端点和有效 FC2 都保持在同一约束空间内。投影最小化标准物理参数的欧氏变化，通过已有 lift/adjoint 与 LSMR 完成，不构造全局稠密零空间。投影未收敛时抛出 `RuntimeError`。

### 混合比例、容差与迭代上限

`mixing` 必须有限且在 `(0, 1]` 内，默认 `0.2`。较小比例使每次更新更保守，可能缓解振荡，也会增加迭代次数；设为 1 则直接采用本轮目标 FC2。

`tolerance` 默认 `1e-9`，必须正且有限，单位 THz。收敛指标是相邻两轮**带符号频率的星加权 RMS 变化**，覆盖所有模式，包含 Γ 平移模式；不是 FC2 系数变化或固定点方程残差。不同点的模式按特征值排序，不进行振动模态追踪。

`max_iterations` 默认 200，要求正的 Python 整数，布尔值不合法。非法温度、混合比例、容差或迭代上限会抛出 `ValueError`。

达到迭代上限不会抛出“未收敛”异常，而是返回最后一轮结果，`result.converged` 为 `False`。使用结果前应显式查看它。减小混合比例后每轮频率变化也会减小，因此不能只凭这一数值就断言最终有效模型已经对所有设置收敛；仍需检查迭代轨迹、设置变化和网格收敛。

### 复用一个初始 FC2

可以把已经得到的有效 FC2 作为另一轮计算的初值：

```python
result_300 = scph.run(300.0)
if result_300.converged:
    result_250 = scph.run(250.0, start=result_300.fc2)
```

`start` 必须是含 FC2 的 `ForceConstants`，二阶参数向量形状必须与裸模型一致。类型、阶数或长度不符合会抛出 `ValueError`。晶胞、参数基、位点顺序与质量是否配套由调用者保证，程序不做跨模型兼容性检查。

初值只替换迭代起点，不会把 `start` 变成新的裸 FC2，也不会更换质量或 FC4。最稳妥的是使用同一个 `scph` 产生的已收敛结果。

## 3. 分别判断数值收敛与动力学稳定

`converged=True` 表示频率变化满足停止阈值，并不保证所有物理模式稳定。某些自洽解仍可能存在虚频。

```python
print("数值收敛：", result.converged)
print("含虚频：", result.has_imaginary_modes)
print("最小非平移模式：", result.minimum_mode_thz)
```

结果的所有公开字段与属性如下：

| 字段或属性 | 怎样解释 |
|---|---|
| `temperature` | 计算温度，单位 K。 |
| `fc2` | 最后一轮有效 `ForceConstants`，只包含 FC2；未收敛时也是最后的迭代值。 |
| `frequencies` | `(不可约点数, 3*N)` 的带符号 THz 频率，包含 Γ 平移模式，与 `stars.points` 对齐。 |
| `stars` | 本次网格及不可约代表点、成员和权重。 |
| `history` | 按迭代顺序排列的 `SCPHStep` 元组。 |
| `converged` | 是否满足频率变化停止判据。 |
| `minimum_mode_thz` | 网格上最小的非平移带符号频率；在 Γ 点先排除三个刚性平移方向。若网格没有剩余物理模式，例如单原子参考胞的仅 Γ 网格，则为 `None`。 |
| `iterations` | `history` 的长度，即实际执行的更新次数。 |
| `has_imaginary_modes` | `minimum_mode_thz` 是否严格小于零；与数值收敛分开判断，没有额外虚频幅度容差。 |

每个 `SCPHStep` 的公开字段是 `index`、`frequency_change_thz` 和 `minimum_frequency_thz`：迭代编号从 1 开始，后两者单位 THz。

其中，`minimum_frequency_thz` 是这一轮频率数组的最小值，**包含 Γ 平移模式**；它与最终结果中排除平移的 `minimum_mode_thz` 不是同一个诊断。两者可能不同，不应将极小的 Γ 平移舍入误差直接当成内部软模。

可以查看收敛轨迹：

```python
for step in result.history:
    print(step.index, step.frequency_change_thz, step.minimum_frequency_thz)
```

频率变化持续振荡、停滞或仍较大时，应检查混合设置、迭代次数和模型本身，而不是只增加绘图点数。

### 当前如何处理不稳定或零模式

Γ 点的三个刚性平移方向在协方差计算前被投影掉，避免它们的零频率导致位移方差发散。但普通频率输出仍保留这些模式，这与 `Harmonic` 的输出约定一致。

对于负试探曲率，当前实现使用其绝对值构造正的试探协方差，同时保留带符号频率用于诊断。这是一项数值迭代选择，不表示负曲率对应稳定谐振子，也不保证不稳定裸模型必然会稳定化。

非平移的严格零模会触发发散协方差的 `ValueError`，即使在经典 0 K 下也会先做这项检查。非有限协方差或环修正同样会报错；程序不会自动加入频率下限来继续迭代。

## 4. 计算一组温度

```python
results = scph.run_many(
    [0, 100, 200, 300],
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)

for result in results:
    print(result.temperature, result.converged, result.minimum_mode_thz)
```

`run_many(temperatures, **kwargs)` 要求输入为非空、一维、有限、非负且严格递增的温度序列。重复温度、降序列表、负值或非有限值都会抛出 `ValueError`。

实际求解按高温到低温进行，返回列表仍按输入的升序排列。每个已收敛结果的有效 FC2 会作为下一个低温的初值；未收敛时，下一个温度重新从裸 FC2 开始。即使结果有虚频，只要数值收敛，它仍会传递给下一温度。

`kwargs` 可传入 `run()` 的 `mixing`、`tolerance` 和 `max_iterations`。不能传 `start`，因为温度序列内部管理初值；传入会抛出 `TypeError`。其他不支持的关键字也会被拒绝。

因此，温度序列并非一组各自从裸模型开始的独立计算。若要比较初值依赖，应另外逐温度调用 `run()`，明确选择起点。

## 5. 用有效 FC2 绘制声子谱

有效模型可以直接交给 `Harmonic`。比较裸谱与重整化谱时，先生成一份共同路径，避免不同采样影响比较：

```python
from mlfcs.phonon import Harmonic

path = model.cluster_space.primitive_atoms.cell.bandpath(npoints=200)
bare_bands = Harmonic(model).run_band_path(path)
if result.converged:
    effective_bands = Harmonic(result.fc2).run_band_path(path)
```

`result.frequencies` 只是在 SCPH 积分网格上的代表点频率，不是沿高对称路径的能带。绘图时要重新用有效 FC2 求路径频率；这不会重新运行 SCPH。

需要完整积分网格频率时，可以使用 `result.stars.expand(result.frequencies)`。该操作适用于频率这种对称不变量；矩阵协方差则需要完整对称变换，不能这样直接复制。

`result.fc2.save(...)` 保存这个温度下的有效二阶模型，不会同时保存迭代历史、温度标签或原输入 FC4。建议在文件名或独立计算记录中保留温度、网格、统计方式和收敛状态。

## 6. 这项近似的边界

当前计算只包含静态四阶环修正，不使用三阶自能，不计算线宽、寿命或热导率，也不是 SSCHA。温度进入协方差，参考几何和输入 FC4 保持固定；它不会自动求解热膨胀或改变平衡结构。

环修正保留在输入空间已有的 FC2 相互作用支持内。FC4 收缩产生但不属于该 FC2 空间的原子对贡献会被忽略，不会自动扩大二阶 cutoff。对保留的代表张量，当前通过选定物理分量恢复参数，并非对全部修正张量做一次全局 Frobenius 最小二乘投影。

ASR 开关继承自模型的 `ClusterSpace`：`asr=True` 时，SCPH 使用已准备的声学坐标投影初值及每轮目标；`asr=False` 时保持未约束迭代，不自动补建坐标。投影针对物理参数欧氏度量，不是展开张量的 Frobenius 度量，也不扩大 FC2 支持范围。Γ 协方差中的平移投影仍只是去除零模，与参数 ASR 投影是不同操作。

原生文件只保存物理模型，加载时不恢复声学坐标，因此 `ForceConstants.load(...)` 得到的空间默认 `asr=False`。需要约束 SCPH 时，可在恢复同一参数布局后重新绑定已准备 ASR 的空间：

```python
from mlfcs import ClusterSpace

saved_space = model.cluster_space
space = ClusterSpace(
    saved_space.primitive_atoms,
    cutoffs={block.order: block.cutoff for block in saved_space.blocks},
    max_body_orders={block.order: block.max_body_order for block in saved_space.blocks},
    symprec=saved_space.symprec,
    asr=True,
)
# 调用者须确认重建空间与保存模型的参数布局一致。
model = ForceConstants(space, model.coefficients)
scph = SCPH(model, (4, 4, 4))
```

重新构造空间会重新准备对称性和声学坐标；同一次拟合流程中已有 `asr=True` 模型时可以直接传入，不必重复这一步。

同样，它不接受额外长程谐波背景，也不自动加入 NAC。不能将由纯短程模型得到的协方差解释成已经含完整极性长程作用的协方差。当前能力与边界见[范围与限制](theory/limitations.md)。

## 7. 从已有模型到温度声子谱的完整例子

下面假设已有一个保存为 `fc2-fc4.mlfcs` 的模型。它可以包含更多阶数，但必须含 FC2 和 FC4。以下加载示例默认未准备 ASR；需要约束时先执行上一节的空间重绑定，再创建 `SCPH`。示例计算三个温度，并只绘制数值已收敛的结果：

```python
import matplotlib.pyplot as plt
from mlfcs import ForceConstants
from mlfcs.phonon import Harmonic, SCPH

model = ForceConstants.load("fc2-fc4.mlfcs")
scph = SCPH(model, (4, 4, 4), statistics="quantum", time_reversal=True)
results = scph.run_many(
    [100, 200, 300],
    mixing=0.2,
    tolerance=1e-9,
    max_iterations=200,
)

path = model.cluster_space.primitive_atoms.cell.bandpath(npoints=200)
bare = Harmonic(model).run_band_path(path)
fig, ax = plt.subplots()
for index, segment in enumerate(bare.segments):
    ax.plot(
        bare.distances[segment], bare.frequencies_thz[segment],
        color="0.5", linestyle="--",
        label="Bare" if index == 0 else None,
    )

for color_index, result in enumerate(results):
    print(result.temperature, result.converged, result.iterations,
          result.minimum_mode_thz, result.has_imaginary_modes)
    if not result.converged:
        continue
    bands = Harmonic(result.fc2).run_band_path(path)
    for index, segment in enumerate(bands.segments):
        ax.plot(
            bands.distances[segment], bands.frequencies_thz[segment],
            color=f"C{color_index}",
            label=f"{result.temperature:g} K" if index == 0 else None,
        )

ax.axhline(0, color="0.5", linewidth=0.5)
ax.set_xticks(bare.tick_positions, bare.tick_labels)
ax.set_xlim(bare.distances[0], bare.distances[-1])
ax.set_ylabel("Frequency (THz)")
ax.legend()
fig.tight_layout()
plt.show()
```

网格、容差与迭代设置仅展示调用方式，能否收敛及得到怎样的频率取决于输入模型。带完整执行输出的案例见 [K₄As₄Pt₂ 教程](notebooks/k4as4pt2.ipynb)。

使用时先确认模型同时含 FC2 与 FC4，再选择积分网格和统计方式。`run()` 或 `run_many()` 给出温度相关的有效 FC2；判断结果时分别查看数值收敛和虚频诊断，随后再交给 `Harmonic` 绘制声子谱。
