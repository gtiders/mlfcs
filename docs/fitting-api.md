# 力拟合：从位移与力样本恢复模型

有限差分要求按指定的正负位移组合计算力。但如果已经有随机扰动、分子动力学或其他来源的位移构型，往往更希望直接利用这些样本：给定每帧的位移和原子力，找到一组能够解释它们的力常数。

`FitSystem` 将这个问题整理为最小二乘方程。它使用 `ClusterSpace` 中的全部阶数，因此可以同时拟合二阶、三阶和更高阶力常数。模型、参考超胞和数据仍通过[核心概念](core-concepts.md)中的三个对象准备；拟合从 `ForceDataset` 开始。

下面先说明怎样建立和求解系统，再讨论误差、方程表示及复用方式。最后给出一个可独立运行的 Al/EMT 小例子。

## 1. 把数据交给拟合系统

如果原胞、参考超胞和带力构型已经保存在文件中，可以这样准备：

```python
from ase.io import read, iread
from mlfcs import ClusterSpace, ClusterMap, ForceDataset, FitSystem

space = ClusterSpace(
    read("POSCAR"),
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
    asr=True,
)
mapping = ClusterMap(space, read("SPOSCAR"))
dataset = ForceDataset(mapping, iread("train.extxyz", index=":"))

system = FitSystem(dataset)
model = system.solve()
print("训练力 RMSE：", system.rmse(model), "eV/Å")
```

这里同时拟合 FC2 和 FC3，因为它们都包含在 `space` 中。`FitSystem` 没有另一个 `order=` 选项；希望只拟合某些阶数时，应在建立 `ClusterSpace` 时就选择它们。

构造的第一个参数 `dataset` 必须是 `ForceDataset`，不能直接传入 ASE 结构列表或力数组。每帧的力必须已经计算并保存，构造系统不会调用 calculator。位移、原子顺序和晶胞的要求见[核心概念](core-concepts.md)。

可选关键字 `representation` 决定方程怎样保存以及使用哪一种内置求解器，默认是 `"normal"`，另一个选项为 `"raw"`。这两个选项都使用相同的数据与目标，区别将在后面说明。

构造系统会要求超胞保留模型全部标准参数方向。如果超胞折叠造成结构混叠，会抛出 `AliasingError`，不能靠增加同一超胞的训练样本绕过。输入不是数据集会抛出 `TypeError`；表示名称不合法会抛出 `ValueError`。很大的模型还可能超出可表示的数组范围或实际内存容量。

### 拟合中使用了哪些信息

每帧位移决定这一帧的力如何依赖模型参数。Taylor 展开的负号、阶乘、对称性以及周期像贡献都已经计入设计矩阵，无需手动调整原子力的符号或缩放高阶系数。

将所有帧的力分量放在一起，问题写为

$$
\min_\eta \|A\eta-f\|_2^2.
$$

$f$ 是目标力，$A$ 是位移生成的设计矩阵，$\eta$ 是当前拟合坐标。每帧按照原子顺序展开，每个原子的 x、y、z 分量连续；不同帧按数据集顺序连接。

模型从二阶开始，没有另一个常数参考力项。系统不会自动减去参考力或平均力，也不提供逐帧权重。若目标是去掉某份已知贡献后的力，应先使用 `dataset.subtract_forces(...)`，再构造系统。例如长程力扣除见[长程力与 Ewald](ewald-api.md)。

### 极性材料先处理长程力

对于极性绝缘体，若已知与参考结构配套的 Born 电荷和电子介电张量，建议先扣除长程偶极力，再联合拟合短程 FC2 与高阶力常数。这样可以减少有限截断模型将遗漏的长程谐波响应吸收到高阶参数中的风险，通常更有利于高阶参数的稳定性与可解释性；是否改善仍应通过独立验证与截断收敛比较确认。

```python
# ewald 是按同一个 mapping 构造的 DipoleEwald 对象。
short_data = dataset.subtract_forces(ewald.forces(dataset.displacements))
system = FitSystem(short_data, representation="raw")
model = system.solve()
```

当前 Ewald 只扣除长程 FC2，不处理真正的高阶长程作用。需要总 FC2 时，应回加扣除时使用的同一份长程贡献。完整流程见[长程力与 Ewald](ewald-api.md)。

### ASR 的开关仍在 `ClusterSpace`

`asr=True` 时，拟合直接在满足平移不变性的自由声学坐标中求解；关闭时使用标准物理参数。`solve()` 返回的模型始终恢复为标准物理参数布局，因此后续保存和使用模型的方式不变。

这意味着 `system.n_parameters` 等于 `space.n_free_parameters`，开启 ASR 后通常小于 `space.n_parameters`。后者仍是完整标准参数布局的长度，不能拿它代替拟合矩阵的列数。

拟合中的 ASR 与有限差分的使用方式不同：这里从求解开始就限制在约束空间内，有限差分则在恢复物理参数后投影。数学定义见[平移与旋转不变性](theory/invariance-constraints.md)。

## 2. 选择怎样保存方程

对小到中等系统，如果希望直接求解并查看逐条方程，可以选择 `raw`：

```python
raw = FitSystem(dataset, representation="raw")
model = raw.solve()
```

`raw` 保留完整的 $A$ 和 $f$，使用列缩放后的直接 SVD 最小二乘。它没有需要用户设置的迭代容差或步数，但大型稠密分解可能需要较多时间与内存。

训练帧很多时，可以使用默认的 `normal`：

```python
normal = FitSystem(dataset, representation="normal")
model = normal.solve()
```

它逐帧生成设计矩阵并累计

$$
H=A^TA,\qquad g=A^Tf,\qquad c=f^Tf,
$$

保存这些统计量，然后用 MINRES 求解。这样无需保留全部设计矩阵行；但 $H$ 仍是稠密的参数平方规模矩阵，并非矩阵无关求解。

| 表示 | 保存的方程 | 内置求解器 | 更需要留意的地方 |
|---|---|---|---|
| `raw` | 原始设计矩阵与目标力 | 直接 SVD 最小二乘 | 完整矩阵与分解工作区的内存、稠密分解时间 |
| `normal` | 正规矩阵、右端及目标力平方范数 | MINRES | 参数平方规模内存、迭代收敛、正规方程的精度损失 |

两者在精确算术下描述同一个最小二乘目标，数值结果却不一定相同。满列秩时，形成 $A^TA$ 会将二范数条件数平方。参数方向本来就难以区分时，正规方程可能损失更多精度；直接路线仍有数值秩判断，也不能让缺乏信息的数据变得充分。

如果有 $m$ 条力方程、$d$ 个拟合坐标，单个 `raw` 设计矩阵约需 $8md$ 字节，单个 `normal` 矩阵约需 $8d^2$ 字节。这只是主矩阵：构造、ASR 坐标转换、归一化和求解都另需暂存。不能仅根据最终矩阵大小判断峰值内存。

`ForceDataset` 的位移与力数组也已整体驻留内存。`normal` 的逐帧累积只针对设计矩阵，并不意味着训练数据本身从头到尾以流式方式保存。系统构造完成后保留 `cluster_space` 和方程，不保留输入数据集或 mapping。

## 3. 求解参数与列归一化

先尝试默认的 `system.solve()` 即可。表示已决定内置算法，没有额外的求解器选择参数，也没有自动换算法或正则化回退。

对于 `normal`，可以设置两个关键字：

```python
model = normal.solve(rtol=1e-8, maxiter=1000)
```

`rtol` 默认 `1e-8`，必须正且有限，是 MINRES 的相对停止容差。`maxiter` 默认 `1000`，必须为正整数，是迭代上限。容差不是目标力 RMSE，也不是数据归一化参数；收紧容差不会自动改善模型截断或训练数据。

对于 `raw`，调用 `raw.solve()`，不传求解参数。传入 `rtol`、`maxiter` 或其他不支持的选项会抛出 `TypeError`。直接求解采用 SciPy/LAPACK 默认的机器精度秩阈值，不暴露迭代停止设置。

### 为什么默认会缩放参数列

不同阶数的参数对力的贡献含不同次幂的位移。同一组样本中，各列的数值大小可能差很多。求解器会临时将每列平衡到单位二范数，求解后再恢复原尺度。

这种缩放由方程决定，不需要再设置一组“归一化参数”。公开矩阵和目标力仍保持原值，求解不会改变它们。返回的 $p$ 阶力常数单位仍是 eV/Å$^p$。

缩放能平衡列的量级，却不能消除列之间的线性相关，也不能补上完全没有被激发的方向。`raw` 数值秩亏时返回缩放坐标中的最小范数解；这不等于未经缩放的物理参数最小范数解。`normal` 同样在缩放坐标中迭代，不应假定两种路线对秩亏问题一定选择相同参数。

### 如果求解失败

在两条路线中，完全为零的参数列都会触发 `UnobservedParameterError`。可先查看 `system.unobserved_parameters`，它给出从 0 开始的缺失列索引，开启 ASR 时指的是自由声学坐标。应补充能激发这些方向的构型，或重新选择模型与采样。

这个检查只找**精确零列**。返回空元组不代表设计矩阵满秩：两个非零列也可能线性相关。超胞满秩、没有零列，以及训练数据具有良好的数值条件，是三个不同的判断。

MINRES 未满足停止条件时抛出 `RuntimeError`，直接分解也可能因数值问题失败。非法容差、迭代上限或非有限求解数据会被拒绝。失败后应先检查数据、规模和参数相关性，再决定是否增加迭代、调整容差或改用 `raw` 比较。

## 4. 看结果时先看力误差

`solve()` 返回一个包含全部拟合阶数的 `ForceConstants`。系数可按 `model.coefficients[order]` 查看；系统自身的参数数则是求解坐标数，两者在 ASR 开启时不必相同。

```python
print(model.orders)
print("残差范数：", system.residual(model))
print("力 RMSE：", system.rmse(model), "eV/Å")
print("相对力误差：", system.relative_error(model))
```

三个方法都接收 `model_or_parameters`：可以是匹配的完整模型，也可以是长度为 `system.n_parameters` 的有限、未经求解器缩放的拟合坐标向量。

- `residual(...)` 返回所有力分量的残差二范数，是一个标量，不是逐帧残差数组。
- `rmse(...)` 将该范数除以 `sqrt(system.n_equations)`，得到以 eV/Å 为单位的每个笛卡尔力分量 RMSE，不是每原子矢量误差。
- `relative_error(...)` 将残差范数除以目标力范数，结果无量纲。目标力全为零时，零残差返回 0，否则返回无穷大。

当前这三项衡量的是系统中的数据。训练误差小并不证明预测准确，应保留独立构型，在同一个模型空间下另建验证系统：

```python
# validation_structures 是未参与训练、已带力的构型。
validation_data = ForceDataset(mapping, validation_structures)
validation = FitSystem(validation_data)
print("验证力 RMSE：", validation.rmse(model), "eV/Å")
```

这里不用对验证系统调用 `solve()`。评估要求模型包含系统的全部阶数，参数布局匹配；程序不验证跨模型的几何或基是否相同。开启 ASR 时，传入的模型还必须属于该系统准备的声学约束子空间，任意未满足 ASR 的模型不能自动投影后评估。无效形状、非有限坐标或不匹配的阶数会抛出 `ValueError`。

`raw` 直接计算力残差；`normal` 从 $\eta^TH\eta-2\eta^Tg+c$ 恢复残差平方。接近完美拟合时，这个大数相减可能丢失微小残差；实现会在浮点舍入范围内将其截为零。超出舍入范围的负残差平方会报错，因此 `normal` 返回零不能作为机器精度下零误差的证明。

## 5. 检查系统中保存了什么

除了用于日常诊断的计数，还可以读取公开方程，与外部求解器或分析程序衔接。数组均只读且未经内置求解器缩放。

| 属性 | 含义 |
|---|---|
| `cluster_space` | 方程使用的模型空间；用于解释参数布局与恢复模型。 |
| `representation` | 当前保存方式，`"raw"` 或 `"normal"`。 |
| `n_structures` | 累计训练帧数。 |
| `n_equations` | 笛卡尔力方程总数；单个超胞的数据为 `3 * mapping.n_atoms * n_structures`。 |
| `n_parameters` | 拟合列数，等于 `cluster_space.n_free_parameters`。 |
| `force_squared_norm` | 目标力平方和 $c=f^Tf$，两种表示都保存，用于误差评估。 |
| `unobserved_parameters` | 当前方程中精确零列的索引元组，不是完整的秩诊断。 |
| `design_matrix`、`forces` | 仅 `raw` 可用，分别为 `(m, d)` 设计矩阵与 `(m,)` 目标力向量。这里的 `forces` 已展开，与数据集的三维数组不同。 |
| `normal_matrix`、`normal_rhs` | 仅 `normal` 可用，分别为 `(d, d)` 的 $H$ 和 `(d,)` 的 $g$。 |

在错误表示上访问专属属性会抛出 `ValueError`，例如不能从 `normal` 读取 `design_matrix`。正规统计量没有保留每条原始方程，不能据此恢复逐帧残差。

## 6. 转换与合并已有系统

若已有原始方程，可以显式转为正规表示：

```python
normal_from_raw = raw.to_normal()
```

`to_normal()` 没有参数，保留目标力平方和及样本计数，原 `raw` 系统不变；对 `normal` 调用则返回它自身。转换失去逐行信息，没有反向恢复 `raw` 的接口。

分批准备数据时，可以将两个系统合并再求解：

```python
# 两批数据使用同一个 mapping 与参数布局。
first = FitSystem(ForceDataset(mapping, first_structures))
second = FitSystem(ForceDataset(mapping, second_structures))
combined = first + second
model = combined.solve()
```

`+` 返回新系统，不改变输入。`normal` 累加统计量，`raw` 按左边再右边拼接方程；计数和目标力平方和也相加。表示不同会抛出 `ValueError`，混合前应显式使用 `to_normal()`。

合并要求列具有相同含义，不只是数量相同。模型阶数、基、参数顺序及 ASR 坐标必须一致，调用者负责核对；程序不会检查跨系统的模型身份。数据批次可以来自不同超胞，但必须映射到同一套模型与拟合坐标。

`FitSystem` 没有保存／加载接口。保存结果用 `model.save("fit.mlfcs")`；若需要保存方程，可自行存储公开数组及其解释所需的模型信息，但当前没有从文件或公开数组直接恢复系统的公共构造入口。

## 7. 使用自己的求解器

多数场景可以直接使用 `solve()`。需要自己的求解算法时，可以读取方程，然后通过 `force_constants(parameters)` 将结果绑定为模型：

```python
from scipy.linalg import lstsq

raw = FitSystem(dataset, representation="raw")
parameters = lstsq(raw.design_matrix, raw.forces)[0]
model = raw.force_constants(parameters)
print(raw.rmse(model))
```

这个直接调用没有内置 `raw.solve()` 的列归一化，秩亏或病态数据下结果可能不同。外部求解器的设置、收敛状态和缩放均由调用者负责。

`force_constants(parameters)` 不进行拟合，也不替外部求解器反归一化。输入应是长度为 `n_parameters` 的有限拟合坐标，按方程列顺序排列；如果外部求解的是缩放坐标，须先恢复到原列坐标。ASR 开启时再由该方法提升为物理参数。它也接受参数布局匹配、包含全部拟合阶数的 `ForceConstants`，输入要求与误差评估相同。

关闭 ASR 时，列按阶数、orbit 与分量的标准参数布局排列，可通过空间的参数切片和 `parameter_offsets` 解释。开启 ASR 时，列按各阶自由声学坐标顺序拼接，不能将 `model.parameters()` 的物理系数向量直接当成这些自由坐标。

对于 `normal`，外部求解器面对的是 $H\eta=g$。如果把 $(H,g)$ 再当作新的最小二乘输入，就改成了最小化正规方程残差，不能称为直接求解原始力残差问题。需要自定义原始最小二乘目标时，应保留 `raw` 方程。

## 8. 一个完整的 FC2+FC3 小例子

下面使用 EMT 生成 Al 的位移与力样本，联合拟合二阶和三阶。选择 `raw` 是为了展示不需要迭代参数的直接路线；将表示改为 `normal` 即可使用默认正规方程路线。

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from mlfcs import ClusterSpace, ClusterMap, ForceDataset, FitSystem

primitive = bulk("Al", "fcc", a=4.05)
space = ClusterSpace(
    primitive,
    cutoffs={2: 4.0, 3: 3.0},
    max_body_orders={2: 2, 3: 3},
    asr=True,
)
mapping = ClusterMap(space, space.primitive_atoms.repeat((3, 3, 3)))
mapping.rank_info().require_full()

structures = []
for seed in range(24):
    sample = mapping.supercell_atoms
    sample.rattle(stdev=0.01 + 0.01 * (seed % 3), seed=seed)
    sample.calc = EMT()
    forces = sample.get_forces()
    sample.calc = SinglePointCalculator(sample, forces=forces)
    structures.append(sample)

dataset = ForceDataset(mapping, structures)
system = FitSystem(dataset, representation="raw")
print("帧数、方程数、拟合坐标数：",
      system.n_structures, system.n_equations, system.n_parameters)
print("没有被观测的列：", system.unobserved_parameters)

model = system.solve()
print("包含的阶数：", model.orders)
print("训练力 RMSE：", system.rmse(model), "eV/Å")
print("训练相对误差：", system.relative_error(model))
model.save("al-fit.mlfcs")
```

这些截断、位移幅度和样本数只展示流程，不代表收敛设置。实际研究中，应比较训练集与独立验证集的误差，以及不同截断、阶数和采样范围下的物理结果。

带执行输出的教学例子见 [Ba₈Ga₁₆Ge₃₀ 联合拟合](notebooks/ba8ga16ge30.ipynb)和 [Si 高阶拟合](notebooks/si-fitting.ipynb)。有关模型与位移展开的推导见[从原子力恢复力常数](theory/reconstruction.md)。

开始拟合时，先确定模型和参考超胞，再把已经带力的构型收集为数据集。`FitSystem(dataset, representation=...)` 决定怎样保存与求解方程，`solve()` 给出力常数模型，误差方法帮助判断这份模型是否解释了数据。模型可保存并用于后续计算，训练质量仍需用独立样本与收敛比较确认。
