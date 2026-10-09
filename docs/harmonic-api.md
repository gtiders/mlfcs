# 谐波声子：从 FC2 到频率与声子谱

恢复二阶力常数之后，通常先看声子频率：参考结构是否稳定，声学支在 Γ 点附近是否合理，以及不同截断或拟合设置有没有改变色散关系。

`Harmonic` 将模型中的 FC2 转为质量加权动力学矩阵，再求出声子频率。同一个对象可以计算几个指定 q 点、规则网格，或带高对称点标签的能带路径。本章从已经得到的力常数模型开始，重点介绍如何选择 q 点、解释结果和绘制声子谱。

如果还没有 FC2，可以先阅读[有限差分](finite-difference-api.md)或[力拟合](fitting-api.md)。文末也给出一个从 EMT 力计算到声子谱的完整小例子。

## 1. 从已有模型构造谐波对象

```python
from mlfcs import ForceConstants
from mlfcs.phonon import Harmonic

model = ForceConstants.load("fit.mlfcs")
phonon = Harmonic(model)
print("参考胞质量：", phonon.masses)
```

构造参数只有 `model`。它必须是包含二阶系数的 `ForceConstants`；类型不对或没有 FC2 都会抛出 `ValueError`。模型可以同时包含 FC3、FC4 等阶数，但这里的计算只使用 FC2。

`phonon.model` 是关联的力常数模型，`phonon.masses` 是按模型参考胞原子顺序排列的只读质量数组，单位为原子质量单位。`Harmonic` 不需要另一个 mapping，也不接受超胞 FC2 数组或 `CompactForceConstants` 作为构造输入。

频率计算始终以模型的参考晶胞为基底。代码中沿用 `primitive` 命名，但输入参考胞不必是最小原胞。若使用较大的周期参考胞，其倒格基、布里渊区及声子支数也相对于这个胞定义，谱中会出现相应的能带折叠。

## 2. 先算几个 q 点

```python
gamma = phonon.frequencies([0.0, 0.0, 0.0])
frequencies = phonon.frequencies([
    [0.0, 0.0, 0.0],
    [0.25, 0.0, 0.0],
])
print(gamma)
print(frequencies.shape)
```

`frequencies(qpoints)` 的输入是模型参考胞倒格基中的**分数坐标**，不是以 Å⁻¹ 为单位的笛卡尔波矢。`[0, 0, 0]` 是 Γ 点；`[0.25, 0, 0]` 沿该基底的第一条倒格矢取四分之一。

一个 `(3,)` 坐标返回 `(3*N,)` 频率数组，其中 `N = model.cluster_space.n_atoms`。非空的 `(nq, 3)` 批量输入返回 `(nq, 3*N)`。例如 `[[0, 0, 0]]` 仍按批量处理，不会自动去掉第一维。输入形状不对、为空或包含非有限值时抛出 `ValueError`。

所有频率都是 **THz 普通频率**，已经包含角频率转换所需的 $1/(2\pi)$，不用再除一次。每个 q 点的模式按动力学矩阵特征值独立升序排列，不追踪相邻点之间的模态连接。

### 负频率与 Γ 点怎样看

负值表示动力学矩阵存在负特征值，也就是通常所说的虚频。程序用带负号的实数记录虚频幅度，不返回复数频率，也不将其裁为零。

小负值可能来自拟合或数值误差，明显负值也可能反映真实不稳定性。应比较量级、模型截断和数据质量，不能只因为曲线上出现负值就直接改变质量或抹去它。

`Harmonic` 不自动施加 ASR，也不强制将 Γ 点的三个平移模式置零。若希望模型满足平移不变性，应在拟合或有限差分之前使用 `ClusterSpace(..., asr=True)`。满足 ASR 的模型在 Γ 点应有三个接近零的刚性平移模式，仍可能保留浮点误差；不稳定结构还可能有其他负模，因此它们不一定是排序后的前三个频率。

## 3. 自动生成路径并绘制声子谱

```python
bands = phonon.run_band_path(npoints=200)
print("ASE 选取的路径：", bands.path)
```

`run_band_path(path=None, *, npoints=None)` 返回带绘图信息的 `HarmonicBandResult`。省略 `path` 时，ASE 根据参考胞的 Bravais 晶格选择默认高对称路径，并转换到实际输入晶胞的倒空间基底。

这个选择依据晶胞几何，不分析元素、原子排列或质量，也不等同于使用 seekpath 对完整结构进行标准化。自动路径适合快速查看谱；用于比较文献或不同结构时，应确认 `bands.path` 与预期路径、晶胞基底一致。

`npoints` 是整条路径的目标采样点数，默认 200，显式给出时必须是至少为 2 的整数，布尔值不合法。ASE 会保留必要顶点，因此较小目标可能被超过。它不是每一条直线段的点数。

### 指定标签与断开的路径

```python
# 对本章的 fcc Al 原胞，G、X、L 是 ASE 的特殊点标签。
bands = phonon.run_band_path("GX,LG", npoints=200)
```

字符串使用 ASE 为当前晶胞提供的特殊点标签；不是每种晶格都具有同一组名字。这里 `G` 表示 Γ，逗号表示路径在 X 与 L 之间断开，不沿 X→L 插值。

对已经采样的路径，结果包含以下公开属性。数组只读，可以直接用于绘图或另存：

| 属性 | 使用方式 |
|---|---|
| `path` | 实际路径字符串，便于确认自动选择或记录计算设置。 |
| `qpoints` | `(nq, 3)` 分数倒空间坐标，使用模型参考胞基底。 |
| `frequencies_thz` | `(nq, 3*N)` 带符号频率，负值表示虚频。 |
| `distances` | `(nq,)` 沿路径累计距离，单位 Å⁻¹，包含倒格矢的 $2\pi$ 因子。 |
| `tick_positions`、`tick_labels` | 横轴刻度位置与标签；`G` 显示为 Γ。 |
| `segments` | 每条连续直线段对应的 slice，包含两端顶点，用于分别绘图。 |

相邻线段共享转折点；断开路径两端没有新增跳跃距离。同一个横坐标上的标签可能合并为 `X|L`。绘图时应使用 `segments`，避免画出跨断点的连线：

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for segment in bands.segments:
    ax.plot(
        bands.distances[segment],
        bands.frequencies_thz[segment],
        color="C0",
        linewidth=1,
    )
for position in bands.tick_positions:
    ax.axvline(position, color="0.8", linewidth=0.5)
ax.axhline(0, color="0.5", linewidth=0.5)
ax.set_xticks(bands.tick_positions, bands.tick_labels)
ax.set_xlim(bands.distances[0], bands.distances[-1])
ax.set_ylabel("Frequency (THz)")
fig.tight_layout()
plt.show()
```

路径采样不会逐点包装回第一布里渊区，频率也没有跨点模态追踪。因此连线表示每点按特征值排序的结果，不能据此推断同一振动模在交叉处的连续身份。

### 复用或自定义 ASE 路径

比较两个模型时，可以先准备一份 ASE `BandPath`，然后交给各自的谐波对象，确保 q 点完全相同：

```python
path = model.cluster_space.primitive_atoms.cell.bandpath(
    "GAB",
    special_points={
        "G": [0, 0, 0],
        "A": [0.25, 0, 0],
        "B": [0.25, 0.25, 0],
    },
    npoints=150,
)
bands = phonon.run_band_path(path)
```

这里 A、B 是自行定义的点，并不声称它们是晶体的标准高对称点。传入 `BandPath` 时必须已经采样，并采用与模型相同的晶胞基底；不能同时再传 `npoints`，应先在 ASE 中修改采样。

晶胞不匹配、空 q 点、缺少按顺序出现的标签顶点，或连续路径少于两个顶点时会报错。`path` 不是字符串、`BandPath` 或 `None` 时抛出 `TypeError`。`run_band_path()` 返回数据，不自动绘图或写文件；保存图可在绘图后调用 `fig.savefig(...)`。

## 4. 用网格查看整个布里渊区

能带路径只覆盖一些线段。希望检查更广的 q 点，或准备布里渊区平均所需的频率与权重时，可以计算网格：

```python
result = phonon.mesh((8, 8, 8), time_reversal=True)
print("不可约点数：", len(result.qpoints))
print("完整点数：", result.weights.sum())
```

`mesh(mesh, *, time_reversal=True)` 接受三种网格输入：正整数三元组 `(nx, ny, nz)`、非奇异整数 `(3, 3)` 矩阵，或已经构造的 `QGrid`。三元组相当于对角矩阵；一般矩阵使用 `cell_super = matrix @ cell_reference` 的行晶格约定，其完整点数为行列式绝对值。这只是指定倒空间网格，不需要另建原子超胞或 `ClusterMap`。

网格包含 Γ 点，目前没有偏移参数。非法尺寸、奇异或非整数矩阵会被拒绝；不支持的整数范围会显式失败。

默认将时间反演相关的 q 与 -q 归入同一组；`time_reversal=False` 关闭这项归并，仍使用网格兼容的空间群操作。约化只采用保持当前原子质量分配的操作，因此不同同位素质量可能减少可用对称性。

返回的 `HarmonicMeshResult` 保存不可约点结果：

| 属性或方法 | 含义 |
|---|---|
| `qpoints` | 不可约代表点的分数倒空间坐标。 |
| `frequencies_thz` | `(不可约点数, 3*N)` 带符号 THz 频率。 |
| `weights` | 每个代表点覆盖的完整网格点数，是整数而非归一化概率；总和等于完整点数。 |
| `mesh_matrix` | 定义网格的整数行晶格矩阵。 |
| `full_frequencies()` | 无参数，返回新建的 `(完整点数, 3*N)` 频率数组。 |

只在不可约代表点计算频率，完整频率按需展开。完整坐标与展开结果按 `QGrid` 整数标签的字典序对齐：

```python
from mlfcs.phonon import QGrid

full_qpoints = QGrid(result.mesh_matrix).points
full_frequencies = result.full_frequencies()
```

结果中的数组只读。这里只得到频率与权重，不自动生成声子态密度或热力学量；后续平均应使用 `weights / weights.sum()` 作为各代表点的权重。网格与对称等价组的推导见[倒空间对称性](theory/reciprocal-symmetry.md)。

## 5. 需要动力学矩阵时

如果希望自己做特征分解或分析振动本征向量，可以直接读取动力学矩阵：

```python
import numpy as np

matrix = phonon.dynamical_matrices([0.25, 0.0, 0.0])
eigenvalues, eigenvectors = np.linalg.eigh(matrix)
```

`dynamical_matrices(qpoints)` 接受与 `frequencies()` 相同的坐标输入，形状检查也相同。单点返回 `(3*N, 3*N)`，批量返回 `(nq, 3*N, 3*N)` 的复 Hermitian 矩阵；原子与 x/y/z 分量按原子优先顺序排列。

矩阵已经质量加权，单位为 eV/(Å²·原子质量单位)，尚未转换为 THz。上面得到的本征向量属于质量加权坐标，不能直接当作未加权的原子位移；解释振幅时还需相应质量因子。

MLFCS 使用包含原子基元位置的 Fourier 相位，即位置规范（positional gauge）：

$$
D_{i\alpha,j\beta}(q)
=\frac{1}{\sqrt{m_i m_j}}
\sum_R \Phi_{i\alpha,j\beta}(R)
\exp\!\left[2\pi i q\cdot(R+s_j-s_i)\right].
$$

因此，与采用不同相位规范的软件比较时，矩阵和本征向量可能带不同相位，频率则不受这种规范选择影响。计算末尾的 Hermitian 对称化用于清理数值舍入，不应视为修复错误力常数或自动施加物理约束。

高级调用也可以将 `QStars` 传给 `frequencies()` 或 `dynamical_matrices()`，此时只计算不可约代表点，返回批量结果。用户须保证 stars 与模型的晶胞、位点顺序和质量保持对称性配套，程序不检查外部 stars 的兼容性。通常使用 `mesh()` 自动准备即可；矩阵在等价点上的恢复还涉及旋转、位点置换与相位，不能仅按标量权重复制。

## 6. 改变质量而不重新拟合

力常数描述能量对位移的导数，质量只在动力学矩阵中进入。若只研究同位素质量变化，可以保留原来的系数：

```python
heavy_space = model.cluster_space.with_masses(2 * model.cluster_space.masses)
heavy_model = ForceConstants(heavy_space, model.coefficients)
heavy_phonon = Harmonic(heavy_model)
```

`with_masses(masses)` 要求按参考胞顺序提供有限、严格为正的 `(N,)` 数组。它返回新空间，不改变原模型的质量、几何和参数布局。`Harmonic.masses` 没有 setter；更换质量后应构造新模型与新谐波对象。

全部质量乘以 $a>0$ 时，频率除以 $\sqrt a$。逐位点质量变化则可能同时改变频率与网格约化所使用的对称性。原生力常数文件会保存质量，加载后无需再手动附加默认元素质量。

## 7. 极性材料与 NAC 的边界

`Harmonic` 的所有接口都只计算给定参数化 FC2 的 Fourier 响应，不自动叠加长程模型，也没有 Born 电荷、介电张量或 NAC 的构造参数。

特别是，扣除长程力后拟合出的短程 FC2，直接传给 `Harmonic` 得到的仍是短程响应。不能将它解释为已经含 LO–TO 分裂的总声子谱。`DipoleEwald` 输出的有限超胞折叠 FC2 也不是任意 q 的解析长程模型，不能直接替换这里的输入。

当前极性材料的总 FC2 导出与 Phonopy NAC 流程见 [NaCl 教程](notebooks/nacl-long-range.ipynb)。长程项扣除、回加和 NAC 需要一致的物理分解，应按该流程处理，避免重复计入长程贡献。

## 8. 一个从 FC2 到声子谱的完整例子

下面使用 EMT 的有限差分结果绘图。它展示接口衔接，不代表示例截断、步长和超胞已经完成物理收敛检查。

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
import matplotlib.pyplot as plt
from mlfcs import ClusterSpace, ClusterMap, FiniteDifference, ForceDataset
from mlfcs.phonon import Harmonic

primitive = bulk("Al", "fcc", a=4.05)
space = ClusterSpace(primitive, cutoffs={2: 4.0}, asr=True)
mapping = ClusterMap(space, space.primitive_atoms.repeat((3, 3, 3)))
fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))

calculated = []
for sample in fd.sow():
    sample.calc = EMT()
    forces = sample.get_forces()
    sample.calc = SinglePointCalculator(sample, forces=forces)
    calculated.append(sample)

model = fd.reap(ForceDataset(mapping, calculated))
phonon = Harmonic(model)
bands = phonon.run_band_path(npoints=200)
print("路径：", bands.path)
print("Γ 点频率：", phonon.frequencies([0, 0, 0]))

fig, ax = plt.subplots()
for segment in bands.segments:
    ax.plot(bands.distances[segment], bands.frequencies_thz[segment], color="C0")
ax.axhline(0, color="0.5", linewidth=0.5)
ax.set_xticks(bands.tick_positions, bands.tick_labels)
ax.set_xlim(bands.distances[0], bands.distances[-1])
ax.set_ylabel("Frequency (THz)")
fig.tight_layout()
plt.show()
```

已有拟合模型时，直接将其中包含 FC2 的模型交给 `Harmonic`，无需重复力计算。带执行结果的声子谱例子见 [Ba₈Ga₁₆Ge₃₀ 教程](notebooks/ba8ga16ge30.ipynb)与 [Si 有限差分教程](notebooks/si-finite-difference.ipynb)。

日常使用可以从几个 q 点检查开始，再用 `run_band_path()` 画谱、用 `mesh()` 检查网格。质量和坐标基底来自模型，虚频保留在结果中；ASR、长程分解与 NAC 则需要在各自的步骤中明确处理。
