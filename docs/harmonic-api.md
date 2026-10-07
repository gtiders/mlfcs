# 谐波频率与原子质量

## 查看或替换质量

`ClusterSpace` 从输入 ASE `Atoms` 复制逐站点质量，单位为原子质量单位。`cs.masses` 是形状为 `(cs.n_atoms,)` 的只读数组。修改输入 atoms 或 `cs.primitive_atoms` 返回的独立副本，不会改变 cs。

```python
from mlfcs import ForceConstants
from mlfcs.phonon import Harmonic

heavy_cs = model.cluster_space.with_masses(2 * model.cluster_space.masses)
heavy_model = ForceConstants(heavy_cs, model.coefficients)
heavy_harmonic = Harmonic(heavy_model)
```

`with_masses(masses)` 要求形状为 `(n_atoms,)`，质量有限且严格为正。新 cs 只复制质量，共享原有的只读几何、对称性、orbit 和参数化数据，不重复运行 spglib、cluster enumeration 或整数核构造。原 cs 和模型保持不变。

质量不改变力常数物理参数布局，因此能量导数参数可以直接绑定到新 cs，无须重新拟合。原生 HDF5 力常数文件会保存质量。`Harmonic.masses` 是无 setter 的只读质量快照；换质量应创建新 cs 和新谐波对象，不能原地修改缓存中的质量权重。

## 指定 q 点计算

```python
harmonic = Harmonic(model)  # model 必须包含 FC2
one = harmonic.frequencies([0.25, 0.0, 0.0])
many = harmonic.frequencies([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
matrices = harmonic.dynamical_matrices([[0.25, 0.0, 0.0]])
```

q 使用模型原胞基底中的分数倒空间坐标。单点 `(3,)` 返回 `(3*n_atoms,)` 频率数组；非空批量 `(nq, 3)` 返回 `(nq, 3*n_atoms)`。每个点的模式按动力学矩阵特征值升序排列，不做跨 q 点的能带连接。

频率固定为 **THz 普通频率**，包含从角频率转换所需的 $1/(2\pi)$。负值表示虚频，即 $-\sqrt{|\lambda|}$ 乘以转换因子，不会裁剪为零。全部质量乘以 $a>0$，频率除以 $\sqrt a$，FC 参数保持不变。

`dynamical_matrices(qpoints)` 返回 positional Fourier gauge 下的复 Hermitian 矩阵。单点形状为 `(3*n_atoms, 3*n_atoms)`，批量为 `(nq, 3*n_atoms, 3*n_atoms)`；矩阵元素单位为 eV/(angstrom²·原子质量单位)，尚未转换为 THz。原名 `matrices()` 已删除。

## 自动计算声子路径

```python
bands = harmonic.run_band_path(npoints=200)
print(bands.path)  # 查看 ASE 实际选取的路径

# 指定路径；逗号表示断开。
bands = harmonic.run_band_path("GX,LG", npoints=200)
```

自动路径由 ASE 根据模型原胞晶胞的 Bravais 晶格生成，并转换到实际原胞倒空间基底。
它依据晶胞几何，不分析元素、原子排列或质量，不宣称是完整原子结构唯一的高对称路径。
`npoints` 是整条路径的目标采样点数，默认 200；ASE 会保留必要顶点，因此很小的目标值可能被超过。

结果 `HarmonicBandResult` 包含：

| 属性 | 含义 |
|---|---|
| `path` | 实际路径字符串 |
| `qpoints` | 模型原胞中的分数倒空间坐标 |
| `frequencies_thz` | 形状 `(nq, 3*n_atoms)` 的带符号 THz 频率 |
| `distances` | 沿路径累计距离，单位 Å⁻¹，包含倒格矢的 $2\pi$ 因子 |
| `tick_positions`、`tick_labels` | 绘图刻度位置与名称；`G` 显示为 Γ |
| `segments` | 每条直线段的 slice，包含两个端点 |

相邻线段共享转折点。逗号两侧不增加跳跃距离，也没有跨断点的 segment；
同一横坐标上的不同端点名称合并为 `X|L` 这类标签。路径不逐点 wrap，不追踪模式连接。

```python
import matplotlib.pyplot as plt

for segment in bands.segments:
    plt.plot(bands.distances[segment], bands.frequencies_thz[segment], color="C0")
plt.xticks(bands.tick_positions, bands.tick_labels)
plt.ylabel("Frequency (THz)")
plt.show()
```

自定义顶点或复用同一采样路径时，先用 ASE 生成 `BandPath`：

```python
path = model.cluster_space.primitive_atoms.cell.bandpath(
    "GAB",
    special_points={"G": [0, 0, 0], "A": [0.25, 0, 0], "B": [0.25, 0.25, 0]},
    npoints=150,
)
bands = harmonic.run_band_path(path)
```

传入 `BandPath` 时必须采用模型原胞的同一晶胞基底，且不能再传 `npoints`；
没有采样点或缺少有序顶点的路径会报错。函数不绘图、不写文件、不保存隐式结果缓存。
本接口尚不支持 NAC；极性材料的外部 NAC 流程见 [NaCl 教程](notebooks/nacl-long-range.ipynb)。

## 直接计算网格

```python
result = harmonic.mesh((8, 8, 8), time_reversal=True)
result.qpoints            # 不可约代表点，分数倒空间坐标
result.weights            # 每个 star 的整数成员数，不是归一化概率
result.frequencies_thz    # 代表点上的带符号 THz 频率
result.mesh_matrix        # 精确整数行约定超胞矩阵
full = result.full_frequencies()
```

`mesh` 接受正整数三元组 `(nx, ny, nz)`、非奇异整数 3×3 超胞矩阵，或现有 `QGrid`。矩阵采用 `supercell_cell = matrix @ primitive_cell`；三元组转换为对角矩阵。这是包含 Gamma 的精确网格，目前不提供偏移网格。

`HarmonicMeshResult` 中的数组只读。权重之和等于完整网格大小，代表点频率形状为 `(n_representatives, 3*n_atoms)`。只在代表点上计算 Fourier 矩阵；`full_frequencies()` 按需创建完整频率数组，顺序为 `QGrid` 精确整数 label 的字典序。对应的完整坐标是 `QGrid(result.mesh_matrix).points`。

显式 q 点允许结构等价站点使用不同质量。网格只采用保持当前逐站点质量的结构对称操作，并按请求加入时间反演；不会改变 cs 的结构对称性或 FC basis。SCPH 使用相同规则。

高级调用仍可向 `frequencies()` 或 `dynamical_matrices()` 传入 `QStars`；用户负责保证其操作属于模型原胞对称性并保持质量，程序不检查外部 stars 与模型的配套性。`Harmonic.mesh()` 自动构造适当的质量保持子群。

计算继续保持三维全周期；本次 API 修改不涉及部分 PBC 或真空检查。
