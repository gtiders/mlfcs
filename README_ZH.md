# MLFCS

[English](README.md) | 中文

MLFCS 是一个以 ASE 为公共接口、根据原子力计算对称性约化力常数的 Python 库。
二阶及任意更高阶力常数使用同一套阶数参数化流程，其中三阶和四阶是当前主要生产
验证路径。五阶及更高阶也可以直接计算，并通过通用稀疏 HDF5 格式导出；实际可算
规模取决于团簇数量、近邻截断、超胞大小和可用内存。

有限差分路径只使用 ASE、NumPy 和 SciPy，因此外部计算器可完全自行决定使用 CPU 或
GPU。联合拟合同时支持 CPU 和 GPU：安装 CUDA 版 JAX 后，仅将计算密集的 Wick 特征核
放到 GPU；几何、对称性、稀疏约束和最终求解仍在宿主端。实现通过对称性约化、连续稀疏
数组、惰性稠密物化、约束零空间坐标和有界特征 tile 控制内存；每次拟合只准备一次静态
JAX 缓冲区与已编译核，训练、验证和诊断均复用它们。

基础包不规定力由什么程序产生。用户可以使用任意 ASE Calculator，也可以把位移结构
交给外部任务系统计算。原生 SSCHA 模块结合 q 空间量子谐波采样与 MLFCS Gram 拟合器，
计算温度相关的有效二阶力常数。

MLFCS 只提供 Python API，不提供 CLI。

> **开发分支说明：** 下文的 FC2--FCn 联合仅力数据拟合 API 当前在 `dev` 分支开发。
> 稳定的 `main` 分支保留有限差分 API 和公共力常数 IO；拟合验证达到发布条件后再移除
> 此标注并合入稳定分支。

## 基本原理

`n` 阶力常数是势能对原子位移的 `n` 阶导数，也等价于力的 `n-1` 阶位移导数。
MLFCS 的计算流程为：

```text
ASE 原胞结构
    │
    ▼
确定性超胞和近邻截断
    │
    ▼
空间群与指标置换约化后的团簇轨道
    │
    ▼
递归中心有限差分位移计划
    │
    ▼
用户提供的原子力
    │
    ▼
稀疏对称性重建和可选求和规则投影
    │
    ▼
ForceConstants → HDF5 / ShengBTE / phonopy 兼容格式
```

空间群对称性、力常数指标置换和团簇稳定子约束会把每个团簇张量约化为独立分量，
有限差分只采样这些必要分量。重建结果默认保持稀疏，只有用户明确请求时才物化为
完整稠密张量。

声学求和规则（ASR）作为不可约轨道参数空间中的受约束投影施加：

```text
对一个原子指标求和：Σ Φ(i1, ..., in) = 0
```

所有约束系统统一使用稀疏、矩阵无关的 LSMR 投影。二阶计算还可主动开启可选的
Born–Huang 旋转求和规则，详见[求和规则](docs/zh/methods/sum-rules.md)。

## 主要特点

- `order>=2` 使用同一个 API 和数值流程；
- 三阶和四阶力常数经过生产级测试；
- 四阶通过独立 JAX 微分的 FCC Morse 解析能量验证，并检查有限差分的二阶步长收敛；
- 二阶和五阶完成端到端验证；
- 公共结构与计算器接口完全采用 ASE；
- 支持适合外部调度和断点续算的 `sow()` / `reap()`；
- 直接 calculator 路径支持可配置偶次幂阶数的零步长外推；
- 稳定的构型 ID 和显式原子顺序映射；
- 与既有逻辑兼容的周期团簇截断；
- 递归中心有限差分模板；
- 拟合支持 CPU/GPU 的常驻 JAX Wick 特征核；有限差分保持宿主端执行；
- JAX JIT、`vmap` 和批量张量收缩降低高阶拟合特征计算开销；
- 连续稀疏数组、矩阵无关变换和惰性稠密物化降低峰值内存；
- 位移键去重减少需要调用势函数的构型数量；
- 使用 Wick 正交特征联合拟合 FC2--FCn，并输出兼容通用格式的 Taylor 力常数；
- 稀疏矩阵无关 LSMR 实现严格平移 ASR；
- 二阶力常数可选 Born–Huang 旋转求和规则；
- 任意阶通用稀疏 HDF5；
- 三阶、四阶 ShengBTE 输出；
- 二阶完整 phonopy 文本输出；
- 原生量子/经典 SSCHA，并支持任意 ASE Calculator。

## 安装

MLFCS 要求 Python 3.12 或更高版本，使用 uv 安装和运行：

```bash
uv sync
```

[`examples/`](examples/) 提供可执行的 API 示例：

- [`basic_fc2.py`](examples/basic_fc2.py) 使用 ASE 自带 EMT calculator 直接计算 FC2；
- [`vasp_external_fc3.py`](examples/vasp_external_fc3.py) 给出完整的外部 VASP
  `sow`、力收集和 `reap` 工作流；
- [`nep89_orders.py`](examples/nep89_orders.py) 通过 calorine ASE calculator，使用用户提供的
  NEP89 模型计算一个或多个阶数。

calorine、MACE 等势函数包不是基础依赖，请根据实际任务单独安装。

## 单位

| 物理量 | 单位 |
|---|---|
| 晶胞、坐标和位移 | Å |
| 力 | eV/Å |
| `n` 阶力常数 | eV/Åⁿ |
| 正数截断 | Å |
| 负整数截断 | 近邻壳层 |
| `None` 截断 | 当前超胞可枚举的最大半径 |

JAX 数值核使用 64 位浮点数。

## 先确定后处理结构

计算力常数之前，应先确定最终使用哪个程序进行后处理，并优先采用该程序生成或验证的
原胞与参考超胞。建议把这两个结构以相同的 ASE `Atoms` 形式直接交给 MLFCS 计算。
MLFCS 能验证和转换原子重排、整数换基等严格等价的原胞/超胞表示，但从后处理软件自身
认可的结构开始，可以避免在最终导出时引入不必要的原子映射和周期平移歧义。

## 快速开始

### 外部力计算

`sow()` 本身不写文件，也不启动 DFT 程序；它返回一个有确定顺序的 ASE `Atoms` 位移
结构列表。用户可以自行用 ASE 将这些结构写成 VASP 的 `POSCAR-xxx`，也可以选择其他
第一性原理软件的输入格式，再通过任意本地调度系统提交。计算完成后，仍由用户使用 ASE
读取每个结果、提取力，并恢复 `sow()` 的原始顺序（或按构型 ID 建立字典），最后只把
力交给 `reap()`。只要能够保证这个位置顺序，就不需要构型 ID。

下面是按位置顺序组织的 VASP 案例：

```python
from pathlib import Path

import numpy as np
from ase.io import read, write
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("POSCAR"),
    order=3,
    reference=read("reference-supercell.vasp"),
    cutoff=-5,
    displacement=0.01,
    symprec=1e-5,
)

# 1. sow()：获得与 reap 完全对应的 ASE 位移结构列表。
structures = calculation.sow()
Path("vasp-jobs").mkdir(exist_ok=True)
for configuration_id, atoms in enumerate(structures):
    job = Path("vasp-jobs") / f"POSCAR-{configuration_id + 1:03d}"
    job.mkdir(exist_ok=True)
    write(job / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)

# 2. 用户提供 INCAR、KPOINTS、POTCAR，并提交所有目录。
#    MLFCS 不启动 VASP，也不规定 VASP 参数。

# 3. 计算完成后，按照相同文件名和顺序用 ASE 读取 vasprun.xml。
forces = []
for configuration_id in range(len(structures)):
    job = Path("vasp-jobs") / f"POSCAR-{configuration_id + 1:03d}"
    completed = read(job / "vasprun.xml", index=-1)
    forces.append(completed.get_forces())
forces = np.asarray(forces)

# 4. reap()：forces[i] 必须对应 structures[i]。
fc3 = calculation.reap(forces, acoustic_sum_rule=True)
fc3.write("fc3.h5", format="hdf5")
```

若使用 Quantum ESPRESSO、ABINIT、CP2K 或其他外部程序，只需更换 ASE `write()` 和
`read()` 所使用的格式，并补充该程序需要的输入参数；`sow/reap` 契约不变。若文件名和
返回的力严格保持 sow 顺序，位置式 `reap()` 不需要任何额外元数据。POSCAR 这类文件
不会保存 Python 中的 `atoms.info`，因此对于任务乱序、断点续算、长期归档或防止混入
其他数据集的情况，建议用 manifest 保存文件名—构型 ID 对应关系。完整的
[`vasp_external_fc3.py`](examples/vasp_external_fc3.py) 示例把 manifest 作为可选安全层，
并实现力收集、缺失结果检查和最终导出，详见
[外部 VASP 工作流](docs/zh/workflows/external-calculators.md)。

力数组的形状必须为：

```text
(len(calculation.sow()), len(calculation.supercell), 3)
```

当位移结构仍是 ASE 对象时，它还包含以下可选审计元数据：

```python
atoms.info["mlfcs_configuration_id"]
atoms.info["mlfcs_atom_order"]
atoms.arrays["mlfcs_displacement"]
```

如果外部任务乱序返回，可以按任意顺序读取，但应传入以原始零基构型 ID 为键的字典：

```python
fc3 = calculation.reap(forces_by_configuration_id)
```

缺少或多出 ID、形状错误以及非有限数值都会被拒绝。

### 直接使用 ASE Calculator

```python
calculator = make_my_ase_calculator()

fc3 = calculation.run(
    calculator,
    progress=lambda done, total: print(f"{done}/{total}"),
)
```

Calculator 默认串行执行，避免同时复制多个大型机器学习势导致内存峰值。需要外部
并行或断点续算时，应使用 `sow()` / `reap()`。

直接 ASE calculator 路径可以选择将指定阶导数外推到零位移：

```python
fc4 = calculation.run(
    calculator,
    derivative_backend="extrapolate",
    extrapolation_spacing=0.005,
    extrapolation_side_steps=2,
    extrapolation_degree=1,
)
```

中心位移为 `0.03` Å 时，上例采样 `0.02`、`0.025`、`0.03`、`0.035` 和 `0.04` Å。
默认阶数 `1` 拟合 `D(h) = D0 + c2 h²`；更高阶数继续加入偶次幂。该后端有意只通过
`run()` 开放，不属于外部 `sow()` / `reap()` 工作流。详见
[零步长外推](docs/zh/workflows/extrapolation.md)。

显式保存力：

```python
forces = calculation.evaluate(calculator)
np.savez_compressed("forces.npz", forces=forces)
fc3 = calculation.reap(forces)
```

`sow()`、`reap()` 和直接 ASE Calculator 运行默认开启阶段信息，输出对称性、不可约
团簇、位移计划、ASR 和力计算进度。这些信息只使用计算过程中已经得到的数据，不会
重复执行昂贵分析。传入 `verbose=False` 可关闭全部阶段信息和截断输出；
`report_cutoff=False` 只关闭详细的近邻壳层两行。

## 阶数和近邻截断

所有阶数使用同一个构造接口：

```python
fc2_calculation = ForceConstantCalculation(atoms, order=2, cutoff=-6)
fc4_calculation = ForceConstantCalculation(atoms, order=4, cutoff=-3)
fc5_calculation = ForceConstantCalculation(atoms, order=5, cutoff=-1)
full_calculation = ForceConstantCalculation(atoms, order=3, cutoff=None)
```

正数截断表示 Å 半径；负整数表示从 1 开始编号的近邻壳层。例如 `cutoff=-8` 表示
第八近邻。`cutoff=None` 表示使用当前有限超胞可枚举的最大半径，并不表示无穷大的
相互作用范围。程序同时打印超胞容量和本次选择的实际半径：

```text
Supercell neighbor limit: maximum shell = 33, maximum cutoff radius = 15.7504983443 Å
Selected neighbor cutoff: shell = 8, cutoff radius = 7.5419604204 Å
```

第一行是有限超胞的容量诊断；第二行才是计算实际使用的截断。请求超过可枚举容量时
会报错。传入 `report_cutoff=False` 可以关闭两行输出。

高阶计算的团簇组合、张量分量、置换和有限差分符号都会快速增长。建议从较小近邻
开始，并监控构型数量和内存。

## 原子顺序与结构参考系

用户提供的 reference 超胞原子顺序是唯一权威。`sow()` 返回该顺序，传入 `reap()` 的
每组力也必须保持该顺序；MLFCS 不再提供 internal/grouped 模式，更不会静默重排外部力或
拟合快照。计算 API 只接受显式 reference 结构：

```python
calculation = ForceConstantCalculation(primitive, reference=reference_supercell, order=3)
```

若需显式构造这个 reference，可使用稳定的公开函数：

```python
from mlfcs import build_supercell

reference_supercell = build_supercell(primitive, [[2, 1, 0], [0, 2, 0], [0, 0, 1]])
```

该函数固定采用 phonopy old-style 的原子排列；MLFCS 的超胞矩阵约定保持不变。显式提供的
`reference_supercell` 不会被重排。

phonopy 等格式所需的排序仅在导出边界生成。
对于独立产生且已重排的快照，可显式调用 `mlfcs.align_structures(reference, atoms)`；它返回
对齐后的结构和最大周期匹配残差。

## 声学求和规则

ASR 默认启用：

```python
constrained = calculation.reap(forces, acoustic_sum_rule=True)
raw = calculation.reap(forces, acoustic_sum_rule=False)
```

受约束结果是在独立参数空间中距离测量结果最近、同时满足平移不变性的解。置换对称性
会给出其他原子轴上的等价约束。

Born-Huang 与 Huang 条件是独立的 FC2 后处理，有限差分和拟合结果共用同一入口。默认
`strength=1.0` 为严格模式，FC3 及更高阶不会被改动：

```python
constrained = result.enforce_rotational_sum_rules(
    born_huang=True,
    huang=True,
)
```

投影器始终满足 FC2 的 ASR，使用全部简并最近周期像，并报告残差与修正量。`[0, 1]`
中的 `strength` 只缩放 Born-Huang/Huang 修正。详见[求和规则](docs/zh/methods/sum-rules.md)。

## 输出格式

输出格式必须显式指定：

```python
fc2.write("FORCE_CONSTANTS", format="phonopy")
fc2.write("fc2.hdf5", format="phonopy_hdf5")
fc3.write("fc3.h5", format="hdf5")
fc3.write("fc3.hdf5", format="phono3py_hdf5")
fc3.write("FORCE_CONSTANTS_3RD", format="shengbte")
fc4.write("FORCE_CONSTANTS_4TH", format="shengbte")
fc234.write("force_constants.xml", format="alamode")
```

原生稀疏 HDF5 可通过对应的公开 API 读取：

```python
from mlfcs import read_hdf5

fc234 = read_hdf5("fc3.h5")
```

| 格式 | 阶数 | 表示 |
|---|---|---|
| `hdf5` | 任意阶 | 原生 v2 晶格标记稀疏 IFC（`sites`、平移代表与笛卡尔张量） |
| `shengbte` | 3、4 | 对称闭合、基于晶格平移的文本块 |
| `phonopy` | 2 | 完整稠密超胞 FC2 文本 |
| `phonopy_hdf5` | 2 | phonopy 兼容的完整超胞 `force_constants` HDF5 |
| `phono3py_hdf5` | 3 | phono3py 兼容的完整超胞 `fc3` HDF5 |
| `alamode` | 2--4 | 合并 FC2--FC4 的 ALAMODE FCSXML 文档 |

ShengBTE 严格写出重建后稀疏结果携带的对称闭合团簇支撑域，并把晶格 residue 解析为
联合相容的最近周期像。

phonopy 和 phono3py HDF5 保持显式 reference 超胞顺序，并逐个第一原子 slab
流式写入，因此不会在内存中构造完整 FC3。原生 `hdf5` 使用 v2 schema，保存 primitive、
reference、经验证映射和晶格标记稀疏 IFC；旧原生 schema 被明确拒绝。

ALAMODE XML 严格保留 `fc.supercell` 当前的原子顺序。原胞原子身份和晶格平移映射仅取自
MLFCS 的 `primitive_index` 与 `cell_translation` 元数据；导出阶段不会让 spglib 或
ALAMODE 重新识别、重排晶胞。传入 `order=2`、`3` 或 `4` 可只写指定阶，省略则把当前
可用的 FC2--FC4 合并到一个 XML。完整映射与周期像约定见
[ALAMODE XML 指南](docs/zh/formats/alamode.md)。

高阶结果推荐使用稀疏 HDF5。显式稠密化超过默认 2 GB 建议预算时会发出警告：

```python
fc5.write("fc5.h5", format="hdf5")
dense = fc5.materialize(5)
dense = fc5.materialize(5, max_bytes=None)  # 显式关闭警告预算
```

## 原生 SSCHA

独立的 `mlfcs.anharmonic.sscha` 模块根据热位移上的原子力拟合温度相关有效 FC2：

```python
from mlfcs.anharmonic.sscha import SSCHA

sscha = SSCHA(
    atoms,
    reference=read("reference-supercell.vasp"),
    temperature=300,
    statistics="quantum",
    snapshots=1000,
    max_iterations=10,
    random_seed=42,
)

sscha.run(calculator)
sscha.use_average(last=5)
sscha.write("fc2-300K.hdf5", format="hdf5")
```

第零轮通过小随机笛卡尔位移拟合初始 FC2。后续每轮直接在 compact FC2 的 q 空间采样
谐振子正则系综，再由原生流式 Gram 拟合器重拟合 FC2。因此 `max_iterations=10` 表示
一次初始化加十次更新。

也支持逐轮外部计算：

```python
structures = sscha.sow()
result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

拟合 FC2 只需要力；只有估计自由能时才需要能量。已完成轮次和采样诊断保存在
`sscha.history`。phonopy 兼容文本和 HDF5 由公共 MLFCS writer 写出，运行时无需 phonopy。
正则系综轮次还会报告 FC2 相对更新幅度，试探采样哈密顿量保持为内部细节。

该方法是随机有效谐波方法，不是显式 FC3 bubble 或 FC4 loop 计算。详见
[SSCHA 文档](docs/zh/workflows/sscha.md)。

## 当前限制

- 三阶和四阶是主要生产测试路径；
- 更高阶沿用同一实现，但计算成本可能无法承受；
- ShengBTE 输出仅支持三阶和四阶；
- 尚未实现显式 FC3 bubble 和 FC4 loop 自能；
- 基础重建流程不包含非解析长程静电修正；
- SSCHA 收敛由调用者控制，不内置通用自动停止阈值。

## 文档

完整双语文档：[English site](https://gtiders.github.io/mlfcs/) 和 [中文站点](https://gtiders.github.io/mlfcs/zh/)。参见[可运行案例](examples/README_ZH.md)和[开发与验证](docs/zh/development/validation.md)

## 开发与测试

所有命令使用 uv，测试串行运行：

```bash
uv sync
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

本地测试只包含确定性的单元和公共 API 回归。材料比较和第三方输运工作流记录在
`examples/`，需要时手动运行。CI 只构建双语文档站。测试组织见 [tests/README.md](tests/README.md)。

当前开发预发布版本为 `4.0.0a2`（4.0 alpha 2）。版本变化见 [CHANGELOG_ZH.md](CHANGELOG_ZH.md)，开发流程见
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MLFCS 使用 [GNU 通用公共许可证第 3 版或更高版本](LICENSE)发布。改编的第三方组件和
ALAMODE 适配器的第三方来源与许可条款直接保留在其源码模块顶部。
