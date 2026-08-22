# MLFCS

[English](README.md) | 中文

MLFCS 是一个以 ASE 为公共接口、根据原子力计算对称性约化力常数的 Python 库。
二阶及任意更高阶力常数使用同一套阶数参数化流程，其中三阶和四阶是当前主要生产
验证路径。五阶及更高阶也可以直接计算，并通过通用稀疏 HDF5 格式导出；实际可算
规模取决于团簇数量、近邻截断、超胞大小和可用内存。

MLFCS 的三阶技术路线明确参考并借鉴了 GPL 许可的
[thirdorder](https://gitlab.com/sousaw/thirdorder) 项目，包括其算法、周期镜像约定和
`sow`/`reap` 工作流。MLFCS 在此基础上发展出阶数参数化的 ASE/JAX 架构，并新增稀疏
表示、约束求解、加速和互操作层。具体归属和范围见[第三方来源说明](THIRD_PARTY_ZH.md)。

数值计算同时支持 CPU 和 GPU：CPU 模式适合常规计算和大规模稀疏线性代数，安装
CUDA 版 JAX 后可将高阶笛卡尔张量旋转与批处理变换放到 GPU。实现通过对称性约化、
连续稀疏数组、惰性稠密物化、矩阵无关张量操作、小 Gram 零空间和稀疏 LSMR 控制
内存；通过 JAX JIT、`vmap`、批量张量收缩和位移任务去重优化速度。具体收益取决于
体系、阶数和硬件，GPU 不会替代仍在 CPU 上执行的团簇枚举与稀疏求解。

基础包不规定力由什么程序产生。用户可以使用任意 ASE Calculator，也可以把位移结构
交给外部任务系统计算。独立的可选模块使用 phonopy 和 symfc，通过随机自洽谐波近似
（SSCHA）计算温度相关的有效二阶力常数。

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
ForceConstants → HDF5 / NumPy / ShengBTE / phonopy
```

空间群对称性、力常数指标置换和团簇稳定子约束会把每个团簇张量约化为独立分量，
有限差分只采样这些必要分量。重建结果默认保持稀疏，只有用户明确请求时才物化为
完整稠密张量。

声学求和规则（ASR）作为不可约轨道参数空间中的受约束投影施加：

```text
对一个原子指标求和：Σ Φ(i1, ..., in) = 0
```

所有约束系统统一使用稀疏、矩阵无关的 LSMR 投影。二阶计算还可主动开启可选的
Born–Huang 旋转求和规则，详见[求和规则](docs/SUM_RULES_ZH.md)。

## 主要特点

- `order>=2` 使用同一个 API 和数值流程；
- 三阶和四阶力常数经过生产级测试；
- 四阶通过独立 JAX 微分的 FCC Morse 解析能量验证，并检查有限差分的二阶步长收敛；
- 二阶和五阶完成端到端验证；
- 公共结构与计算器接口完全采用 ASE；
- 支持适合外部调度和断点续算的 `sow()` / `reap()`；
- 直接 calculator 路径支持可配置偶次幂阶数的零步长外推；
- 稳定的构型 ID、计划哈希和显式原子顺序映射；
- 与既有逻辑兼容的周期团簇截断；
- 递归中心有限差分模板；
- JAX 加速高阶张量变换，可选择 CPU 或 GPU；
- JAX JIT、`vmap` 和批量张量收缩减少高阶张量处理开销；
- 连续稀疏数组、矩阵无关变换和惰性稠密物化降低峰值内存；
- 位移键去重减少需要调用势函数的构型数量；
- 使用 Wick 正交特征联合拟合 FC2--FCn，并输出兼容通用格式的 Taylor 力常数；
- 稀疏矩阵无关 LSMR 实现严格平移 ASR；
- 二阶力常数可选 Born–Huang 旋转求和规则；
- 任意阶通用稀疏 HDF5；
- 三阶、四阶 ShengBTE 输出；
- 二阶完整 phonopy 文本输出；
- 可选 phonopy/symfc SSCHA，并支持任意 ASE Calculator。

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

需要 SSCHA 时安装可选依赖：

```bash
uv sync --extra sscha
```

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

## 快速开始

### 外部力计算

`sow()` 本身不写文件，也不启动 DFT 程序；它返回一个有确定顺序的 ASE `Atoms` 位移
结构列表。用户可以自行用 ASE 将这些结构写成 VASP 的 `POSCAR-xxx`，也可以选择其他
第一性原理软件的输入格式，再通过任意本地调度系统提交。计算完成后，仍由用户使用 ASE
读取每个结果、提取力，并恢复 `sow()` 的原始顺序（或按构型 ID 建立字典），最后只把
力交给 `reap()`。只要能够保证这个位置顺序，就不需要构型 ID 和计划哈希。

下面是按位置顺序组织的 VASP 案例：

```python
from pathlib import Path

import numpy as np
from ase.io import read, write
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("POSCAR"),
    order=3,
    supercell=(2, 2, 2),
    cutoff=-5,
    displacement=0.01,
    symprec=1e-5,
    jax_platform="auto",  # "auto"、"cpu" 或 "gpu"
)

# 1. sow()：获得与 reap 完全对应的 ASE 位移结构列表。
structures = calculation.sow(atom_order="grouped")
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
fc3 = calculation.reap(
    forces,
    atom_order="grouped",
    acoustic_sum_rule=True,
)
fc3.write("fc3.h5", format="hdf5")
```

若使用 Quantum ESPRESSO、ABINIT、CP2K 或其他外部程序，只需更换 ASE `write()` 和
`read()` 所使用的格式，并补充该程序需要的输入参数；`sow/reap` 契约不变。若文件名和
返回的力严格保持 sow 顺序，位置式 `reap()` 不需要任何额外元数据。POSCAR 这类文件
不会保存 Python 中的 `atoms.info`，因此对于任务乱序、断点续算、长期归档或防止混入
其他数据集的情况，建议用 manifest 保存文件名—构型 ID 对应关系和计划哈希。完整的
[`vasp_external_fc3.py`](examples/vasp_external_fc3.py) 示例把 manifest 作为可选安全层，
并实现力收集、缺失结果检查和最终导出，详见
[外部 VASP 工作流](docs/EXTERNAL_VASP_WORKFLOW_ZH.md)。

力数组的形状必须为：

```text
(len(calculation.sow()), len(calculation.supercell), 3)
```

当位移结构仍是 ASE 对象时，它还包含以下可选审计元数据：

```python
atoms.info["mlfcs_configuration_id"]
atoms.info["mlfcs_plan_hash"]
atoms.info["mlfcs_atom_order"]
atoms.arrays["mlfcs_displacement"]
```

如果外部任务乱序返回，可以按任意顺序读取，但应传入以原始零基构型 ID 为键的字典：

```python
fc3 = calculation.reap(
    forces_by_configuration_id,
    atom_order="grouped",
    plan_hash=calculation.plan.hash,
)
```

缺少或多出 ID、形状错误、非有限数值以及计划哈希不匹配都会被拒绝。

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
[零步长外推](docs/EXTRAPOLATION_ZH.md)。

显式保存力：

```python
forces = calculation.evaluate(calculator)
np.savez_compressed("forces.npz", forces=forces, plan_hash=calculation.plan.hash)
fc3 = calculation.reap(forces, plan_hash=calculation.plan.hash)
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

## 原子顺序

内部超胞的标准顺序为：

```text
z → y → x → primitive_atom
```

原胞原子编号变化最快。这也是 `sow()` 和 `reap()` 的默认顺序。

需要按原胞原子分组时：

```python
structures = calculation.sow(atom_order="grouped")
force_constants = calculation.reap(forces, atom_order="grouped")
```

显式映射为：

```python
calculation.index.grouped_permutation
calculation.index.internal_from_grouped
calculation.index.group_atoms(atoms)
```

导出 phonopy 格式时，MLFCS 会自动完成 grouped 顺序转换。

## 声学求和规则

ASR 默认启用：

```python
constrained = calculation.reap(forces, acoustic_sum_rule=True)
raw = calculation.reap(forces, acoustic_sum_rule=False)
```

受约束结果是在独立参数空间中距离测量结果最近、同时满足平移不变性的解。置换对称性
会给出其他原子轴上的等价约束。

二阶力常数可以主动开启旋转求和规则；默认关闭：

```python
fc2 = calculation.reap(
    forces,
    acoustic_sum_rule=True,
    rotational_sum_rule=True,
)
```

程序会联合投影平移和旋转约束。严格的高阶旋转条件会耦合相邻阶力常数，因此当前
单阶 API 只允许在二阶设置 `rotational_sum_rule=True`。详见
[求和规则](docs/SUM_RULES_ZH.md)。

## 输出格式

输出格式必须显式指定：

```python
fc2.write("FORCE_CONSTANTS", format="phonopy")
fc2.write("fc2.hdf5", format="phonopy_hdf5")
fc3.write("fc3.h5", format="hdf5")
fc3.write("fc3.hdf5", format="phono3py_hdf5")
fc3.write("fc3.npz", format="numpy")
fc3.write("FORCE_CONSTANTS_3RD", format="shengbte")
fc4.write("FORCE_CONSTANTS_4TH", format="shengbte")
```

| 格式 | 阶数 | 表示 |
|---|---|---|
| `hdf5` | 任意阶 | 稀疏团簇张量或稠密数组 |
| `numpy` / `npz` | 任意阶 | 物化后的 NumPy 数组 |
| `shengbte` | 3、4 | 对称闭合、基于晶格平移的文本块 |
| `phonopy` | 2 | 完整稠密超胞 FC2 文本 |
| `phonopy_hdf5` | 2 | phonopy 兼容的完整超胞 `force_constants` HDF5 |
| `phono3py_hdf5` | 3 | phono3py 兼容的完整超胞 `fc3` HDF5 |

ShengBTE 默认使用保真模式：严格写出重建后稀疏结果携带的对称闭合团簇支撑域。
如需复现旧 thirdorder 的二次 joint-image 过滤和块顺序，必须显式启用兼容模式：

```python
fc3.write(
    "FORCE_CONSTANTS_3RD",
    format="shengbte",
    compatibility="thirdorder",
)
```

phonopy 和 phono3py HDF5 使用按原胞原子分组的超胞顺序，并逐个第一原子 slab
流式写入，因此不会在内存中构造完整 FC3。原生 `hdf5` 仍然是 MLFCS 的紧凑、
阶数参数化表示。

高阶结果推荐使用稀疏 HDF5。显式稠密化超过默认 2 GB 建议预算时会发出警告：

```python
fc5.write("fc5.h5", format="hdf5")
dense = fc5.materialize(5)
dense = fc5.materialize(5, max_bytes=None)  # 显式关闭警告预算
```

## 可选 SSCHA

独立的 `mlfcs.sscha` 模块根据热位移上的原子力拟合温度相关有效 FC2：

```python
from mlfcs.sscha import SSCHA

sscha = SSCHA(
    atoms,
    supercell=(3, 3, 3),
    temperature=300,
    snapshots=1000,
    max_iterations=10,
    random_seed=42,
)

sscha.run(calculator)
sscha.use_average(last=5)
sscha.write("fc2-300K.hdf5", format="hdf5")
```

第零轮通过小随机笛卡尔位移拟合初始 FC2。后续每轮由 phonopy 采样谐振子正则系综，
再由 symfc 拟合完整 FC2。因此 `max_iterations=10` 表示一次初始化加十次更新。

也支持逐轮外部计算：

```python
structures = sscha.sow()
result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

拟合 FC2 只需要力；只有估计自由能时才需要能量。已完成轮次保存在
`sscha.history`，`sscha.phonopy` 暴露底层 Phonopy 对象，可继续计算网格、能带、
DOS 和热力学性质。

该方法是随机有效谐波方法，不是显式 FC3 bubble 或 FC4 loop 计算。详见
[SSCHA 文档](docs/SSCHA_ZH.md)。

## 当前限制

- 三阶和四阶是主要生产测试路径；
- 更高阶沿用同一实现，但计算成本可能无法承受；
- ShengBTE 输出仅支持三阶和四阶；
- 尚未实现显式 FC3 bubble 和 FC4 loop 自能；
- 基础重建流程不包含非解析长程静电修正；
- SSCHA 收敛由调用者控制，不内置通用自动停止阈值。

## 文档

- [文档索引](docs/README_ZH.md)（[English](docs/README.md)）
- [外部 VASP 工作流](docs/EXTERNAL_VASP_WORKFLOW_ZH.md)
- [技术总览](docs/TECHNICAL_OVERVIEW_ZH.md)
- [数值验证与持续集成](docs/VALIDATION_ZH.md)
- [仅力数据拟合](docs/FITTING_ZH.md)
- [SSCHA 使用说明](docs/SSCHA_ZH.md)
- [开发路线图](docs/ROADMAP_ZH.md)

## 开发与测试

所有命令使用 uv，测试串行运行：

```bash
uv sync --extra sscha
uv run pytest
uv run pytest -m "not reference"
uv run ruff check src tests reference_tools examples
uv run ruff format --check src tests reference_tools examples
uv build
```

hiphive 和 phono3py 仅作为开发验证依赖。CI 中的 AlN 三阶基准先由 hiphive 将双方的
原子顺序和张量表示统一为完整超胞 FC3，再与独立的 phono3py 有限差分结果做数值比较。
测试层级和各项独立参考测试的运行命令见 [tests/README.md](tests/README.md)。

当前正式版本为 `3.1.0`。版本变化见 [CHANGELOG.md](CHANGELOG.md)，开发流程见
[CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MLFCS 使用 [GNU 通用公共许可证第 3 版或更高版本](LICENSE)发布。三阶算法来源以及
参考借鉴内容与 MLFCS 新增开发之间的边界见[第三方来源说明](THIRD_PARTY_ZH.md)。
