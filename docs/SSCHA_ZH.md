# SSCHA 模块

[English](SSCHA.md)

`mlfcs.sscha` 使用 phonopy 生成随机位移、使用 symfc 拟合有效二阶力常数，并通过
ASE `Calculator` 接收任意势函数。它是一个独立的可选模块，不会让 MLFCS 原有的
二至高阶有限差分流程依赖 phonopy 或 symfc。

## 安装

```bash
uv sync --extra sscha
```

可选依赖目前为：

- `phonopy>=2.43`：超胞、谐振子正则系综采样、声子和热力学量；
- `symfc>=1.5`：利用空间群、置换和求和规则拟合完整 FC2。

用户负责安装并构造实际使用的 ASE Calculator。MLFCS 不依赖 calorine、MACE、
GAP 或其他具体势函数包。

## 算法流程

默认计算包含一次初始化和 `max_iterations` 次自洽更新：

```text
小随机笛卡尔位移
        │
        ▼
ASE 力/能量 ──► symfc 拟合初始 FC2
                         │
                         ▼
             phonopy 谐振子正则系综采样
                         │
                         ▼
                 ASE 力/能量
                         │
                         ▼
                 symfc 重新拟合 FC2
                         │
                         └── 重复 max_iterations 次
```

因此 `max_iterations=10` 总共进行 11 次 FC2 拟合。若构造时传入
`initial_force_constants`，第一轮便直接使用它进行正则系综采样，不再执行笛卡尔
初始化；总拟合轮数仍为 `max_iterations + 1`。

这与显式 FC3 气泡图、FC4 loop 图或 ALAMODE 的确定性 SCPH 展开不同。这里的非谐性
来自势函数在热位移构型上的真实力，最终得到指定温度下的有效谐波 FC2。

## 直接使用 ASE Calculator

```python
from ase.io import read
from mlfcs.sscha import SSCHA

atoms = read("POSCAR")
calculator = make_my_ase_calculator()

sscha = SSCHA(
    atoms,
    supercell=(3, 3, 3),
    temperature=300.0,
    snapshots=1000,
    max_iterations=10,
    initial_displacement=0.01,
    random_seed=42,
    symprec=1e-5,
    log_level=1,
)

sscha.run(calculator)
```

`run()` 串行计算每个构型，避免同时复制多个大型计算器造成内存峰值。可以传入
进度回调：

```python
def progress(done: int, total: int) -> None:
    print(f"{done}/{total}")

sscha.run(calculator, progress=progress)
```

默认还会计算每个构型和未位移超胞的势能，以估计自由能。若只需要温度重整化
FC2，可以减少能量调用：

```python
sscha.run(calculator, calculate_free_energy=False)
```

## 外部 sow/reap 工作流

对于集群调度、断点续算或其他外部力计算程序，可以逐轮使用 `sow()` 和 `reap()`：

```python
structures = sscha.sow()

for atoms in structures:
    iteration = atoms.info["mlfcs_sscha_iteration"]
    configuration_id = atoms.info["mlfcs_configuration_id"]
    dispatch(iteration, configuration_id, atoms)

result = sscha.reap(
    forces_by_configuration_id,
    energies=energies_by_configuration_id,
    reference_energy=equilibrium_supercell_energy,
)
```

一轮内的构型编号从零开始，并且必须完整覆盖 `0..N-1`。顺序数组必须严格按照
`sow()` 返回顺序；字典可以乱序到达，但键必须是 `mlfcs_configuration_id`。

力的形状为：

```text
(number_of_snapshots, number_of_supercell_atoms, 3)
```

单位遵循 ASE：位移为 Å、力为 eV/Å、能量为 eV，FC2 为 eV/Å²。

只有力是 FC2 拟合的必要输入。`energies` 和未位移超胞的 `reference_energy` 只用于
自由能估计；缺少其中任意一项时，该轮 `free_energy` 和 `free_energy_error` 为
`None`。

## 每轮结果和收敛监控

每次 `reap()` 或 `step()` 返回不可变的 `SSCHAIteration`，同时追加到
`sscha.history`。记录字段包括：

- `index`：从零开始的拟合轮次；
- `sampling`：`cartesian` 或 `canonical`；
- `force_constants`：完整超胞 FC2，形状 `(N, N, 3, 3)`；
- `free_energy`：每原胞自由能，单位 eV；
- `free_energy_error`：非谐校正样本均值的标准误差；
- `potential_energy`：超胞平均相对势能；
- `harmonic_potential_energy`：超胞平均谐波势能。

例如监控相邻两轮 FC2：

```python
import numpy as np

previous = sscha.force_constants
result = sscha.step(calculator)
if previous is not None:
    rms = np.sqrt(np.mean((result.force_constants - previous) ** 2))
    print("FC2 RMS change:", rms)
```

当前 API 严格执行用户指定的轮数，不会擅自使用某个通用阈值提前停止。用户可以
逐轮调用 `step()`，根据 FC2、自由能或频率的收敛情况自行决定是否继续。

## 末轮平均和写出

随机拟合可能在收敛点附近波动。可以计算或启用末几轮平均：

```python
average = sscha.averaged_force_constants(last=5)
sscha.use_average(last=5)  # 同时设为 phonopy 当前 FC2
```

写出使用 phonopy 原生完整 FC2 格式：

```python
sscha.write("fc2-300K.hdf5", format="hdf5")
sscha.write("FORCE_CONSTANTS", format="text")
```

底层 Phonopy 对象可用于后续分析：

```python
ph = sscha.phonopy
ph.run_mesh([20, 20, 20])
ph.run_thermal_properties(temperatures=[300])
```

## 自由能定义

当前实现沿用 phonopy SSCHA 的估计式。对当前 FC2 `Φ`：

```text
F = F_harm + mean[(E(u) - E(0) - 1/2 u Φ u) / n_cell]
```

其中 `F_harm` 是 phonopy 计算的每原胞谐波自由能，`n_cell` 是超胞包含的原胞数。
报告的误差是括号内有限样本均值的标准误差，不包含 FC2 自身拟合不确定度，也不
表示不同自洽轮次之间的系统误差。

## 稳定性和内存注意事项

- 热采样依赖当前 FC2 能够建立合理的谐振子分布。明显不稳定的初始结构可能需要
  先提供稳定的 `initial_force_constants`，或先完成零温谐波计算。
- `cutoff_frequency` 可排除过低频率模式；`max_displacement` 可限制随机位移幅度。
- 固定 `random_seed` 可重复随机位移。每轮使用相同基础随机样本，但 FC2 改变后模态
  变换也会改变实际笛卡尔位移。
- 完整 FC2 的存储规模为 `N²×3×3`。大型超胞的快照力数组和 symfc 基底也会占用
  显著内存，应根据体系大小选择 `snapshots`。
- `run()` 有意串行调用 ASE Calculator。外部并行应通过 `sow/reap` 完成，并由用户
  控制并发和内存。
- 非解析长程修正可通过 `sscha.phonopy.nac_params` 配置，但需要用户提供与 phonopy
  一致的 Born 电荷和介电张量。

## 与旧版接口的主要区别

| 项目 | 旧版 `MLPSSCHA` | 当前 `mlfcs.sscha.SSCHA` |
|---|---|---|
| Calculator 接入 | phonopy MLP 适配器 | 直接 ASE Calculator |
| 外部计算 | 不提供完整逐轮接口 | `sow()` / `reap()` |
| 计算时机 | 构造时计算参考能量 | 构造无势函数计算，首次 `step/run` 才计算 |
| 随机性 | 未完整暴露种子 | `random_seed` |
| 每轮结果 | 主要通过打印 | 结构化 `SSCHAIteration` 历史 |
| 自由能误差 | 未提供 | 提供有限样本标准误差 |
| FC2 平均 | `run()` 内部选项 | `averaged_force_constants()` / `use_average()` |
| 写出 | 运行结束自动写出 | 显式 `write()` |
| 可选依赖边界 | secondorder 包内耦合 | 独立 `mlfcs.sscha` 模块 |

显式写出和显式平均让计算过程不产生隐式文件副作用，也便于用户在不同温度、不同
收敛判据和外部调度系统中复用同一个 API。
