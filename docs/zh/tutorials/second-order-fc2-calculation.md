---
title: 二阶力常数计算：有限差分与力拟合
audience:
  - beginner
status: stable
code_verified: 4.0.0a6
localized_only: true
examples:
  - examples/finite-difference/Si/harmonic
  - examples/fitting/Si/harmonic
---

# 二阶力常数计算：有限差分与力拟合

二阶力常数（FC2）描述体系在参考结构附近的谐性力响应。MLFCS 提供两种常用的 FC2 计算方法：

* **有限差分法**：对参考结构施加小位移，计算相应构型的原子力，再通过中心差分得到 FC2。
* **力常数拟合**：使用一组已知位移和原子力的结构，根据位移与力之间的关系拟合 FC2。

两种方法最终都会得到 `ForceConstants` 对象，因此可以使用相同的 `write_force_constants()` 接口导出为 MLFCS 原生 HDF5、phonopy 或其他支持的格式。

## 如何选择

| 场景                            | 推荐方法                      | 原因                                               |
| ----------------------------- | ------------------------- | ------------------------------------------------ |
| 只需要谐性 FC2，并且可以直接运行 calculator | 有限差分法                     | 流程简单直接，位移生成和差分计算均由 MLFCS 完成                      |
| 已经有 MD、随机位移或其他带力结构            | 力常数拟合                     | 可以直接利用已有数据，并通过多个结构共同确定 FC2                       |
| 使用 VASP 等外部程序计算原子力            | 有限差分法的 `sow()` / `reap()` | MLFCS 生成位移构型，外部程序计算原子力，最后再由 MLFCS 重建 FC2         |
| 需要同时拟合 FC2、FC3 或 FC4          | 力常数拟合                     | 可以通过 `ForceConstantFitter` 的 `orders` 参数联合拟合多个阶数 |

无论使用哪种方法，`primitive` 都表示晶体原胞，`reference` 表示实际计算或拟合所使用的参考超胞。

整个计算过程中，所有结构都必须与 `reference` 保持一致的原子数、元素和原子顺序。位移构型以及对应的力数组也必须遵循相同的顺序。

## 方法一：有限差分法

### Si 实例：使用 NEP 势计算 FC2

本教程使用 Si 的 NEP 势作为 ASE calculator，演示一次可以实际运行的有限差分计算。NEP 模型文件来自相关论文作者维护的 `nep-data` 仓库；本项目只借助该模型计算原子力，不训练或修改 NEP 势。

完整示例位于 `tutorial/Si/finite-difference-ase`。脚本使用本目录中的相对路径，因此必须先进入该目录，再运行脚本：

```bash
cd tutorial/Si/finite-difference-ase
uv run python run.py
```

普通 Python 用户可以将最后一行替换为：

```bash
python run.py
```

如果提示缺少 `calorine`，先安装教程依赖，再重新运行：

```bash
uv pip install calorine
```

不使用 `uv` 时，可以运行：

```bash
python -m pip install calorine
```

`run.py` 的实际代码如下：

```python
import json
from pathlib import Path

from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import FiniteDifferenceCalculation, build_supercell, write_force_constants


MODEL = "Si_2022_NEP3_5body.txt"


def main() -> None:
    primitive = read("POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)
    calculator = CPUNEP(str(MODEL))
    calculation = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=0.01,
    )
    force_constants = calculation.run(calculator)

    write_force_constants(force_constants, "fc2-mlfcs.h5", format="hdf5")
    write_force_constants(
        force_constants,
        "FORCE_CONSTANTS_2ND",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        force_constants,
        "force_constants.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    Path("metadata.json").write_text(
        json.dumps(force_constants.metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

这里的 `primitive` 是从 `POSCAR.vasp` 读取的 Si 原胞，`reference` 是通过 `(4, 4, 4)` 复制得到的 128 原子参考超胞。`SPOSCAR` 保存的就是这个参考超胞，后续 phonopy 读取它来计算声子谱。

`FiniteDifferenceCalculation(order=2)` 指定计算二阶力常数。`displacement=0.01` 的单位是 Å，表示对称有限差分所使用的位移幅度；`cutoff=None` 表示由程序根据周期超胞自动选择不会产生周期像歧义的截断范围。脚本默认使用中心差分，并施加声学和规则 ASR。

运行完成后会得到以下文件：

| 文件 | 作用 |
| --- | --- |
| `SPOSCAR` | 供 phonopy 使用的参考超胞 |
| `fc2-mlfcs.h5` | MLFCS 原生 HDF5 格式，保存稀疏 exact-$R$ 表示 |
| `FORCE_CONSTANTS_2ND` | phonopy 文本格式的二阶力常数 |
| `force_constants.hdf5` | phonopy HDF5 格式的二阶力常数 |
| `metadata.json` | 本次计算的参数和结果元数据 |
| `run.log` | 完整的运行日志|

接着在同一目录运行 `plot.py` 绘制声子谱：

```bash
uv run python plot.py
```

本案例生成的声子谱如下：

![Si 声子谱](https://raw.githubusercontent.com/gtiders/mlfcs/dev/tutorial/Si/finite-difference-ase/phonon-band.png)

如果提示缺少绘图依赖，再安装：

```bash
uv pip install phonopy seekpath matplotlib
```

脚本读取 `SPOSCAR` 和 `force_constants.hdf5`，由 seekpath 生成高对称路径，再由 phonopy 计算路径上的声子频率，最终输出 `phonon-band.png` 和记录路径信息的 `phonon-band.json`。

### 零步长外推

普通中心差分只使用一个位移幅度。有限差分零步长外推会在多个位移幅度上分别计算中心差分，然后根据有限差分结果随 $h^2$ 的变化趋势外推到 $h=0$，从而减小单个位移幅度带来的截断误差。该方法会导致计算量成倍增加因此建议使用机器学习势。

本案例提供了对应的脚本 `run_extrapolation.py`。它同样必须在 `tutorial/Si/finite-difference-ase` 目录中运行：

```bash
uv run python run_extrapolation.py
```

普通 Python 用户可以运行：

```bash
python run_extrapolation.py
```

脚本会将完整运行过程同时显示在终端并保存到 `extrapolation.log`。外推结果使用带有 `-extrapolation` 后缀的文件名保存，不会覆盖普通中心差分的结果：

| 参数或文件 | 含义 |
| --- | --- |
| `displacement=0.01` | 中心位移 $h_0$，单位为 Å；它是外推网格的中心步长 |
| `derivative_backend="extrapolate"` | 选择零步长外推后端 |
| `extrapolation_spacing=0.002` | 相邻位移步长之间的间隔，单位为 Å |
| `extrapolation_side_steps=2` | 在 $h_0$ 两侧各增加 2 个步长，因此一共使用 5 个位移幅度 |
| `extrapolation_degree=1` | 对 $h^2$ 做一次多项式拟合，通常是含噪声原子力下较稳健的选择 |
| `extrapolation.log` | 外推过程日志，包括位移网格、力计算进度、拟合残差和最终 ASR 修正 |

本脚本实际使用的位移网格为 $0.006$、$0.008$、$0.010$、$0.012$ 和 $0.014$ Å。`extrapolation_degree=p` 表示拟合 $D(h)=D_0+c_2h^2+c_4h^4+\cdots$ 到 $h^{2p}$；拟合阶数必须小于位移步长数量。提高 `extrapolation_side_steps` 会增加计算量，使用更高的 `extrapolation_degree` 也不一定更准确，因为原子力噪声可能被放大。

外推脚本还会生成 `SPOSCAR-extrapolation`、`fc2-mlfcs-extrapolation.h5`、`FORCE_CONSTANTS_2ND_EXTRAPOLATION`、`force_constants-extrapolation.hdf5` 和 `metadata-extrapolation.json`。如果要绘制外推结果的声子谱，需要让 `plot.py` 读取外推版的 `SPOSCAR-extrapolation` 和 `force_constants-extrapolation.hdf5`。

### 使用外部程序计算原子力

如果无法在当前 Python 进程中直接调用 VASP、Quantum ESPRESSO 等程序，可以使用 `sow()` 和 `reap()` 将位移结构生成与 FC2 重建分开进行。

首先生成需要计算的位移构型：

```python
structures = calculation.sow()
```

随后可以将这些结构写出，并交给外部程序计算原子力。

如果计算结果按照 `structures` 的原始顺序保存，可以将原子力整理为形状为

```text
(n_configurations, n_reference_atoms, 3)
```

的数组：

```python
import numpy as np

forces = np.asarray([
    np.load(f"forces/{index:05d}.npy")
    for index in range(len(structures))
])

force_constants = calculation.reap(forces)
```

这里要求 `forces[i]` 与 `structures[i]` 一一对应。

如果外部任务的完成顺序与提交顺序不同，建议使用每个结构自带的 `mlfcs_configuration_id` 进行匹配。`reap()` 也可以直接接收以该 ID 为键的字典：

```python
forces_by_id = {
    atoms.info["mlfcs_configuration_id"]: np.load(
        f"forces/{atoms.info['mlfcs_configuration_id']:05d}.npy"
    )
    for atoms in structures
}

force_constants = calculation.reap(forces_by_id)
```

程序会检查构型 ID 是否完整、力数组形状是否正确、数据中是否存在非有限值，以及原子顺序是否与参考超胞一致。

因此，在收集外部计算结果时，应始终保持位移构型与原子力之间的正确对应关系。

## 方法二：力常数拟合

力常数拟合不要求使用成对的正、负位移构型。只要有一组相对于参考结构发生位移、并且已经计算好原子力的结构，就可以直接用于拟合 FC2。

本教程使用真实的 Si NEP 势生成训练数据。完整案例位于
`tutorial/Si/force-fitting-ase`，必须进入该目录运行：

```bash
cd tutorial/Si/force-fitting-ase
uv run python run.py
```

普通 Python 用户可以运行：

```bash
python run.py
```

脚本首先构造 4×4×4、128 原子的 Si 超胞，然后调用顶层的
`perturb_structures()` 生成 3 个去质心高斯微扰结构：

```python
from ase.io import read, write
from calorine.calculators import CPUNEP

from mlfcs import build_supercell, perturb_structures, write_force_constants
from mlfcs.fitting import ForceConstantFitter

primitive = read("POSCAR.vasp")
reference = build_supercell(primitive, (4, 4, 4))
write("SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)

snapshots = perturb_structures(
    primitive,
    reference=reference,
    snapshots=3,
    displacement=0.01,
    random_seed=42,
)

calculator = CPUNEP("Si_2022_NEP3_5body.txt")
for snapshot in snapshots:
    snapshot.calc = calculator
    forces = snapshot.get_forces().copy()
    snapshot.info.clear()
    for name in tuple(snapshot.arrays):
        if name not in {"numbers", "positions"}:
            del snapshot.arrays[name]
    snapshot.new_array("forces", forces)
    snapshot.calc = None
write("train.extxyz", snapshots, format="extxyz")

training = read("train.extxyz", index=":")

fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2,),
    cutoffs={2: None},
)

gram = fitter.prepare_gram(
    training,
    acoustic_sum_rule=True,
)
gram.save("training-gram.npz")
result = fitter.fit(gram, acoustic_sum_rule=True)

write_force_constants(
    result.force_constants,
    "fc2-fit.h5",
    format="hdf5",
)

write_force_constants(
    result.force_constants,
    "FORCE_CONSTANTS_2ND",
    format="phonopy",
    order=2,
)
```

`perturb_structures()` 是 SSCHA 在没有初始 FC2 时使用的高斯笛卡尔微扰路径。它只生成结构，不计算力，也不执行自洽迭代。`displacement=0.01` 的单位是 Å，`snapshots=3` 表示生成 3 个结构，`random_seed=42` 用于复现实验。

拟合完成后，在同一目录运行 `plot.py`：

```bash
uv run python plot.py
```

它读取 `SPOSCAR` 和 `force_constants-fit.hdf5`，使用 seekpath 和 phonopy 生成拟合 FC2 的声子谱，输出 `fit-phonon-band.png` 和 `fit-phonon-band.json`。这 3 个结构用于演示从采样、清理内部元数据、保存、读取、拟合到声子谱的完整接口；由于训练结构较少，正式拟合仍应增加快照数量并保留独立验证集检查泛化误差。

所有训练结构都必须与 `reference` 保持一致的：

* 晶格；
* 原子数；
* 元素标签；
* 原子顺序。

拟合过程只使用原子力，不使用总能量。

如果训练结构从文件读取，每个 `Atoms` 对象都必须能够提供原子力，例如已经保存 ASE calculator 的计算结果，或者包含可读取的 `forces` 数据。

如果参考结构本身存在非零原子力，MLFCS 会将其作为参考残余力，并从训练目标中扣除。这样拟合得到的是相对于参考结构的力响应，而不会把参考结构上的静态残余力混入力常数。

`cutoffs={2: 5.4}` 和 `max_body_orders={2: 2}` 用于定义 FC2 的拟合参数空间。

在有限差分接口中，对应的参数分别为 `cutoff` 和 `max_body_order`。如果需要比较有限差分和拟合得到的 FC2，两种方法应尽量使用一致的：

* `primitive`；
* `reference`；
* cutoff；
* body order；
* ASR 设置。

训练集和测试集分别构造独立的 Gram 对象。`fit()` 不再隐式划分验证集或预测测试集力；
测试误差应由用户使用 `MLFCSCalculator` 显式计算。

默认情况下：

```python
regularization=None
```

表示进行不带正则项的拟合。

如果希望对轨道组施加稀疏约束，可以显式设置：

```python
regularization="scaled_group_lasso"
```

使用正则化后，应结合 `result` 中的收敛信息和力误差判断拟合结果是否可靠。

## 结果检查与导出

有限差分得到的 `ForceConstants` 可以直接检查其阶数和元数据：

```python
print(force_constants.orders)
print(force_constants.metadata)
```

对于力常数拟合，最终的力常数保存在：

```python
result.force_constants
```

例如：

```python
force_constants = result.force_constants
```

拟合过程中的参数、误差和收敛信息则保存在 `result` 的其他字段中。

随后可以使用统一的 writer 导出结果：

```python
write_force_constants(
    force_constants,
    "fc2.h5",
    format="hdf5",
)

write_force_constants(
    force_constants,
    "FORCE_CONSTANTS_2ND",
    format="phonopy",
    order=2,
)
```

MLFCS 原生 HDF5 保存的是稀疏 exact-$R$ 表示。

导出为 phonopy 格式时，MLFCS 会将 FC2 写成下游程序能够直接读取的超胞力常数格式。

如果需要将力常数用于与原参考超胞不同的目标超胞，应先显式调用 `realize_force_constants()`，在目标超胞上构造对应的力常数。

不应直接将原参考超胞上的力常数数组重新解释为另一个超胞中的力常数。

## 常见问题

### 位移幅度应该取多大？

位移过小时，原子力中的数值噪声会在差分过程中被放大；位移过大时，高阶非谐项又会逐渐影响二阶导数。

因此，在第一性原理计算中，建议测试多个位移幅度，并比较 FC2 和声子频率是否稳定。

对于有限差分法，还应同时检查参考超胞大小和 cutoff 的收敛性。

### 为什么力常数拟合与有限差分的结果不一致？

两种方法都用于描述参考结构附近的二阶力响应，但所使用的数据和数值过程不同，因此实际结果可能存在一定差异。

常见原因包括：

* 拟合数据覆盖的位移范围过窄；
* 两种方法使用了不同的 cutoff 或 body order；
* 训练结构与参考超胞的原子顺序不一致；
* 参考结构存在残余力，但两种方法的处理方式不同；
* 有限差分位移过大，高阶非谐项进入了二阶导数估计；
* 两种方法使用了不同的 ASR 设置。

比较之前，建议先统一 `primitive`、`reference`、cutoff、body order、单位和 ASR 设置，再比较 FC2 张量、声子频率以及对原子力的重建误差。

### 可以固定已有的 FC2，再单独拟合 FC3 或 FC4 吗？

不能。

MLFCS 的高阶力常数拟合在统一的 Taylor 参数空间中进行。不同阶数共享同一份力数据，因此需要在同一次拟合中共同确定。

例如，同时拟合 FC2、FC3 和 FC4：

```python
orders=(2, 3, 4)
```

如果只需要 FC2，则使用：

```python
orders=(2,)
```

即可。

更多高阶力常数拟合方法见[力常数拟合](first-fc2-fitting.md)。

有限差分的完整接口见[有限差分 API](../reference/finite-difference-api.md)，拟合器参数和诊断信息见[拟合 API](../reference/fitting-api.md)。
