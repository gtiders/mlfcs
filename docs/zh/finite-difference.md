# 有限差分

`FiniteDifference` 定义一次力常数重建任务。它生成有序的 ASE 结构序列；每个结构的力都必须与该几何及原子顺序准确对应，直到重建完成。

## 对象与构造参数

```python
FiniteDifference(mapping, *, order, disps=0.01)
```

| 参数 | 含义 |
| --- | --- |
| `mapping` | 将原胞模型连接到明确超胞的 `ClusterMap`。 |
| `order` | 要重建的力常数阶次；该阶必须在簇空间中配置，且在此超胞中结构可辨识。一个 `FiniteDifference` 对象处理一个阶次。 |
| `disps` | 一个正位移长度（Å），或多个互不相同的正长度；默认 `0.01` Å。多个长度会触发零位移外推。 |

对象公开 `mapping`、`order`、`disps` 和 `n_configurations`。`displacements()` 按重建顺序生成规范结构序列。若选取的簇键数量为 $N_k$、位移长度数量为 $N_d$、力常数阶次为 $n$，则 `n_configurations = N_k N_d 2^{n-1}`。

## 使用 ASE 计算器的 Si 最小示例

```python
from ase.calculators.emt import EMT
from ase.io import read
from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, Supercell

primitive_atoms = read("POSCAR")
space = ClusterSpace(primitive_atoms, cutoffs={2: 6.0}, max_body_orders={2: 2})
supercell = Supercell.from_atoms(space.primitive, read("SPOSCAR"))
mapping = ClusterMap.build(space, supercell)

fd = FiniteDifference(mapping, order=2, disps=0.01)
evaluated = fd.evaluate(EMT())
fc2 = fd.reconstruct(evaluated)
```

`evaluate(calculator)` 会对每个位移结构强制进行一次新力计算，并返回已存储力的 ASE 结构。任何支持所选元素和结构的 ASE 计算器都可以提供力，包括封装 MACE、NEP 等机器学习势的计算器。

## 外部 DFT 和其他外部力计算程序

如果力由 ASE 之外的程序计算，仍使用相同的生成结构序列：

1. 遍历 `fd.displacements()`，以能够保持晶胞和原子顺序的格式写出结构。
2. 对每个结构运行外部计算。
3. 按相同顺序读回结构和力。
4. 把每组力附到对应 ASE 结构，再进行重建。

```python
from ase.calculators.singlepoint import SinglePointCalculator

for atoms, forces in zip(displaced_atoms, force_arrays, strict=True):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)

fc2 = fd.reconstruct(displaced_atoms)
```

这里的 `force_arrays` 是每个结构一组形状为 `(n_atoms, 3)`、单位 eV/Å 的力。重建输入是带有力的有序 ASE 结构，而不是裸力数组集合。仓库中的 [Si VASP 示例](../../tutorial/SI/finite-difference-fc2/run.py)展示了按序写出 VASP 结构、读取 `vasprun.xml` 并重建 FC2。

不提供 ASE 计算器接口的外部机器学习势或电子结构程序也可以使用相同方式：写出结构、计算力，再将力放回对应 ASE `Atoms`。若势函数已有 ASE calculator，则直接使用 `evaluate()`。

## 多个位移长度与外推

`disps` 可以是多个长度，例如：

```python
fd = FiniteDifference(mapping, order=3, disps=(0.005, 0.01, 0.015))
```

对每个位移模式，重建过程先计算中心差分估计，再使用关于位移平方的偶次误差多项式，将不同长度的结果外推至零位移。如果不同幅度的力计算保持一致，这种做法可以减小有限步长截断误差。构型数量会随位移长度数量成比例增加。所有长度都使用同一个 mapping、原子顺序、力单位和力源定义。

外推与力来自 ASE DFT 计算器、ASE 机器学习计算器或外部计算流程无关。[Si FC2/FC3 外推示例](../../tutorial/SI/fc23-extrapolated/run.py)展示了多位移长度流程及生成的模型文件。

## 重建与结果

`fd.reconstruct(structures)` 返回只包含所选阶次的 `ForceConstants`。它会检查序列长度、原子序数和顺序、周期晶胞与坐标，以及已存储力是否存在且有限。它不会启动计算，也不会猜测原子重排。

```python
projection = fc2.enforce_asr(orders=(2,))
fc2_asr = projection.force_constants
fc2_asr.save("fc2-si.mlfcs")
fc2_asr.write("fc2-phonopy.hdf5", mapping, format="phonopy_hdf5", order=2)
```

ASR 是明确的后续投影步骤。`ForceConstants` 支持原生保存/加载和格式导出；详见[力常数](force-constants.md)。
