# 有限差分：从有序位移结构恢复力常数

有了原胞模型和参考超胞，接下来要决定怎样移动原子，才能测到模型中的独立力常数分量。对于二阶力常数，可以比较原子向正、负方向移动后产生的力；对更高阶力常数，则需要组合多个位移方向，得到力的混合导数。

`FiniteDifference` 从模型所需的代表分量出发，准备这些位移构型。使用时先用 `sow()` 生成结构，在外部计算并保存力，然后把有序结果收集为 `ForceDataset`，交给 `reap()` 重建。力由什么程序计算，可以自行选择；结构与结果之间的对应顺序必须保留。

如果还没有建立原胞模型和 mapping，建议先阅读[核心概念](core-concepts.md)。本章继续使用 Al 与 ASE 的 EMT 势，展示从采样到重建的完整过程。

## 1. 为一个阶数准备采样

有限差分需要的位移取决于希望恢复哪一阶力常数。二阶力常数需要力对位移的一阶导数，三阶需要二阶导数；一般的 $p$ 阶力常数需要 $p-1$ 次位移求导。因此，一个 `FiniteDifference` 对象只处理一个阶数。

```python
from ase.build import bulk
from mlfcs import ClusterSpace, ClusterMap, FiniteDifference

primitive = bulk("Al", "fcc", a=4.05)
space = ClusterSpace(
    primitive,
    cutoffs={2: 4.0},
    max_body_orders={2: 2},
    asr=True,
)
mapping = ClusterMap(space, space.primitive_atoms.repeat((3, 3, 3)))
fd = FiniteDifference(mapping, order=2, disps=0.01)

print("需要计算的构型数：", fd.n_configurations)
```

构造时，第一个参数 `cluster_map` 是前面准备的参考超胞映射。关键字 `order` 必须给出，取不小于 2、且已包含在 `ClusterSpace` 中的整数。构造时会检查这一阶在超胞中是否完整可辨识：超胞折叠丢失参数时抛出 `AliasingError`，不会继续生成一个无法恢复全部参数的计划。未包含的阶数抛出 `KeyError`，小于 2 的阶数被拒绝。

`disps` 是以 Å 为单位的中央差分步长，默认 `0.01`。可以给一个数，也可以给一组互不相同的正、有限步长，例如 `disps=(0.01, 0.02)`。空序列、重复值、零、负值或非有限值会抛出 `ValueError`。步长会升序排列，因此传入 `(0.02, 0.01)` 与 `(0.01, 0.02)` 产生相同的计划。

构造完成后，可以查看所有公开属性：

| 属性 | 在采样时的含义 |
|---|---|
| `fd.cluster_map` | 与计划绑定的参考超胞映射。 |
| `fd.order` | 本次恢复的力常数阶数。 |
| `fd.disps` | 升序排列的步长元组；输入单个数时也保存为元组。 |
| `fd.n_configurations` | 整个计划所需的结构数量，包括所有步长和符号组合。 |

后续采样与重建应使用同一个计划，保持这些设置不变。更换阶数或步长时应构造新的 `FiniteDifference`，并重新准备相应的力结果。

### 怎样理解步长的选择

步长并非越小越好。减小步长能够降低有限差分截断误差，但也会放大力计算中的噪声，尤其在高阶求导时。这里的 `0.01 Å` 是示例与默认设置，不是所有材料、阶数或电子结构计算都适用的收敛结论。

多个步长允许在重建时做零步长外推。中央差分估计的误差通常按步长的偶次幂展开：

$$
D(h)=D(0)+c_2h^2+c_4h^4+\cdots.
$$

例如，使用 `disps=(0.01, 0.02)` 时，程序会结合两个步长的估计，消除领先的 $h^2$ 误差。这需要计算两套采样结构；不是在同一份力数据上更换一个重建选项。单步长也可以正常工作，只是没有多步外推。

外推不能消除力噪声。步长过于接近时，组合权重也可能放大噪声，因此应比较不同步长下的结果，而不是单纯增加步长数量。具体推导见[从原子力恢复力常数](theory/reconstruction.md)。

## 2. 用 `sow()` 生成结构

```python
structures = fd.sow()
first = structures[0]
print(len(structures))
```

`sow()` 没有参数，返回懒生成的 ASE `Atoms` 序列。可以迭代、用整数索引访问，也可以切片；负索引按普通序列处理，越界会抛出 `IndexError`。切片返回对应结构的元组。

每次访问都会从参考超胞生成一份独立结构。修改 `first` 不会改变 mapping，也不会改变之后再次访问 `structures[0]` 得到的构型。返回结构不携带力结果或采样 metadata；`sow()` 本身也不写文件。

### 顺序为什么是接口的一部分

一个混合导数需要若干 Cartesian 位移坐标，程序将这组坐标称为位移键。每个键下，按升序步长生成所有正负符号组合；位移键本身也按确定顺序排列。因此，采样依次经过：位移键、步长、符号组合。

用户通常不需要拆解这些键，但必须保持 `sow()` 给出的顺序。它也是 `reap()` 读取力结果的顺序。并行计算可以按任意顺序完成，收集结果时却必须回到原来的采样索引，不能按完成时间或文件名字典顺序拼接。

高阶导数可能重复使用同一个原子、同一个方向。这些带符号位移会在同一个自由度上相加。例如，二次力导数对同一方向求导时，构型中会出现 `-2h`、`0`、`0`、`+2h`。所以 `disps` 表示差分因子的步长，并不保证每个构型的总位移都等于这个数。

每个键有 $2^{p-1}$ 个符号组合。当前计划保留这些条目，即使某些结构几何相同，也应保持所有条目及其原有位置。不能先去重结构再直接将缩短的结果交给 `reap()`。

## 3. 在外部计算并保存每帧的力

现在可以把这些结构交给计算器。下面逐帧使用 EMT，并将已计算的力保存为 ASE 单点结果：

```python
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from mlfcs import ForceDataset

calculated = []
for sample in fd.sow():
    sample.calc = EMT()
    forces = sample.get_forces()
    sample.calc = SinglePointCalculator(sample, forces=forces)
    calculated.append(sample)

dataset = ForceDataset(mapping, calculated)
```

力计算发生在 `get_forces()`，属于外部流程。`ForceDataset` 随后只读取已有结果；仅附上一个尚未计算的 calculator 不够。保存力后应保持结构不变，避免结果失效。

也可以将 `sow()` 的结构写给 VASP 等程序，计算完成后通过 ASE reader 读回已有力的结构。外部工作流应保存采样索引，并据此收集结果。重新生成计划时，需要使用相同原胞模型、参考超胞原子顺序、阶数与步长；MLFCS 不会从计算结果中推断原计划。

数据收集会检查晶胞、元素序列、三维周期性，以及力的形状和有限值。但这些检查不能识别同元素原子的交换，也不能确认帧是否对应预期的有限差分位移。完整的数据约定见[核心概念中的数据集部分](core-concepts.md)。

## 4. 用 `reap()` 恢复力常数

```python
fc2 = fd.reap(dataset)
print(fc2.orders)
print(fc2.coefficients[2])
```

`reap(dataset)` 接受一个 `ForceDataset`，不直接接受结构列表或原始力数组。数据集必须来自这个计划的 mapping，并包含与 `sow()` 一一对应的全部力结果。

重建先对正负位移力做中央混合差分，再组合多个步长的估计，最后恢复各对称轨道的物理参数。由于原子力是能量对位移导数的负值，输出已经转换为能量导数的符号；不需要再人为取负号。

返回值是 `ForceConstants`，只包含本次 `order` 的系数，即使关联的 `ClusterSpace` 定义了多个阶数。$p$ 阶系数单位为 eV/Å$^p$；它是原胞参数化模型，不是完整的超胞张量数组。`coefficients[order]` 可用于查看所恢复的参数；这些参数对应标准物理参数布局，而非 ASR 自由坐标。

### 重建能检查什么

不是 `ForceDataset` 会抛出 `TypeError`；力数组数量、形状或有限值不符合计划时会抛出 `ValueError`。预期形状是 `(fd.n_configurations, mapping.n_atoms, 3)`。

通过这些检查，并不代表采样对应关系正确。`reap()` 使用的是存储的力，不核对数据集的位移是否等于计划位移，也不通过 metadata 修复顺序；它不会另行验证数据集 mapping 与计划的参数布局是否一致。同样长度的错误排序可能得到错误结果而不报错。因此最稳妥的做法是始终使用同一个 `mapping`，按 `sow()` 顺序计算、存储和收集。

### ASR 在哪里开启

ASR 由构造模型空间时的 `ClusterSpace(..., asr=True)` 开启，本章示例已经这样设置。`FiniteDifference` 和 `reap()` 没有另一个 ASR 参数。

开启后，重建先得到未约束物理参数，再通过最小二乘投影到预先准备的声学约束子空间。它并不将所有原始力样本与 ASR 方程放进一次联合拟合。关闭 ASR 时直接返回未约束参数；两种设置下采样数量、顺序、中央差分和步长外推都相同。

这次投影可能调整重建出的分量，使参数满足平移不变性。它不能修正错误采样顺序，也不保证模型截断或力计算已经收敛。若投影迭代未收敛，会抛出 `RuntimeError`。约束的数学定义见[平移与旋转不变性](theory/invariance-constraints.md)。

## 5. 需要先扣除一份已知力时

有时希望恢复的是扣除某种已知贡献后的力常数。应先改变数据集的目标力，再用同一个计划重建：

```python
reference = mapping.supercell_atoms
reference.calc = EMT()
reference_forces = reference.get_forces()

residual_data = dataset.subtract_forces(reference_forces)
residual_fc2 = fd.reap(residual_data)
```

这里 `(mapping.n_atoms, 3)` 的参考力从每帧扣除，原数据集不变。严格不随位移变化的参考力本来会在中央差分中相消，因此并不是所有有限差分计算都需要这一步。

如果扣除的是随位移变化的贡献，应提供与数据集同顺序的逐帧力，例如：

```python
# long_range 是已为同一个 mapping 构造的 DipoleEwald 对象。
residual_data = dataset.subtract_forces(long_range.forces(dataset.displacements))
short_fc2 = fd.reap(residual_data)
```

重建此时针对剩余力。多步外推仍使用原来的采样与步长，只是每帧的目标力已经改变。扣除与回加的物理约定见[长程力与 Ewald](ewald-api.md)。

## 6. 一个完整的二阶例子

下面的例子可独立运行。它展示接口衔接；4 Å 截断和两组步长仍需在实际研究中做收敛检查。

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from mlfcs import ClusterSpace, ClusterMap, FiniteDifference, ForceDataset

primitive = bulk("Al", "fcc", a=4.05)
space = ClusterSpace(
    primitive,
    cutoffs={2: 4.0},
    max_body_orders={2: 2},
    asr=True,
)
mapping = ClusterMap(space, space.primitive_atoms.repeat((3, 3, 3)))
fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))

print("力计算次数：", fd.n_configurations)
calculated = []
for sample in fd.sow():
    sample.calc = EMT()
    forces = sample.get_forces()
    sample.calc = SinglePointCalculator(sample, forces=forces)
    calculated.append(sample)

dataset = ForceDataset(mapping, calculated)
fc2 = fd.reap(dataset)
print("包含的阶数：", fc2.orders)
print("二阶参数数：", len(fc2.coefficients[2]))
```

要恢复三阶或四阶，先在 `ClusterSpace` 中包含相应阶数，再分别构造 `FiniteDifference(mapping, order=3, ...)` 或 `order=4` 并计算各自的采样。不同阶数的计划和结果不能混用。

同一空间下得到的不同阶数模型可以用 `ForceConstants.combine([fc2, fc3])` 合并。`save()` 保存原生参数化模型，`write()` 在指定 mapping 下导出其他格式；数据准备与参数恢复完成后，再选择后续输出流程即可。

更完整的计算与外推结果见[Si 有限差分教程](notebooks/si-finite-difference.ipynb)。如果已有任意位移的力样本，而非本章要求的中央差分采样，可以考虑[力拟合](fitting-api.md)。

整个流程中，`FiniteDifference` 给出该测哪些结构，`sow()` 生成它们；外部程序算力，`ForceDataset` 按原顺序收集结果，`reap()` 恢复所选阶数。保持计划、mapping 和结果顺序一致，是正确重建的前提。
