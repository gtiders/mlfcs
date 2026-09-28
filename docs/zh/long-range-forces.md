# 极性晶体中的长程力

极性晶体中的偶极-偶极相互作用会产生长程谐性力常数。有限范围的簇模型可以高效表示短程部分，长程部分则由 Born 有效电荷和介电张量计算。MLFCS 通过 ASE 计算器 `Ewald` 显式提供这种分离流程。

## 物理分解

将谐性力常数写为

$$
\Phi^{\mathrm{total}} = \Phi^{\mathrm{short}} + \Phi^{\mathrm{Ewald}}.
$$

对位移 $u$，Ewald 计算器给出谐性长程力 $F^{\mathrm{Ewald}}=-\Phi^{\mathrm{Ewald}}u$。给定训练力 $F^{\mathrm{data}}$，短程模型拟合的目标是

$$
F^{\mathrm{short}} = F^{\mathrm{data}} - F^{\mathrm{Ewald}},
$$

之后将 Ewald 张量加到拟合得到的短程 FC2 张量上，组装总 FC2。两项力必须使用相同的 mapping 和同一位移几何。

## `Ewald` 构造函数与参数

```python
from mlfcs.tools import Ewald

ewald = Ewald(
    space,
    mapping,
    born,
    dielectric,
    accuracy=1e-10,
    eta=None,
)
```

完整签名为 `Ewald(space, mapping, born, dielectric, *, accuracy=1e-10, eta=None)`。

| 参数 | 含义 |
| --- | --- |
| `space` | 描述原胞基元的 `ClusterSpace`。 |
| `mapping` | 同一簇空间对应的 `ClusterMap`，超胞固定且为三维周期结构。 |
| `born` | Born 有效电荷张量，形状 `(n_primitive, 3, 3)`，单位基本电荷。第一个指标遵循原胞位点顺序；张量指标依次表示电场方向和位移方向。物理上，对原胞位点求和应满足电荷中性。 |
| `dielectric` | 电子介电张量，形状 `(3, 3)`，无量纲。实现会将输入对称化，并要求结果正定。 |
| `accuracy` | Ewald 实空间/倒空间求和精度目标，严格位于 0 与 1 之间；默认 `1e-10`。更小的值要求更完整的晶格求和。 |
| `eta` | 可选的正 Ewald 分割参数，单位 Å$^{-1}$。`None` 时根据变换后晶胞体积选取。改变 `eta` 会改变实空间和倒空间的分配，不改变收敛后的物理总和。 |

`space` 与 `mapping.space` 必须对应同一个模型。计算器以 `mapping.supercell` 固定晶胞和原子序列；输入结构需保持该晶胞、周期性和原子顺序。

## 力与 FC2 输出

`Ewald` 实现 ASE 计算器的 `energy` 和 `forces` 属性。对使用该映射超胞的位移 ASE 结构：

```python
long_range_forces = ewald.get_forces(atoms)
long_range_fc2 = ewald.fc2
```

`get_forces(atoms)` 计算该几何位置的谐性长程力。`ewald.fc2` 返回形状为 `(n_primitive, n_supercell, 3, 3)` 的副本，使用与 `ForceConstants.get(2, mapping)` 相同的原胞优先紧凑约定，单位 eV/Å²。Ewald FC2 构造时将现场块设为其他块之和的负值，因此有限超胞长程分量按构造满足 ASR。

## 拟合短程力并组装总 FC2

```python
from ase.calculators.singlepoint import SinglePointCalculator
from mlfcs import FitSystem
from mlfcs.force_constants import write_phonopy

short_frames = []
for frame in training_frames:
    short = frame.copy()
    short.calc = SinglePointCalculator(
        short,
        forces=frame.get_forces() - ewald.get_forces(frame),
    )
    short_frames.append(short)

system = FitSystem.from_atoms(mapping, short_frames)
short_model = system.force_constants(system.solve())
short_model = short_model.enforce_asr(orders=(2,)).force_constants

total_fc2 = short_model.get(2, mapping) + ewald.fc2
write_phonopy("fc2-total.hdf5", total_fc2, mapping, format="phonopy_hdf5")
```

力应逐帧扣除，而不是拟合后再对参数做补偿。短程模型施加 ASR 与本身满足 ASR 的 Ewald 分量相容；两者相加后仍满足同一平移约束。示例中的数组导出仅支持 FC2。模型格式的导出范围见[力常数](force-constants.md)。

## 该计算的物理范围

Ewald 张量是有限超胞上的谐性偶极修正。它本身不添加极性声子能带中非解析 $q\to0$ LO-TO 分裂。因此，导出的 FC2 文件不能替代下游声子程序 NAC 功能所需的独立 Born 电荷/介电张量输入。

[NaCl 教学案例](../../tutorial/NaCl/README.md)包含直接拟合与长程扣除流程，以及声子谱对比。[拟合脚本](../../tutorial/NaCl/fit.py)读取 `BORN` 中的 Born 和介电张量，并生成 `fc2-direct.hdf5` 与 `fc2-corrected.hdf5`。
