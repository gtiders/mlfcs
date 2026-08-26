---
title: ASE Calculator
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# ASE Calculator

`MLFCSCalculator` 将已经保存的 canonical Taylor 力常数解释为固定晶格上的多项式势能，提供
ASE 标准的相对能量与原子力接口。它不重新拟合，不加载 JAX，也不依赖训练数据。

## 完整签名

```python
MLFCSCalculator(
    force_constants: ForceConstants,
    *,
    reference: Atoms | None = None,
    maximum_displacement: float | None = None,
)

MLFCSCalculator.from_hdf5(
    source: str | Path,
    *,
    reference: Atoms | None = None,
    maximum_displacement: float | None = None,
) -> MLFCSCalculator
```

| 参数 | 含义 |
|---|---|
| `force_constants` | 包含 FC2 及以上 canonical Taylor IFC 的 `ForceConstants`。metadata 明确声明为其他基底时拒绝。 |
| `source` | MLFCS native HDF5 v3 文件。 |
| `reference` | Calculator 工作的固定显式超胞。省略时使用力常数当前 relation 的 reference；从 HDF5 读取时默认是 primitive。 |
| `maximum_displacement` | 可选的正数，单位 Å。任何原子的位移模超过它时发出 warning，但仍按原位移计算。 |

## 能量和力的语义

Calculator 计算

$$
\Delta E(\mathbf u)=
\sum_{n\ge2}\frac{1}{n!}\Phi^{(n)}\mathbf u^n,
$$

以及

$$
\mathbf F=-\frac{\partial\Delta E}{\partial\mathbf u}.
$$

这里固定 $E_0=0$、FC1 $=0$。因此 `get_potential_energy()` 返回相对于 reference 的能量，
不是 DFT 或势函数的绝对总能；reference 上的能量和力均为零。FC2、FC3、FC4、FC5 以及更高的
已保存 Taylor 阶数使用同一求值规则。

```python
from ase.io import read
from mlfcs import MLFCSCalculator

reference = read("supercell.vasp")
atoms = reference.copy()
atoms.positions[0, 0] += 0.01
atoms.calc = MLFCSCalculator.from_hdf5("mlfcs.h5", reference=reference)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## 固定边界

- 同一 Calculator 只接受与 reference 相同的原子数、元素顺序、晶格和 PBC。
- 周期边界内外的等价坐标使用 MIC 位移处理；不会静默重排原子。
- 可以在构造时将 exact-$R$ IFC realization 到其他合法整数超胞。
- 只实现 `energy` 和 `forces`；stress、virial、绝对能量和晶胞应变不在当前模型中。
- Calculator 表示保存后的 Taylor 多项式，不保证复现拟合前的 reference force。
