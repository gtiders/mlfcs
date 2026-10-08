# 有限差分 API

`FiniteDifference` 为一个力常数阶次生成确定顺序的 ASE 位移结构。力计算属于外部流程；计算完后统一通过 `ForceDataset` 收集，再重建。

## 超胞与采样

```python
from ase.build import bulk
from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, ForceDataset

primitive_atoms = bulk("Al", "fcc", a=4.05)
space = ClusterSpace(primitive_atoms, cutoffs={2: 4.0}, max_body_orders={2: 2})
mapping = ClusterMap(space, primitive_atoms.repeat((3, 3, 3)))
fd = FiniteDifference(mapping, order=2, disps=(0.01, 0.02))
```

`order` 至少为 2，且存在于模型中；对应阶次的超胞映射必须能区分参数。`disps` 是一个正位移长度或一组互不相同的正长度，单位 Å。多步幅度按偶次误差外推到零位移。

`fd.displacements()` 返回懒加载、有序的 Atoms 序列，不附加采样 ID、位移标签或其他 metadata。固定顺序为：位移键、升序位移幅度、符号组合。重复的原子/方向将带符号位移相加。

## 外部 ASE 计算

```python
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

calculator = EMT()
evaluated = []
for atoms in fd.displacements():
    atoms.calc = calculator
    forces = atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, forces=forces)
    evaluated.append(atoms)

data = ForceDataset(mapping, evaluated)
fc2 = fd.reconstruct(data)
```

不存在 `fd.evaluate()`。也可将位移结构送入 VASP 等外部程序，再按原顺序读回带力 Atoms。数据集只读取已有力，不启动计算器；没有 Atoms 便利重建入口。

## 力扣除与重建

```python
# 参考构型的力广播给每一帧。
centered = data.subtract_forces(supercell_forces)
fc2 = fd.reconstruct(centered)

# 若使用长程模型，则按每帧位移计算并扣除。
short_data = data.subtract_forces(long_range.forces(data.displacements))
short_fc2 = fd.reconstruct(short_data)
```

`reconstruct()` 只检查数据集类型、力的形状、数量与有限值。不核对预期位移，不检查帧是否重排，不查找 metadata。相同长度的错误顺序不会被发现；调用者必须保持采样顺序、原子顺序，以及原胞参数布局一致。

中央混合差分把力对位移的导数转为能量导数，再通过 orbit observation matrix 恢复参数。结果只包含所选阶次。不同阶次可用 `ForceConstants.combine` 合并；`save()` 保存参数化模型，`write()` 导出文件。

更多数据和长程约定见[数据集与 Ewald](dataset-api.md)。

## 使用初始化 ASR 自由坐标

若 mapping 的空间由 `ClusterSpace(..., asr=True)` 构造，差分采样仍保持当前数量与顺序。
`reconstruct()` 先完成全部中央差分和步长外推，再通过隐式 ASR 坐标作受约束重建。
因此结果满足 ASR，而采样数尚未减少。有噪声时，各 orbit 的观察结果会共同决定
约束子空间内的最小二乘解，结果可以不同于逐 orbit 独立恢复。

该重建通过矩阵自由 lift/adjoint 和 LSMR 完成，不物化全局稠密 nullspace。
具体语义和后续范围见[初始化阶段 ASR](asr-nullspace.md)。
