# 平移与旋转求和规则

[English](SUM_RULES.md) | 中文

MLFCS 在对称性约化后的不可约轨道参数空间中施加物理求和规则。这些约束不同于张量
约化时已经使用的离散晶体旋转操作。

## 平移不变性

声学求和规则适用于当前支持的任意阶，并且默认开启：

```python
fc = calculation.reap(forces, acoustic_sum_rule=True)
```

程序报告投影前后的最大原子求和残差：

```text
- Max drift of fc3: 2.3410000000e-03 -> 7.1200000000e-12 eV/angstrom^3
```

## 二阶旋转不变性

对于 FC2，可选的 Born–Huang 旋转求和规则要求无穷小刚体旋转不产生恢复力。约束使用
周期最小镜像相对位置，默认关闭：

```python
fc2 = calculation.reap(
    forces,
    acoustic_sum_rule=True,
    rotational_sum_rule=True,
)
```

平移与旋转约束矩阵会堆叠后进行一次稀疏 LSMR 投影，避免一个投影破坏另一个。程序
分别报告两种残差；`verbose=False` 可关闭输出。

参考结构应充分弛豫。残余力、残余应力、过小的超胞或不完整的截断支撑域都可能使
旋转修正过大，甚至失去物理意义。

## 高阶限制

当前单阶 API 会拒绝在二阶以上设置 `rotational_sum_rule=True`。严格的高阶旋转恒等式
会耦合相邻阶，例如 FC3 与 FC2、FC4 与 FC3，因此需要未来的联合阶数约束接口。
MLFCS 不会把同阶近似表述为完整的高阶旋转不变性。
