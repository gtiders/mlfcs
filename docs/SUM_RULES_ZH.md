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
- ASR parameter correction: maximum=8.4100000000e-04 eV/angstrom^3, relative L2=1.7300000000e-03
```

drift 衡量求和规则违反程度，correction 衡量为了满足约束实际修改了多少 IFC 参数。
最终 drift 很小并不代表参数修正也很小，因此截断支撑域不完整时应同时检查这两类数值。
平移与旋转规则共同开启时，修正量会明确标记为二者的联合投影。

## 二阶旋转不变性

旋转不变性是一条从 FC1–FC2 开始的相邻阶层级。有限差分没有待拟合的 FC1 未知量，
并把充分弛豫的参考构型视为 FC1=0；最低层恒等式随即化为 Born–Huang FC2 条件，即
无穷小刚体旋转不产生谐波恢复力。约束使用周期最小镜像相对位置，默认关闭：

```python
fc2 = calculation.reap(
    forces,
    acoustic_sum_rule=True,
    rotational_sum_rule=True,
)
```

平移与旋转约束矩阵会堆叠后进行一次稀疏 LSMR 投影，避免一个投影破坏另一个。程序
分别报告两种残差；`verbose=False` 可关闭输出。

FC1=0 是有限差分边界条件，而不是从位移力中估计的结果。明显的参考残余力说明所选
展开原点与该边界不一致。参考结构应充分弛豫。残余力、残余应力、过小的超胞或不完整
的截断支撑域都可能使
旋转修正过大，甚至失去物理意义。

## 高阶限制

当前单阶 API 会拒绝在二阶以上设置 `rotational_sum_rule=True`。严格的高阶旋转恒等式
会耦合相邻阶，例如 FC3 与 FC2、FC4 与 FC3。独立的 `mlfcs.fitting` 开发接口通过
`rotational_invariance=2` 或 `3` 提供联合阶数约束。Wick 拟合可在所选参考构型产生真实
的 Taylor FC1，因此两种模式都施加完整的拟合 FC1–FC2 恒等式，而不把 FC1 强制为零；
同时包含所有已经表示的相邻阶恒等式。模式 2 仅开放模型最高阶之上的边界，模式 3 还假设该下一
阶贡献为零。由于该拟合器内部使用Wick多项式，
求解前会按 `C_W = C_T @ T(Sigma)` 把Taylor旋转约束映射到Wick坐标，输出时使用同一
换基映射。有限差分单阶API的限制保持不变。
