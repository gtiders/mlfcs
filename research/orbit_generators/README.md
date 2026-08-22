# Space-group generator orbit 原型

## 目标

生产实现目前对每个 interaction representative 遍历全部空间群操作和全部 $n!$ 个 IFC 指标置换，再用字典去重 image，并收集所有 stabilizer constraints。

本原型验证能否把它改写为标准有限群作用：

1. 从 spglib 返回的完整仿射操作表中选出一个确定性小生成集；
2. 使用空间群生成元和对称群 $S_n$ 的相邻换位生成元遍历 interaction-key orbit；
3. 对每条指向已访问 image 的 Schreier edge，构造

   $$
   C_{x,s}=A_sB_x-B_{s x};
   $$

4. 由

   $$
   G=\sum_{x,s}C_{x,s}^TC_{x,s}
   $$

   的零空间得到 representative 的 invariant tensor subspace；
5. 与生产实现的全部 image keys、参数维数和 invariant projector 严格比较。

该原型不修改生产 orbit 构造，不改变 IFC 参数编号，也不进入公共 API。

## 运行

```bash
uv run python research/orbit_generators/prototype.py
```

也可以只运行一个案例：

```bash
uv run python research/orbit_generators/prototype.py si
```

## Go / No-Go 判据

只有同时满足以下条件才制定生产迁移计划：

- Si、SnSe 和 Ba8Ga16Ge30 的 exact image-key sets 完全一致；
- 每个 orbit 的 invariant dimension 完全一致；
- invariant projector 误差不超过 $10^{-9}$；
- generator-edge 数显著少于 $|G|n!$ 全枚举；
- 不需要维护完整群元素和生成元两条运行时路径。

若 image orbit 很小但 stabilizer Schreier edges 仍接近全枚举，或生成元路径的 Python traversal 抵消理论收益，则保留现有生产算法。

## 实验结果

| 案例 | 阶数 | orbit | 参数 | 当前全枚举 actions | generator edges | 最大 projector 误差 |
|---|---:|---:|---:|---:|---:|---:|
| Si | 2 | 4 | 11 | 384 | 174 | $1.60\times10^{-15}$ |
| Si | 3 | 10 | 95 | 2880 | 3368 | $2.60\times10^{-15}$ |
| Si | 4 | 18 | 468 | 20736 | 22130 | $1.52\times10^{-15}$ |
| SnSe | 2 | 55 | 400 | 880 | 2656 | $7.22\times10^{-16}$ |
| SnSe | 3 | 204 | 4354 | 9792 | 36400 | $7.22\times10^{-16}$ |
| SnSe | 4 | 99 | 3818 | 19008 | 36864 | $1.35\times10^{-15}$ |
| Ba8Ga16Ge30 | 2 | 215 | 1809 | 1290 | 2376 | $6.57\times10^{-16}$ |
| Ba8Ga16Ge30 | 3 | 382 | 6520 | 6876 | 9666 | $1.49\times10^{-15}$ |
| Ba8Ga16Ge30 | 4 | 562 | 17166 | 40464 | 29784 | $1.15\times10^{-15}$ |

三个案例的 image-key sets 和 invariant dimensions 全部严格一致，说明 Schreier-edge 数学构造正确。但除 Si FC2 和 Ba FC4 外，generator traversal 需要访问更多 edge；SnSe FC3/FC4 分别约为当前全枚举的 $3.72$ 和 $1.94$ 倍。

## 结论

**No-Go：不以 generator traversal 整体替换生产 orbit image 构造。**

原因不是数学不成立，而是当前空间群最多只有 48 个操作、目标阶数主要为 FC2–FC4，直接 $|G|n!$ 枚举很小；generator BFS 必须为每个已经生成的 image 再访问全部生成元，并承担字典查询和 queue 操作，amortized cost 更高。

Schreier constraints 数量确实明显少于完整 stabilizer elements，但生产实现本来就在同一轮 image 枚举中顺便累积小型 Gram，稳定子约束不是独立瓶颈。只替换这一部分也不足以抵消新增的群闭包、transversal 和 BFS 逻辑。

原型保留为数学验证和未来 FC5 以上的研究工具；生产代码保持当前完整操作表枚举，不加入双运行时分支。

## FC5/FC6 枚举基准

高阶基准只枚举 interaction keys，对完整群操作枚举和生成元 BFS 计时。它不构造 Cartesian tensor basis、realization、design、Gram 或拟合对象，因此不会把 $3^5$ 和 $3^6$ 维张量中间量带入拟合路径。

```bash
uv run python research/orbit_generators/benchmark_high_order.py 5
uv run python research/orbit_generators/benchmark_high_order.py 6
```

Si 使用 4.6 Å cutoff 和最大 3-body 的实测结果：

| 阶数 | 候选 cluster | 代表簇 | images | cluster 时间 | 全枚举时间 | 生成元时间 | 生成元/全枚举 | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FC5 | 2242 | 29 | 17642 | 6.98 s | 1.61 s | 1.68 s | 1.046 | 119 MiB |
| FC6 | 3642 | 44 | 62218 | 15.89 s | 16.80 s | 7.70 s | 0.458 | 122 MiB |

FC5 仍没有收益；FC6 的 key traversal 中，生成元 BFS 比完整 $48\times6!$ 操作枚举快约 2.18 倍。该结论只适用于 interaction-key 枚举：基准刻意没有构造 rank-5/rank-6 Cartesian tensor basis 和 stabilizer 数值核，因此不足以推翻 FC2–FC4 的生产 No-Go 结论，也不构成接入拟合流程的依据。若未来正式支持 FC6，应单独研究张量 invariant 构造，避免生成完整高阶 Kronecker 中间量。
