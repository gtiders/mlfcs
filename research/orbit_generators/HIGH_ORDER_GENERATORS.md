# 高阶空间群生成元 orbit 方案

## 定位

空间群生成元遍历不是 FC2–FC4 生产路径的替代方案，但对 FC6 及更高阶具有明确研究价值。

当前生产实现对每个代表 interaction 遍历全部空间群操作与全部 IFC 指标排列。若空间群大小为 $|G|$、力常数阶数为 $n$，其 key-action 次数为

$$
N_{\mathrm{orbit}}|G|n!.
$$

生成元方案改为使用少量空间群生成元和 $S_n$ 的相邻换位生成元，在实际 orbit image 图上执行 BFS。其工作量近似为

$$
N_{\mathrm{image}}N_{\mathrm{generator}}.
$$

低阶时，BFS 的队列和哈希成本会抵消减少的群操作；随着 $n!$ 增长，生成元遍历可能明显占优。

## 已完成验证

原型已对 Si、SnSe 和 Ba8Ga16Ge30 的 FC2–FC4 验证：

- exact image-key sets 与生产实现完全一致；
- invariant tensor dimension 完全一致；
- invariant projector 误差为 $10^{-15}$ 量级；
- FC2–FC4 多数案例没有性能收益，因此不修改现有生产路径。

进一步对 Si 进行纯 interaction-key 高阶枚举，使用 4.6 Å cutoff 和最大 3-body：

| 阶数 | 代表簇 | images | 完整枚举时间 | 生成元 BFS 时间 | 结果 |
|---:|---:|---:|---:|---:|---|
| FC5 | 29 | 17642 | 1.61 s | 1.68 s | 无收益 |
| FC6 | 44 | 62218 | 16.80 s | 7.70 s | 快约 2.18 倍 |

两条路径产生相同的 image 总数。FC6 的峰值 RSS 约为 122 MiB，并且基准未进入拟合。

## 有价值的后续方向

该方案标记为 **FC6+ 有价值候选**，但不能直接进入生产代码。下一阶段必须同时解决：

1. 利用 Schreier edges 构造 stabilizer constraints，并验证 invariant subspace；
2. 避免显式生成巨大的 rank-6 Kronecker action 中间量；
3. 比较 tensor invariant 构造后的总时间与峰值内存，而不只比较 key traversal；
4. 验证 pivot normalization、参数顺序和最终展开 IFC；
5. 继续严格隔离 design、Gram 和拟合，直到 orbit 层原型通过。

进入生产实现的最低条件为：FC6 完整 orbit 构造数值严格等价、总时间至少加速 1.5 倍、峰值内存不增加，并且不为 FC2–FC4 引入运行时双分支。更合理的接入方式是按阶数选择统一实现边界，例如 FC2–FC5 保持完整操作表，FC6+ 使用生成元算法；该边界必须由完整 tensor benchmark 决定，不能仅依据当前 key-only 数据硬编码。

## 当前结论

- FC2–FC4：生产替换为 No-Go。
- FC5：当前数据为 No-Go。
- FC6+：有价值，建议继续完整 tensor invariant prototype。
- 当前不进入拟合、不改变公共 API、不改变 IFC 数据模型。

