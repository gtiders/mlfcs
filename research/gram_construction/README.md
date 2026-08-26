# Gram 构造阶段优化研究

本研究比较当前 physical design 路径与 reduced-coordinate fused 路径。目标是缩短
完整 Gram 构造时间，而不是只优化 $X^T X$。候选实现的估算峰值工作内存不得超过
当前路径的三倍。

两条路径计算完全相同的统计量：

$$
X_r = XR, \qquad G = X_r^T X_r, \qquad b = X_r^T y.
$$

- `physical`：先 scatter 完整 $X$，再计算 $XR$。
- `fused`：每个 feature tile 直接乘对应的 $R$ 行，并累计到 $X_r$。

运行示例：

```text
uv run python research/gram_construction/prototype.py --frames 2
```

原型会报告 feature、scatter/reduction、BLAS、总时间、数值误差和工作内存估算。
研究代码不进入 `mlfcs` 运行时。

## Si FC2-FC4 结果

CPU JAX、单 BLAS 线程、8 个训练帧的独立进程结果：

| 路径 | batch | Gram 构造 | 每帧 | 最大 RSS |
|---|---:|---:|---:|---:|
| physical | 1 | 29.77 s | 3.72 s | 1,345,444 KiB |
| physical | 4 | 15.58 s | 1.95 s | 1,165,836 KiB |
| fused | 4 | 15.82 s | 1.98 s | 同一进程内 |

`batch=4` 相对 `batch=1` 将完整 Gram 构造缩短约 $47.7\%$，实测最大 RSS
下降约 $13.3\%$。该结果满足内存不得超过当前三倍的约束。

fused 与 physical 的 Gram 相对误差为 $5.3\times10^{-17}$，RHS 相对误差为
$4.9\times10^{-16}$，但 fused 没有速度收益。原因是 feature kernel 占据几乎全部
时间，而 tile-local reduction 增加了小矩阵乘法和累计开销。当前结论是：

- 增大 batch：Go，值得进入更大材料的验证。
- reduced-coordinate fused design：性能 No-Go，可保留为降低 workspace 的候选。
- tile-pair Direct-Gram：继续 No-Go。

这里的最大 RSS 包含结构枚举、JAX 编译、静态参数和两条 Gram 路径，因此是偏保守的
进程级测量。正式生产改动前还需在 SnSe 和教学大体系上验证 batch 4。
