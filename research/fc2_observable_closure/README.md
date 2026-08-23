# FC2 transferable map 与有限超胞 observable closure

本目录只研究当前 primitive/orbit FC2 realization map

$$
M:\Theta_{\mathrm{primitive}}\rightarrow\mathcal H_{\mathrm{SC}}
$$

是否存在

$$
\dim\ker M=0,
\qquad
\dim\operatorname{im}M<\dim\mathcal H_{\mathrm{SC}}.
$$

`prototype.py` 不修改或绕过正式实现。它以 KCl 的 2 原子 primitive 和
$2\times2\times2$ reference 为真实案例，构造完整 finite-supercell symmetry-allowed
compact Hessian basis，将当前 transferable FC2 投影到该空间，并用一次 SVD 构造
正交 complement。实际数据由案例已有 PolyMLP、100 个 $0.01$ Å Gaussian snapshots
和固定随机种子 42 生成。

运行：

```bash
uv run --with phonopy --with pypolymlp python research/fc2_observable_closure/prototype.py
```

`results.json` 保存 dimension、rank、奇异值、重建误差、实际 design rank 和最终
Go/No-Go 结论。KCl 的 representation 层命中 sweet spot，但去除质心位移的实际
design 只有 12/13 列秩，因此本轮结论为 **No-Go**。缺失方向违反 ASR，恰好属于
去质心采样无法观测的均匀平移响应；本轮不通过提前施加 ASR 来改变验收条件。

这里的 closure 只表示：

> finite-supercell harmonic response not represented by the current transferable FC2 basis

它不是 long-range FC2，也不对应唯一的无限晶格 interaction。
