# KCl SSCHA 超胞对照

本案例比较两个独立的 Taylor-SSCHA reference：$2\times2\times2$ 配合 6 Å cutoff，及
$4\times4\times4$ 配合 12 Å cutoff。两者都在 600 K 使用 100 个快照、50 次更新、种子 42，
从随机 Cartesian bootstrap 开始。

`4x4x4-electrostatic-subtracted/` 额外演示 hiPhive 长程修正教程采用的分层方法：phonopy
生成 Gonze–Lee dipole FC2，从 100 个 0.01 Å Gaussian 构型的 PolyMLP 总力中扣除对应
线性静电力，MLFCS 只拟合剩余短程 FC2，最后在有限超胞稠密 FC2 上重新加回 dipole 项。
图中所有曲线均关闭 NAC；因此这里比较的是有限超胞解析力常数，不包含额外的 $q\to0$
非解析 LO–TO 修正。

当前 PolyMLP 没有附带 DFPT Born 电荷和电子介电张量。探索性输入
`born-nominal.txt` 明确采用 $Z^*_{\mathrm K}=+1$、$Z^*_{\mathrm{Cl}}=-1$ 和
$\epsilon_\infty=2.365$ 的各向同性模型，不能替代与该势函数一致的第一性原理响应数据，
也不能作为正式 KCl 材料参数基线。静电力扣除流程参考
[hiPhive 长程相互作用教程](https://hiphive.materialsmodeling.org/advanced_topics/long_range_forces.html)，
$\epsilon_\infty$ 的示例值参考
[Togo 等人的 KCl/NaCl 研究](https://doi.org/10.1088/1361-648X/ac7b01)。

```bash
uv run --with pypolymlp python 2x2x2/run.py
uv run --with pypolymlp python 4x4x4/run.py
uv run --with phonopy --with pypolymlp python 4x4x4-electrostatic-subtracted/prepare.py
uv run --with phonopy python 4x4x4-electrostatic-subtracted/fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```
