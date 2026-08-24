# NaCl 长程静电力分解

本案例原样使用 hiPhive 官方 NaCl 长程修正数据，验证极性晶体中“先扣除解析长程力、拟合短程 FC2、再恢复长程 FC2 并开启 NAC”的完整闭环。

## 数据与模型

- 8 原子 rocksalt conventional cell；
- $4\times4\times4$ conventional reference，共 512 原子；
- 2 个由空间群约化得到的有限位移 DFT 构型；
- hiPhive 官方 `BORN` 中的电子介电张量和 Born 有效电荷；
- MLFCS 与 hiPhive 均使用 11 Å FC2 cutoff。

两帧数据是上游 phonopy 有限位移集合，不是从轨迹抽取的快照，也不应人为扩充为 Gaussian 数据。

## 运行

```bash
uv run python tutorial/long-range-electrostatics/NaCl/prepare.py
uv run python tutorial/long-range-electrostatics/NaCl/fit_mlfcs.py
uv run --with 'numpy<2.5' python tutorial/long-range-electrostatics/NaCl/run_hiphive.py \
  > tutorial/long-range-electrostatics/NaCl/hiphive.log 2>&1
uv run --with matplotlib --with seekpath --with phonopy python \
  tutorial/long-range-electrostatics/NaCl/plot.py
```

hiPhive 1.5 当前通过 numba 要求 NumPy 低于 2.5，因此其复现命令使用 `uv --with` 创建临时隔离环境；这不会修改 MLFCS 的项目环境或锁文件。

## 三种物理路径

直接拟合使用上游总力。长程修正路径先计算

$$
F_{\mathrm{LR}}=-\Phi_{\mathrm{LR}}u,
$$

再形成

$$
F_{\mathrm{SR}}=F_{\mathrm{total}}-F_{\mathrm{LR}}.
$$

短程拟合完成后恢复

$$
\Phi_{\mathrm{corrected}}
=
\Phi_{\mathrm{SR}}^{\mathrm{fit}}
+
\Phi_{\mathrm{LR}}.
$$

phonopy 参考、hiPhive 直接拟合、MLFCS 直接拟合和 MLFCS 修正结果最终都使用同一 `BORN` 开启 NAC，并沿同一 seekpath 路径绘图。

`long-range-electrostatics-comparison.png` 是本案例自己的独立图，不与 KCl SSCHA 图混合。只有 `band-metrics.json` 表明修正结果相对 phonopy 参考确实优于直接拟合时，才能把改善归因于长程静电分解。

本次复现得到：

- MLFCS 与 hiPhive 的总力拟合 RMSE 均为 $2.651939\times10^{-5}$ eV/Å；
- MLFCS 与 hiPhive 的短程力拟合 RMSE 均约为 $1.484223\times10^{-5}$ eV/Å；
- 两者恢复后的 FC2 相对 Frobenius 差异为 $4.27\times10^{-10}$；
- MLFCS 直接总力拟合相对 phonopy 的频率 RMS 差异为 $0.12244$ THz；
- 扣除并恢复长程项后降至 $0.02648$ THz，降低约 $78.37\%$。

因此，在这一同源 NaCl 数据集和固定 11 Å cutoff 下，声子谱误差的主要部分确实来自有限范围模型没有显式分离长程偶极响应，而不是 MLFCS 与 hiPhive 的 orbit 或线性拟合差异。

## 来源与引用

- [hiPhive examples：long-range corrections](https://gitlab.com/materials-modeling/hiphive-examples/-/tree/master/advanced/long_range_corrections)
- [hiPhive 长程力教程](https://hiphive.materialsmodeling.org/advanced_topics/long_range_forces.html)
- F. Eriksson, E. Fransson, P. Erhart, *The Hiphive Package for the Extraction of High-Order Force Constants by Machine Learning*, Advanced Theory and Simulations 2, 1800184 (2019).
- X. Gonze and C. Lee, *Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from density-functional perturbation theory*, Physical Review B 55, 10355 (1997).
