# NaCl 教学数据来源

数据下载自 [hiPhive 官方仓库](https://gitlab.com/materials-modeling/hiphive) 的 `examples/advanced_topics/long_range_forces`，提交 `04a715ea641189ca8af34888737ec8a1377e536f`。

- `NaCl_unitcell.xyz`：8 原子的常规晶胞。
- `supercells_with_forces.xyz`：两帧 512 原子超胞的位移与预计算力。
- `BORN`：Born 有效电荷、介电张量及上游单位因子。
- `LICENSE`：上游 MIT 许可证，原文保留。

三个原始数据文件未修改。我们的 notebook 使用 ASE 的 Coulomb 单位因子，与 MLFCS Ewald 保持一致；上游 BORN 中该因子写作四舍五入后的 `14.400`。

MLFCS 教程代码重新编写，没有引入 hiPhive 生产依赖。上游演示见 [Dealing with long-range interactions](https://hiphive.materialsmodeling.org/advanced_topics/long_range_forces.html)。
