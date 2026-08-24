# 数据来源与致谢

本目录保存 hiPhive 官方长程静电修正示例使用的 NaCl 输入数据：

- `NaCl_unitcell.xyz`：8 原子 rocksalt conventional cell；
- `supercells_with_forces.xyz`：两个 512 原子有限位移 DFT 构型及其力；
- `BORN`：电子介电张量和 Na、Cl 的 Born 有效电荷张量。

原始案例来自 [hiPhive examples](https://gitlab.com/materials-modeling/hiphive-examples/-/tree/master/advanced/long_range_corrections)，计算流程见 [hiPhive 长程力文档](https://hiphive.materialsmodeling.org/advanced_topics/long_range_forces.html)。上游仓库采用 [Mozilla Public License 2.0](https://gitlab.com/materials-modeling/hiphive-examples/-/blob/master/LICENSE)。使用这些数据时应同时引用 hiPhive 及其长程修正所依据的 Gonze–Lee 方法。

`prepare.py` 会逐项验证复制后的训练结构、位移和力，并把输入文件的 SHA256 写入 `preparation.json`。
