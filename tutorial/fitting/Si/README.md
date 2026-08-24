# Si 力常数拟合

本教程包含两条互相独立的 Si 拟合路线：`harmonic/` 仅拟合 FC2，`anharmonic/` 联合拟合
FC2、FC3 与 FC4。两者均使用由 ALAMODE 随机位移数据整理而成的 ASE `extxyz` 快照作为训练来源，
不把 ALAMODE 的 IFC 当作拟合真值。

每个子目录都有独立输入、拟合脚本、绘图脚本和覆盖写入的 `fit.log`。两条路线默认都使用 Taylor
基；生成的 HDF5 与外部格式 IFC 是可再生产物，不纳入 Git。

