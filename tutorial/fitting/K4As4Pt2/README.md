# K4As4Pt2 Taylor FC2–FC4 拟合

本教程以 ALAMODE 随机位移数据转换得到的 `train.extxyz`，拟合 FC2+FC3+FC4。默认使用
Taylor 基、三体 FC4 截断；FC2、FC3、FC4 cutoff 分别为 6.5 Å、$12a_0$、$8a_0$。

```bash
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```
