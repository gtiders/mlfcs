# SnSe 拟合与谐波超胞对照

本教学案例包含三个相互独立的 Taylor 基拟合任务：

- `joint-fc234/`：使用已有 300 K NVE 数据联合拟合 FC2、FC3 和 FC4；
- `harmonic-2x4x4/`：使用 $2\times4\times4$ 超胞的 10 个 Gaussian 位移拟合 FC2；
- `harmonic-3x5x5/`：在现有超胞的三个方向都增加一个 primitive cell，再独立拟合 FC2。

两个谐波任务都使用 0.01 Å Cartesian Gaussian 位移、随机种子 42、`cutoff=None`
和严格声学求和规则。它们使用相同的公开 hiPhive FCP 计算力，但不共享运行脚本或训练数据。

## 运行顺序

联合拟合：

```bash
cd tutorial/SnSe/joint-fc234
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

两个谐波任务分别运行：

```bash
cd tutorial/SnSe/harmonic-2x4x4
uv run --with hiphive python prepare.py
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py

cd ../harmonic-3x5x5
uv run --with hiphive python prepare.py
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

最后在案例根目录绘制超胞对照：

```bash
cd tutorial/SnSe
uv run --with phonopy --with seekpath --with matplotlib python compare_harmonic.py
```

每个拟合任务覆盖写入自己的 `fit.log`。HDF5、外部 IFC 文本和缓存是可再生计算产物，
不进入 Git；结构、训练数据、日志、指标和图片作为教学证据保留。
