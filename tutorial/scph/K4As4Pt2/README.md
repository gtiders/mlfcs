# K4As4Pt2 loop-SCPH

先运行 `prepare_ifcs.py`，它以默认 Taylor 基拟合单一 FC2+FC3+FC4 模型。随后 `run.py`
从这一同源模型中同时读取 FC2（初始谐波）与 FC4（loop 修正），在 300、600、900 K 运行
loop-SCPH。

```bash
uv run python prepare_ifcs.py
uv run python run.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

`fit.log` 与 `run.log` 分别保留拟合和 SCPH 的完整过程；HDF5、文本 IFC 与频率缓存由脚本
重建，不纳入 Git。`plot.py` 以同一个 `source/mlfcs.h5` 的 FC2 作为谐波基线，并叠加各温度的
有效 FC2 声子谱。
