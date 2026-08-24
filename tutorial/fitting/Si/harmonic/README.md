# Si FC2 Taylor 拟合

`input/train.extxyz` 包含 Si 的 64 原子随机位移训练构型。本任务只拟合 FC2，cutoff 为 5.4 Å，
并施加严格声学求和规则。

```bash
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

