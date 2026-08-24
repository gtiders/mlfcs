# Si FC2–FC4 Taylor 联合拟合

本任务在同一批 100 个 64 原子随机位移构型上联合拟合 FC2、FC3 与 FC4。FC2、FC3 cutoff 均为
5.4 Å；FC4 cutoff 为 4.6 Å，对应第三近邻壳层与第四近邻壳层之间的安全截断。FC2 限制为二体，
FC3 与 FC4 限制为三体。

```bash
uv run python fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

