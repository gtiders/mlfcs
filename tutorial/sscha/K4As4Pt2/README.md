# K4As4Pt2 SSCHA

300 K 的有效谐波 SSCHA 案例。它保持 $2\times2\times3$ reference 与 6 Å FC2 cutoff，
使用默认 Taylor 基、100 个快照、30 次更新和种子 42，从确定性的 Cartesian bootstrap 开始。

```bash
uv run --with pypolymlp python T300K/run.py
uv run --with phonopy --with seekpath --with matplotlib python T300K/plot.py
```
