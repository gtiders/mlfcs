# MoS2 单层旋转约束

本案例以同一组显式位移/力数据分别进行 FC2 Taylor 拟合，并比较严格 ASR 与
Born–Huang/Huang 后处理对二维声子谱的影响。两个任务各自保存输入、`fit.py`、
`fit.log` 与 `metrics.json`。

```bash
uv run python asr/fit.py
uv run python born-huang-huang/fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```
