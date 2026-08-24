# 石墨烯旋转约束

本案例从一个 phonopy 位移快照恢复位移和力，以 Taylor FC2 比较严格 ASR 与
Born–Huang/Huang 后处理。两个拟合任务彼此独立，均覆盖写入自己的 `fit.log`。

```bash
uv run python asr/fit.py
uv run python born-huang-huang/fit.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```
