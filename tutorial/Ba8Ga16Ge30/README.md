# Ba8Ga16Ge30 温度相关有效力常数

本教程使用 hiPhive 公开的 Ba8Ga16Ge30 clathrate force-constant potential 生成的固定温度
NVE 快照，分别拟合 300、400、500 和 600 K 的有效 FC2+FC3。每个温度目录都是独立任务：
它包含自己的 primitive、supercell、训练快照、拟合脚本、绘图脚本和结果。

MLFCS 使用默认 Taylor 基拟合；FC2、FC3 截断半径为 5.4、4.35 Å，二者均限制为 2-body。
本教程保留 `nve.extxyz` 作为直接可重现拟合的数据，不保留原始 MD 轨迹。

## 目录

- `input/`：仅保存上游 FCP 势函数与来源、致谢说明；
- `T300K/`、`T400K/`、`T500K/`、`T600K/`：该温度的完整拟合任务；
- 每个温度目录中的 `supercell.vasp` 是训练使用的 reference supercell；
- `fit.log` 是对应拟合的完整 stdout、stderr 与 traceback 记录；每次拟合会覆盖它，属于案例结果。

## 运行

在仓库根目录执行，例如 300 K：

```bash
uv run python tutorial/Ba8Ga16Ge30/T300K/fit.py
uv run --with phonopy --with seekpath --with matplotlib \
  python tutorial/Ba8Ga16Ge30/T300K/plot.py
```

其他温度只需替换目录名。拟合会在本温度目录写出 native HDF5、phonopy FC2 文本、
phonopy HDF5、ShengBTE FC3 文本、`metrics.json` 和完整时间戳日志；绘图写出
`phonon-band.png`。

四个温度目录均保留由当前代码重新拟合得到的 `metrics.json`、声子谱图与完整的
`fit.log`。后续重新运行时，应以新产生的结果覆盖它们。
