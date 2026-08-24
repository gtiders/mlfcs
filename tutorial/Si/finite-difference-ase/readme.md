## NEP 势来源与致谢

本教程借助相关论文发布的 Si NEP 势计算原子力。NEP 势及其训练数据、`nep` 程序的输入输出和完整发布信息见作者维护的 [nep-data 仓库](https://gitlab.com/brucefan1983/nep-data)。本项目不开发、不训练也不重新发布该势；这里仅使用其中的模型文件演示 MLFCS 的有限差分工作流。

该势的训练数据和方法来源于 Albert P. Bartók 等人的工作，NEP 模型和 GPUMD 工具来源于 Zheyong Fan 等人的工作。使用本教程中的模型时，请同时遵循原作者仓库和论文中的许可、引用及使用说明。

| NEP 模型文件 | 训练数据论文 | NEP 模型论文 | 用途 |
| --------- | -------------------------------------- | -------------------------------------- | ------------------- |
| `Si_2022_NEP3_3body.txt`、`Si_2022_NEP3_4body.txt`、`Si_2022_NEP3_5body.txt` | Albert P. Bartók et al., [Machine Learning a General-Purpose Interatomic Potential for Silicon](https://doi.org/10.1103/PhysRevX.8.041048), Phys. Rev. X **8**, 041048 (2018) | Zheyong Fan et al., [GPUMD: A package for constructing accurate machine-learned potentials and performing highly efficient atomistic simulations](https://aip.scitation.org/doi/10.1063/5.0106617), The Journal of Chemical Physics **157**, 114801 (2022) | 本教程中的 Si 原子力计算 |

## 运行流程

必须在当前目录内运行脚本，因为脚本使用本目录下的相对路径。

第一步运行有限差分计算。它读取 `POSCAR.vasp` 和 NEP 模型，生成 4×4×4 参考超胞，写出 `SPOSCAR`，计算原子力并导出三种 FC2 文件。请先直接运行；如果提示缺少依赖，再按后面的说明安装：

```bash
cd tutorial/Si/finite-difference-ase
uv run python run.py
```

可选的零步长外推路线使用同一个 `FiniteDifferenceCalculation`，在五个位移步长上重建导数：

```bash
uv run python run_extrapolation.py
```

普通中心差分的完整日志固定写入 `run.log`，外推路线固定写入 `extrapolation.log`；重复执行会覆盖旧日志。

普通 Python 用户可以使用：

```bash
python run.py
```

第二步绘制声子谱。必须先完成第一步，因为 `plot.py` 读取上一步生成的 `SPOSCAR` 和 `force_constants.hdf5`。它使用 seekpath 生成高对称路径、phonopy 计算声子频率，并输出 `phonon-band.png` 与路径信息 `phonon-band.json`：

```bash
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

普通 Python 用户可以使用：

```bash
python plot.py
```

## 缺少依赖时

开发者使用 `uv` 时，先在仓库根目录安装教程额外依赖：

```bash
uv pip install calorine phonopy seekpath matplotlib
```

安装完成后，回到本教程目录，按上面的顺序运行：

```bash
cd /home/gwins/codespace/mlfcs-new/tutorial/Si/finite-difference-ase
uv run python run.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
```

不使用 `uv` 时，可以用当前 Python 环境的 pip 安装同样的依赖：

```bash
python -m pip install calorine phonopy seekpath matplotlib
```
