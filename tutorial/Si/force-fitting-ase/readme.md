# Si 二阶力常数拟合教程

本案例使用与有限差分案例相同的 Si NEP 势，但 FC2 不通过有限差分得到，而是使用带原子力的结构快照进行力常数拟合。NEP 模型来自相关论文作者维护的 `nep-data` 仓库；本教程只使用模型计算原子力，不训练或修改该模型。

必须在当前目录运行：

```bash
cd tutorial/Si/force-fitting-ase
uv run python run.py
uv run --with phonopy --with seekpath --with matplotlib python plot.py
uv run python calculator.py
```

脚本会依次完成：

1. 从 `POSCAR.vasp` 读取 Si 原胞并构造 4×4×4、128 原子超胞。
2. 使用顶层 `perturb_structures()` 在没有初始 FC2 的情况下生成 3 个去质心高斯位移结构，`displacement=0.01` 的单位是 Å。
3. 使用 `CPUNEP` 计算 3 个结构的原子力，清除 MLFCS 内部元数据，只保留结构和 `forces` 后写入 `train.extxyz`。
4. 重新读取 `train.extxyz`，使用 `ForceConstantFitter` 拟合 FC2。
5. 输出 `SPOSCAR`、拟合力常数和 `fit-metrics.json`。
6. 运行 `plot.py`，读取 `SPOSCAR` 和 `force_constants-fit.hdf5`，输出 `fit-phonon-band.png` 与 `fit-phonon-band.json`。
7. 运行 `calculator.py`，从 native HDF5 构造 `MLFCSCalculator`，并用数值能量梯度验证解析力。

这里的 `perturb_structures()` 只负责生成结构，不计算力，也不执行 SSCHA 自洽迭代。本教程显式使用
`method="gaussian"`；若要按 FC2 与温度生成谐系综，必须显式改用 `method="harmonic"` 并同时提供
`force_constants` 和 `temperature`。运行日志固定覆盖写入 `fit.log`。

生成的 `train.extxyz` 包含每个结构的原子力，因此可以独立读取并用于拟合：

```python
from ase.io import read
from mlfcs import ForceConstantFitter

training = read("train.extxyz", index=":")
fitter = ForceConstantFitter(
    primitive,
    reference,
    orders=(2,),
    cutoffs={2: None},
)
result = fitter.fit(training, validation_split=0.0)
```

普通 Python 用户可以将命令替换为：

```bash
python run.py
python plot.py
```

如果提示缺少教程额外依赖，在仓库根目录执行：

```bash
uv pip install calorine phonopy seekpath matplotlib
```
