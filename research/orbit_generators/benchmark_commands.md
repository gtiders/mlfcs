# 串行 benchmark 命令

## 实验口径

- 每个阶数按连续阶数构造：FC3 = FC2+FC3，FC4 = FC2+FC3+FC4，依此类推。
- 只计对象初始化时间：MLFCS 的 `ForceConstantFitter(...)` 和 hiPhive 的 `ClusterSpace(...)`。
- 不计 primitive/reference 文件读取时间，不计超胞转换时间，不计 Gram、训练、拟合和后处理。
- 每次只运行一个命令；同一阶先运行 MLFCS，结束后再运行 hiPhive。
- 使用 `symprec=1e-4`。
- shell cutoff：FC2=`-8`，FC3=`-6`，FC4=`-3`，FC5=`-2`，FC6=`-2`。
- FC5/FC6 使用最大 3-body；hiPhive 对应 `CutoffMaximumBody(cutoffs, 3)`。

## Ba8Ga16Ge30

工作路径：`tutorial/Ba8Ga16Ge30/T300K/`

### 已完成

FC3（FC2+FC3）：

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case Ba8Ga16Ge30 --order 3 --output research/orbit_generators/mlfcs-Ba8Ga16Ge30-FC3.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case Ba8Ga16Ge30 --order 3 --output research/orbit_generators/hiphive-Ba8Ga16Ge30-FC3.json
```

结果：MLFCS 2.747 s；hiPhive 114.050 s。

FC4（FC2+FC3+FC4）：

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case Ba8Ga16Ge30 --order 4 --output research/orbit_generators/mlfcs-Ba8Ga16Ge30-FC4.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case Ba8Ga16Ge30 --order 4 --output research/orbit_generators/hiphive-Ba8Ga16Ge30-FC4.json
```

结果：MLFCS 6.952 s；hiPhive 525.804 s。

### 以后运行

FC5（FC2+FC3+FC4+FC5，FC5 最大 3-body）：

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case Ba8Ga16Ge30 --order 5 --output research/orbit_generators/mlfcs-Ba8Ga16Ge30-FC5.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case Ba8Ga16Ge30 --order 5 --output research/orbit_generators/hiphive-Ba8Ga16Ge30-FC5.json
```

FC6（FC2+FC3+FC4+FC5+FC6，FC5/FC6 最大 3-body）：

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case Ba8Ga16Ge30 --order 6 --output research/orbit_generators/mlfcs-Ba8Ga16Ge30-FC6.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case Ba8Ga16Ge30 --order 6 --output research/orbit_generators/hiphive-Ba8Ga16Ge30-FC6.json
```

## SnSe

工作路径：`tutorial/SnSe/harmonic-2x4x4/`

### FC3

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe --order 3 --output research/orbit_generators/mlfcs-SnSe-FC3.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe --order 3 --output research/orbit_generators/hiphive-SnSe-FC3.json
```

### FC4

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe --order 4 --output research/orbit_generators/mlfcs-SnSe-FC4.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe --order 4 --output research/orbit_generators/hiphive-SnSe-FC4.json
```

### FC5

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe --order 5 --output research/orbit_generators/mlfcs-SnSe-FC5.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe --order 5 --output research/orbit_generators/hiphive-SnSe-FC5.json
```

### FC6

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe --order 6 --output research/orbit_generators/mlfcs-SnSe-FC6.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe --order 6 --output research/orbit_generators/hiphive-SnSe-FC6.json
```

## 结果记录

不要用 `&&` 同时提交多个后台任务。每一对命令应确认前一个进程结束后再执行后一个。结果 JSON 中的 `initialization_seconds` 是本实验的主要比较指标。

## SnSe expanded primitive（探索性）

这里将原始 SnSe primitive 用 `build_supercell(..., (2, 2, 2))` 扩为 64 原子后作为 MLFCS 输入。原 `2x4x4` reference 对应为 expanded primitive 的 `1x2x2` reference。shell cutoff 仍为 FC2=`-8`、FC3=`-6`、FC4=`-3`、FC5/FC6=`-2`。

### 已完成

FC3：MLFCS 1.815 s；hiPhive 0.347 s。

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe-expanded-2x2x2 --order 3 --output research/orbit_generators/mlfcs-SnSe-expanded-2x2x2-FC3.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe-expanded-2x2x2 --order 3 --output research/orbit_generators/hiphive-SnSe-expanded-2x2x2-FC3.json
```

FC4：MLFCS 3.247 s；hiPhive 0.793 s。

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe-expanded-2x2x2 --order 4 --output research/orbit_generators/mlfcs-SnSe-expanded-2x2x2-FC4.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe-expanded-2x2x2 --order 4 --output research/orbit_generators/hiphive-SnSe-expanded-2x2x2-FC4.json
```

FC5：MLFCS 7.951 s；hiPhive 1.642 s。

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe-expanded-2x2x2 --order 5 --output research/orbit_generators/mlfcs-SnSe-expanded-2x2x2-FC5.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe-expanded-2x2x2 --order 5 --output research/orbit_generators/hiphive-SnSe-expanded-2x2x2-FC5.json
```

FC6：MLFCS 39.100 s；hiPhive 9.771 s。

```bash
uv run python research/orbit_generators/benchmark_tutorial_materials.py --case SnSe-expanded-2x2x2 --order 6 --output research/orbit_generators/mlfcs-SnSe-expanded-2x2x2-FC6.json
uv run --with 'numpy<2.5' --with hiphive python research/orbit_generators/benchmark_hiphive_tutorial.py --case SnSe-expanded-2x2x2 --order 6 --output research/orbit_generators/hiphive-SnSe-expanded-2x2x2-FC6.json
```

限制：hiPhive 的 `ClusterSpace` 会自动识别 expanded structure 的标准 8 原子 Pnma primitive，因此其内部实际仍按 8 原子 primitive 构造。该组数据不能作为 64 原子 primitive 对 64 原子 primitive 的严格公平比较；它只能说明把 expanded structure 传入两套接口后的行为差异。
