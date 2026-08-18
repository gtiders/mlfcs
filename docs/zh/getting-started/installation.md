# 安装

MLFCS 要求 Python 3.12 或更高版本。

```bash
uv sync
```

已有环境也可以使用：

```bash
python -m pip install .
```

基础依赖是 ASE、NumPy、SciPy、spglib、h5py 和 JAX。具体 calculator 与后处理软件保持可选；
例如绘制 phonopy/SeeK-path 声子谱时再使用 `uv run --with phonopy --with seekpath`。
