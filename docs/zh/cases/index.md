# 示例

[English] | 中文

材料案例是可复现的工作流夹具。每个案例的 README 是命令、输入来源、预期输出和第三方参考数据的唯一说明。

## 材料案例

- [Si](https://github.com/gtiders/mlfcs/tree/main/examples/finite-difference/Si/README.md)：有限差分和仅力数据拟合。
- [K4As4Pt2](https://github.com/gtiders/mlfcs/tree/main/examples/fitting/K4As4Pt2/README.md)：FC2--FC4 拟合和 loop-SCPH。
- [Ba8Ga16Ge30](https://github.com/gtiders/mlfcs/tree/main/examples/fitting/Ba8Ga16Ge30/README.md)：公开 hiPhive 训练数据。
- [KCl](https://github.com/gtiders/mlfcs/tree/main/examples/sscha/KCl/README.md)：原生 SSCHA 参考。
- [MoS2 和 graphene](https://github.com/gtiders/mlfcs/tree/main/examples/fitting/MoS2_monolayer/README.md)：二阶旋转约束。

本目录脚本演示 MLFCS 公共 API，请在仓库根目录通过 `uv run` 执行。它们只是示例，
不表示 MLFCS 恢复了命令行接口。
新增脚本和材料案例应遵循[测试与案例设计原则]。

## 直接使用 calculator

- [`basic_fc2.py`] 使用 ASE EMT 弛豫并计算小体系，然后写出 FC2；
- [`nep89_orders.py`] 通过 calorine 加载用户提供的 NEP89 模型，计算
  一个或多个阶数。势函数仍由用户安装和管理。

例如：

```bash
uv run python examples/nep89_orders.py POSCAR nep89.txt \
  --orders 2 3 --supercell 2 2 2 --cutoff -3 --output-directory results
```

## 外部 VASP 计算

[`vasp_external_fc3.py`] 给出完整的三阶段案例：

```bash
uv run python examples/vasp_external_fc3.py sow POSCAR fc3-work \
  --supercell 3 3 3 --cutoff -6

# 建立 calculations/POSCAR-001、calculations/POSCAR-002……；
# 复制对应 POSCAR，提交 VASP，并保留 vasprun.xml。

uv run python examples/vasp_external_fc3.py collect \
  fc3-work fc3-work/calculations
uv run python examples/vasp_external_fc3.py reap \
  fc3-work FORCE_CONSTANTS_3RD --format shengbte
```

该脚本不会提交 VASP。INCAR、KPOINTS、POTCAR 和调度脚本仍由用户根据计算环境负责。
在使用外部力之前请阅读[完整工作流]。
