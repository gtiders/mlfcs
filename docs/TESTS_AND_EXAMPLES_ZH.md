# 测试与案例设计原则

[English](TESTS_AND_EXAMPLES.md) | 中文

## 目标布局

MLFCS 后续只保留两个验证入口：

| 位置 | 责任 | 普通 CI |
|---|---|---|
| `tests/` | 确定性的数学、API、不变量和最小格式契约 | 必须运行 |
| `examples/` | 公共 API 示例与真实材料科学案例 | 只做轻量检查 |

现有 `assets/` 将合并到 `examples/cases/`。外部软件的输入、结果和训练数据可以作为案例
参考继续保留，但不再作为 pytest 的严格数值 oracle。案例也不强制 `case.json`、SHA-256
清单或逐文件 hash 对比；每个案例只强制一个完整的 `README.md` 作为来源和方法说明。

## `tests/` 的边界

### 应保留

- `tests/unit/`：整数超胞、周期 residue、MIC、轨道、有限差分模板、Wick 换基、约束、
  稀疏 IFC 和 writer 语义；
- `tests/integration/`：只通过公共 API 运行最小 `sow → reap`、拟合、HDF5 往返、SSCHA
  和目标结构导出；
- 手算模型、自动微分独立能量、算法收敛阶数、守恒律和结构表示不变量；
- 可选的官方 reader 最小验收，只证明文件可读、尺寸和标签正确。

### 不再新增

- 不把 phonopy、phono3py、ALAMODE、hiphive、ShengBTE 等完整材料结果当作真值数组；
- 不在 tests 中保留大型训练集、热输运目录、外部势函数训练或完整材料拟合；
- 不依靠外部软件当前版本的具体浮点结果决定普通 CI 是否通过；
- 不把旧版 MLFCS 输出作为新版 oracle。

外部案例发现 bug 后，应在 tests 中构造最小独立回归。例如 ShengBTE Si 热导率暴露周期
像错误后，tests 应验证最小简并像和 block 语义，而不是每次 CI 重跑整套 Si 热导率。

测试必须固定随机种子，采用能暴露问题的最小结构，并从量纲、算法阶数或浮点误差确定
容差。不得读取 `examples/cases/`，不得依赖测试执行顺序或开发者机器绝对路径。

## `examples/` 的两种内容

### 通用脚本

`examples/*.py` 演示一个最小公共 API 任务，例如 FC2、外部 `sow/collect/reap` 或声子谱
绘制。它们只使用公开 API，不包含特定材料结果，不依赖绝对路径；额外依赖在文件头给出
`uv run --with ...` 命令。通用脚本不复制进每个案例目录，案例通过参数调用它。

### 真实材料案例

真实数据统一放在：

```text
examples/cases/<Material>/<case>/
  README.md
  structures/
    primitive.vasp
    reference.vasp
  workflow/
    run_mlfcs.py
    compare.py
  fitting/
    train.extxyz
  finite_difference/
    POSCAR-unitcell
    mlfcs-plan.json
    structures/POSCAR-001 ...
    calculations/POSCAR-001/vasprun.xml ...
    forces.npz
  results/
    mlfcs/
    reference/<software>/
  observables/
    phonons/
    thermal_conductivity/
```

目录只创建案例实际需要的部分。谐波拟合案例不必创建 `finite_difference/`，有限差分案例
不必创建 `fitting/`。`<Material>` 使用标准化学式大小写，例如 `Si`、`SrTiO3`、
`K4As4Pt2`；`<case>` 使用物理任务命名，例如 `harmonic-fit`、`fc3-fd`、`shengbte-kappa`。

## README 是唯一必需的案例说明

不要求统一 manifest 或 hash 文件，但每个案例 README 必须用人可读方式说明：

- 科学问题以及该案例证明和不能证明的内容；
- primitive、reference、超胞矩阵和原子顺序约定；
- 拟合阶数、cutoff、body order、位移、约束和单位；
- MLFCS 使用命令或脚本入口；
- 第三方数据和结果来自哪个软件、版本、输入参数或公开来源；
- 哪些文件是输入、MLFCS 结果、第三方参考和最终观测量；
- 声子或热导率比较所使用的 NAC、q 网格、展宽、同位素、边界和迭代/RTA 设置；
- 已知方法差异、未收敛项和结论的不确定性。

README 可以记录来源 URL、提交号或文件校验值，但它们不是统一硬性要求，也不作为每次
运行的严格 hash 门禁。第三方结果保留其原始格式，不需要转换成 MLFCS HDF5；MLFCS 自身
生成的通用力常数结果优先保存 native HDF5 v2，另按案例需要保留 phonopy、ShengBTE 或
ALAMODE 输出。

## 拟合数据组织

拟合训练集统一使用 ASE 可读取的严格 `extxyz`：

```text
fitting/train.extxyz
```

每一帧必须具有与 `structures/reference.vasp` 完全相同的原子数量、元素标签和原子顺序，
晶格必须相同，并在 ASE calculator results 中携带 `(N, 3)` forces。程序不静默重排或
重心化快照。能量、应力、温度、数据分组和来源等可作为 extxyz metadata 保存，但 forces
和 reference 顺序是最低契约。多个物理来源需要分开时可使用 `train-aimd.extxyz`、
`train-random.extxyz`，README 说明最终拟合使用的组合。

第三方原始训练格式可以保存在 `results/reference/<software>/` 或案例所需的 `source/`
子目录，但实际运行 MLFCS 的输入应为上述 extxyz，以免每个案例维护不同解析器。

## 有限差分数据组织

有限差分不转换成 extxyz。它是一个有顺序的位移计划，应保持 `sow()` 生成的结构目录和
外部计算目录：

```text
finite_difference/
  POSCAR-unitcell
  mlfcs-plan.json
  structures/POSCAR-001
  structures/POSCAR-002
  ...
  calculations/POSCAR-001/vasprun.xml
  calculations/POSCAR-002/vasprun.xml
  ...
  forces.npz
```

`structures/` 保留 MLFCS reference 原子顺序；`calculations/` 使用相同文件名建立一一对应。
目录的自然排序方便人工检查，但权威顺序是 `mlfcs-plan.json` 中的 `filenames` 列表和
`forces.npz` 中连续的 `configuration_ids`。收集任务可以乱序完成，collect 阶段必须按
manifest 恢复 sow 顺序。任何外部程序都不能在 POSCAR 与力之间重新排列原子。

若原始 `vasprun.xml` 过大或无再分发权限，可以只保留 `structures/`、计划、收集后的
`forces.npz`、必要外部输入说明和最终 MLFCS HDF5；README 必须说明原始计算存放位置。
专有 POTCAR、cache、电子迭代日志和无助于复现结论的中间文件不进入案例。

## 科学证据与第三方结果

案例允许原样保存第三方软件结果。它们是参考证据而非自动真值，比较前必须在 README
中说明 primitive、超胞平移子晶格、单位、周期像、cutoff 和约束是否一致。

建议按以下顺序建立可信度：

1. 力数据单位和留出误差；
2. ASR、旋转、置换和晶格表示不变量；
3. 相同约定下比较 IFC 或动力学矩阵；
4. 比较声子谱、稳定性、简并和 NAC；
5. 比较热导率，并给出 q 网格、cutoff、展宽和温度收敛；
6. 必要时与实验或文献比较并说明理论层级。

声子案例应保留机器可读的 q 点和频率，不能只有 PNG。热导率案例应保留温度相关张量和
关键收敛表，不能只有一个温度的数字。IFC 不逐元素相等但收敛物理量一致时，应报告为
方法差异；最终物理量接近但上游约定未对齐时，也不能宣称已经证明 writer 正确。

## 待批准的迁移顺序

本节只规划，不在批准前移动或删除现有数据：

1. 在 `examples/cases/` 建立材料/任务层级；
2. 将 `assets/SI` 规范为 `examples/cases/Si/`，拆分 harmonic、FC3 FD 和 ShengBTE kappa；
3. 将 `ALAMODE_SI`、`ALAMODE_SrTiO3` 迁为拟合案例，训练输入统一为 extxyz；
4. 将 `K3Au3Sb2` 按实际化学式和任务纠正命名；
5. 合并重复绘图脚本，案例统一调用 `examples/plot_phonon_band.py`；
6. 为每个案例整理 README，保留必要第三方输入/结果和 MLFCS native HDF5；
7. 清除 cache、重复导出、专有文件和无用中间输出前逐项列出删除清单；
8. 将 `tests/reference` 的材料对比迁入对应案例，先留下最小合成回归测试；
9. 最后删除 reference pytest marker、专用 CI job和不再需要的 reference 测试依赖；
10. 完整运行 unit/integration，并人工复核迁移后的声子与热导率案例。

在批准该迁移计划前，现有 `assets/`、`tests/reference/` 和案例结果保持原位。
