# 外部 VASP sow/collect/reap 工作流

[English]

MLFCS 不启动 VASP，也不规定 INCAR、KPOINTS、POTCAR、调度器或收敛设置。它只定义
有限位移结构，以及力必须按什么契约返回。完整示例为
[`examples/vasp_external_fc3.py`]。

从 API 契约看，严格的位置顺序已经足够：第 `i` 组力必须对应 `sow()` 返回的第 `i` 个
结构，此时 `reap(forces)` 不需要 ID。本示例额外使用 manifest，
用于批处理任务、断点续算、缺失结果检查和长期来源记录。

## 1. 按确定顺序生成结构

从 VASP 原胞结构开始，并确定后续重建必须保持一致的物理参数：

```bash
uv run python examples/vasp_external_fc3.py sow POSCAR fc3-work \
  --supercell 3 3 3 \
  --cutoff -6 \
  --displacement 0.01
```

程序生成：

```text
fc3-work/
├── POSCAR-unitcell
├── mlfcs-plan.json
└── structures/
    ├── POSCAR-001
    ├── POSCAR-002
    └── ...
```

`POSCAR-001` 对应从零开始的构型 ID 0，`POSCAR-002` 对应 ID 1，以此类推。清单保存
超胞、截断、位移量、reference 原子顺序、构型数量和文件顺序。生成以后不要
重命名、遗漏、去重或重新排列这些构型。

## 2. 准备并提交 VASP

为每个生成文件建立独立计算目录。典型准备循环如下：

```bash
mkdir -p fc3-work/calculations
for structure in fc3-work/structures/POSCAR-*; do
  name=$(basename "$structure")
  directory="fc3-work/calculations/$name"
  mkdir -p "$directory"
  cp "$structure" "$directory/POSCAR"
  cp INCAR KPOINTS POTCAR "$directory/"
done
```

完成后的目录约定为：

```text
fc3-work/calculations/
├── POSCAR-001/vasprun.xml
├── POSCAR-002/vasprun.xml
└── ...
```

使用服务器本地调度器提交这些目录。每个任务都必须是设置一致的静态受力计算：采用
完全相同的电子结构参数、足够严格的力精度，并关闭离子弛豫。调度系统和 VASP 环境
具有站点差异，因此 MLFCS 有意不提供通用提交脚本。

## 3. 整理力

确认每个目录都有完整的 `vasprun.xml` 后，按清单顺序收集力：

```bash
uv run python examples/vasp_external_fc3.py collect \
  fc3-work fc3-work/calculations
```

缺少任意结果都会报错。程序通过 ASE 读取每个 `vasprun.xml` 的最终力数组，并写出
`fc3-work/forces.npz`，其中包含构型 ID 和原子顺序。原始 VASP 目录仍然是
权威原始数据，应另外归档。

## 4. Reap 并导出

重建 FC3，并写出默认的保真 ShengBTE 格式：

```bash
uv run python examples/vasp_external_fc3.py reap \
  fc3-work FORCE_CONSTANTS_3RD \
  --format shengbte
```

默认施加严格平移 ASR；传入 `--no-asr` 可保留原始有限差分结果。ShengBTE 直接
序列化重建后的稀疏物理支撑域，并在格式边界解析周期平移。其他 FC3 输出包括通用稀疏
`hdf5` 和完整 `phono3py_hdf5`。

## 恢复与审计约定

- 将 `mlfcs-plan.json`、`POSCAR-unitcell` 和 `forces.npz` 一起保存；
- 另外归档原始 VASP 输入和所有 `vasprun.xml`；
- 结构、超胞、截断、位移量、MLFCS 实现或对称性容差发生变化时，应重新 sow，不能
  混用数据；
- 调度器乱序完成不影响结果，collect 会通过目录名恢复准确顺序；
- 力必须对应生成 POSCAR 中的 reference 原子顺序，VASP 与 collect 之间不能再次排序。
