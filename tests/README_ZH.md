# 测试体系

[English](README.md)

测试只验证当前公开行为、数学约束和独立科学参考，不依赖旧版 MLFCS。

## 层级

- `unit/`：几何、有限差分、重建和 I/O 的快速单元测试；
- `integration/`：`sow`、`reap`、ASE calculator 和 SSCHA 的公开 API 流程；
- `reference/`：来自独立软件和固定数据的科学数值基准。

`reference/analytic/Morse_FCC_FC4/` 使用 ASE Morse 势产生力，并用独立的 JAX Morse
能量四次自动微分作为解析 FC4 参考。它同时验证 FCC 解析平衡晶格常数、四阶张量数值和
中心有限差分的二阶步长收敛，不依赖其他力常数拟合器或 MLFCS 的 ASR。

外部比较必须各自拥有独立目录、数据来源、许可、适配器和 CI 步骤。当前参考基准包括
`reference/phono3py/AlN_FC3/`：

- `test_raw.py` 独立比较未施加 ASR 的 FC3；
- `test_asr.py` 独立比较施加 ASR 后的 FC3 和双方残差；
- `test_adapter.py` 只验证 hiphive 表示转换和原子顺序匹配。
- `test_provenance.py` 独立校验训练数据、势函数和派生夹具的 SHA-256。

以及 `reference/phonopy/AlN_FC2/`：

- `test_fc2.py` 独立比较有、无 ASR 的完整 FC2；
- `test_provenance.py` 校验二阶派生夹具的 SHA-256；
- 二阶与三阶基准共用同一训练数据和 pypolymlp 势函数，但拥有独立参考夹具。

`reference/shengbte/Si_FC3/` 冻结 3x3x3、`cutoff=-6` 的外部 VASP sow 顺序，
并用 168 份 VASP 力端到端重建 ShengBTE `FORCE_CONSTANTS_3RD`，与原始
thirdorder 输出比较周期规范化后的块顺序和 FC3 数值。该比较显式使用
`compatibility="thirdorder"`；默认 ShengBTE 模式另行验证对称闭合支撑域的保真往返。

`reference/sscha/KCl_phonopy/` 将 phonopy 官方 KCl pypolymlp 势函数接入原生 MLFCS
SSCHA 流程，并使用 phonopy 官方测试的 FC2 和自由能范围验证数值尺度。其上层
的解析 q 空间测试则独立对比 MLFCS 与 phonopy 的频率和采样协方差。

## 命名约定

文件或目录出现化学式时必须保留标准大小写，例如 `AlN`、`Si`、`NaCl`，不使用
`aln`、`si`、`nacl`。算法名和项目名遵循其官方写法，例如 `ASR`、`FC3`、
`phono3py` 和 `hiphive`。

## 执行

```bash
uv run pytest -m "not reference"
uv run pytest tests/reference/analytic/Morse_FCC_FC4/test_morse_fc4.py
uv run pytest tests/reference/phono3py/AlN_FC3/test_adapter.py
uv run pytest tests/reference/phono3py/AlN_FC3/test_provenance.py
uv run pytest tests/reference/phono3py/AlN_FC3/test_raw.py
uv run pytest tests/reference/phono3py/AlN_FC3/test_asr.py
uv run pytest tests/reference/phonopy/AlN_FC2
uv run pytest tests/reference/sscha
uv run pytest tests/reference/shengbte/Si_FC3/test_sow_contract.py
uv run pytest tests/reference/shengbte/Si_FC3/test_provenance.py
uv run pytest tests/reference/shengbte/Si_FC3/test_numerical.py
```

参考基准必须串行运行。生成势函数和派生参考数据属于维护者操作，不进入普通 CI。
