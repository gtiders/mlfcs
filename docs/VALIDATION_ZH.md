# 数值验证与持续集成

## 验证目标

测试需要分别回答三个问题，不能仅用“文件能够写出”代替数值验证：

1. 有限差分、对称性展开与力常数重建是否正确；
2. 原子顺序、晶格平移约定和张量指标是否能与外部程序对应；
3. HDF5、ShengBTE 和 phonopy 等输出是否保持相同的物理数据。

## 独立的 AlN 三阶基准

CI 中的 AlN 基准来自 phono3py 官方 `example/AlN-rd` 数据集。用
pypolymlp 0.20.4 在完整的 200 个结构上一次性训练势函数，然后使用同一个势函数
分别产生：

- MLFCS `sow()` 顺序中所有结构的力；
- phono3py 系统有限位移所需的力以及 traditional solver 得到的完整 FC3。

两条路径都使用 2x2x1 超胞和 0.01 Å 位移。CI 不重新训练势函数，也不运行
pypolymlp；它只读取压缩后的力和参考 FC3，因此运行时间和内存占用稳定。

比较时关闭 MLFCS ASR，使两边都保持原始有限差分结果，避免把不同的约束投影方法
混入有限差分验证。测试只比较 MLFCS 截断范围覆盖的原子三元组。

## hiphive 的作用

hiphive 仅属于开发依赖和独立验证工具，不参与 MLFCS 的计算实现。测试先完成：

1. 将 MLFCS 的 `(n_primitive, n_supercell, n_supercell, 3, 3, 3)` 平移约化表示
   展开为完整超胞 FC3；
2. 按元素、周期性最小镜像距离匹配 MLFCS 与 phono3py 的原子顺序；
3. 通过 hiphive `ForceConstants.from_arrays` 将双方规范化为同一种完整 FC3 表示；
4. 比较最大绝对误差和 RMS 误差。

这使 hiphive 成为格式与表示的第三方适配器，而不是被验证算法的一部分。

当前固定夹具在共同截断支持上的结果为：FC3 最大量级约 73.65 eV/Å³，最大绝对差
约 0.0211 eV/Å³，RMS 差约 0.00404 eV/Å³，相对二范数误差约
`2.83e-4`。CI 同时限制这三个指标，避免仅靠相关系数掩盖系统误差。

## 有 ASR 的交叉验证

第二项测试显式比较 MLFCS `acoustic_sum_rule=True` 与 phono3py traditional
`symmetrize_fc3(level=3)` 的结果。原始 phono3py FC3 的最大平移求和残差约为
`1.17 eV/Å³`；投影后 MLFCS 与 phono3py 的残差分别约为 `4.6e-12` 和
`3.1e-14 eV/Å³`，双方都严格满足 ASR。

投影后共同截断支持上的最大绝对差约为 `3.03 eV/Å³`，RMS 差约为
`0.507 eV/Å³`，相对二范数误差约为 `3.56%`，相关系数约为 `0.99943`。
这个差值明显大于无 ASR 情况，但不是 ASR 失败：phono3py 在完整稠密超胞空间寻找
投影解，MLFCS 在截断后的稀疏不可约空间寻找投影解。两者满足同一平移约束，却具有
不同的自由变量和最小改变量准则。因此 CI 分别约束“ASR 残差必须接近零”和“两个
投影结果必须保持高度一致”，不要求不同约束空间给出逐元素相同的 FC3。

## CI 分层

GitHub Actions 分为三个相互独立的任务：

- `unit-and-api`：在 Python 3.12 和 3.13 上运行 Ruff、格式检查以及所有非参考测试；
- `scientific-reference`：在 Python 3.12 上依次独立运行 hiphive 适配器、AlN 无 ASR
  FC3 和 AlN 有 ASR FC3；
- `package`：构建 Python sdist 和 wheel。

BLAS、OpenMP 和 JAX CPU 后端均限制为单线程，避免小型 CI 任务因嵌套并行产生不稳定
内存峰值。官方 AlN 势函数的重新训练是维护者基准，不属于每次 push 的 CI。

完整目录约定和执行命令见 `tests/README.md`。

## 数据来源和再生成

夹具的固定上游提交、许可证、软件版本与生成命令记录在
`tests/reference/phono3py/AlN_FC3/data/README.md`。生成程序为
`tools/generate_AlN_phono3py_fixture.py`。
