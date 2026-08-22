# 完整 SNF 能否统一有限超胞平移映射

本文独立研究完整 Smith normal form 是否适合替代 MLFCS 当前以 HNF 为中心的
有限超胞平移映射。研究不修改正式代码、IFC schema、拟合器、SSCHA、SCPH 或
writer。

结论是：**完整 SNF 在数学上可以统一 direct quotient、cell addition、alias
判定和 reciprocal characters，但目前不应替换 HNF 生产路径。** 推荐保留研究
prototype，不接入运行时。

原因不是 SNF 不正确，而是：

- SNF 的对角矩阵 $D$ 是 canonical invariant，双侧 transformation $U,V$ 不是；
- SNF 群坐标不能从结构几何恢复无限晶格中的 exact $R$；
- 当前 HNF 路径已经没有 BFS、hash 或 representative search；
- lookup 加速折算到实际拟合初始化只有约 $0.1$ 秒，而主要 design 阶段是数百秒；
- 若为获得实际、短小且与超胞行基无关的 translation representative 而保留 HNF，
  就会形成两套坐标系统，代码不会减少；
- 当前 SymPy 1.14 虽提供完整分解，但其 transformation 实现存在尚未关闭的上游
  正确性问题，而且项目依赖范围仍允许不具备该接口的 SymPy 1.13。

独立验证代码位于 `research/snf_mapping/prototype.py`。

## 1. 当前 HNF 链路实际做了什么

MLFCS 采用 row-vector primitive translation。primitive 晶格矩阵为 $A$，整数
supercell matrix 为 $S$，则无限 primitive image 的位置是

$$
r(a,R)=(p_a+R)A,
\qquad R\in\mathbb Z^3,
$$

而 supercell translation 子晶格是

$$
L_S=\mathbb Z^3S.
$$

有限平移群为

$$
G_S=\mathbb Z^3/L_S.
$$

当前 `IntegerLatticeQuotient` 对 $S$ 计算 canonical row-HNF $H$，满足

$$
H=QS,
\qquad Q\in GL(3,\mathbb Z),
$$

所以

$$
\mathbb Z^3H=\mathbb Z^3S.
$$

任意 translation 被唯一分解为

$$
R=nH+r,
$$

其中 $r$ 落在 HNF 基本域。`cell_index()` 将 $r$ 通过混合进制转换为连续整数。

当前实现需要澄清两点：

1. 它已经不使用 BFS 或搜索式 representative enumeration；
2. reduction 和 cell indexing 不使用 hash table，atom lookup 才使用预构造的稠密
   `(primitive_site, cell_index)` 数组。

因此候选 SNF 对比的是一个已经是 $O(1)$ 批量整数运算的 HNF 实现，而不是旧的
搜索算法。

## 2. SymPy 完整 SNF 能力调查

### 2.1 高层 API

`sympy.matrices.normalforms.smith_normal_form()` 只返回 $D$。SymPy 的
[normal forms 文档](https://docs.sympy.org/latest/modules/matrices/normalforms.html)
也只公开展示该结果，不返回 transformation。

### 2.2 DomainMatrix / polys 底层 API

当前环境使用 SymPy 1.14。进一步检查
`sympy.polys.matrices.normalforms` 后确认它包含：

```text
smith_normal_decomp(DomainMatrix) -> (D, U, V)
```

并满足

$$
D=USV.
$$

该函数不是以下划线开头的私有函数，具有 docstring 和示例；其内部调用
`_smith_normal_decomp(..., full=True)`，确实在每次 elementary row/column operation
时同步更新 $U,V$。因此，在 SymPy 1.14 上无需复制或扩展其内部算法。

但它仍有三个生产风险：

- 它位于较低层的 `polys.matrices.normalforms`，没有出现在高层 normal-forms API
  文档中；
- MLFCS 当前声明 `sympy>=1.13`，而实测 SymPy 1.13.3 不存在
  `smith_normal_decomp`；
- SymPy 上游存在一个 2026 年仍未关闭的
  [transformation 非 unimodular 报告](https://github.com/sympy/sympy/issues/29139)。

本研究针对 19,850 个随机、满秩、元素范围 $[-8,8]$ 的 $3\times3$ 整数矩阵，
以及正式 prototype 中 2,000 个随机矩阵验证了：

$$
D=USV,
$$

$$
|\det U|=|\det V|=1,
$$

$$
d_1\mid d_2\mid d_3,
$$

没有发现失败。这说明 MLFCS 的小型满秩 $3\times3$ 范围内可用于研究，但不能
替代生产代码接入时必须执行的完整 validation。

### 2.3 替代库

- [hsnf](https://hsnf.readthedocs.io/en/latest/) 直接返回 $(D,U,V)$，依赖轻，API
  符合需求；但引入它只为一次 $3\times3$ 分解，并不能解决 transformation
  非唯一或 representative presentation 问题。
- [python-flint](https://python-flint.readthedocs.io/en/latest/fmpz_mat.html) 的
  `snf()` 只返回 SNF；它的 HNF 和 LLL 支持 transformation，但 SNF Python API
  不返回双侧 transformation。
- [SageMath](https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html)
  的 `smith_form(transformation=True)` 正式返回完整分解，但 Sage 对 MLFCS 是明显
  过重的运行时依赖。
- `snforacle` 提供完整 transformation 的统一后端接口，但它是较新的封装层，
  常用后端仍是 PARI、FLINT 或 Sage；对于固定 $3\times3$ 矩阵没有实际收益。
- `smithnormalform` 能返回 transformation，但使用 GPLv3，版本和实现都不比当前
  SymPy/hsnf 路线更适合作为基础依赖。

如果未来仅做受控 prototype，优先直接调用已经依赖的 SymPy 1.14 并严格验证。
如果未来真的进入生产，则应先解决最低 SymPy 版本和上游 correctness 风险，而
不是复制 `_smith_normal_decomp()`。

## 3. SNF direct-space 坐标的严格推导

设完整分解为

$$
D=USV,
$$

其中

$$
U,V\in GL(3,\mathbb Z),
\qquad
D=\operatorname{diag}(d_1,d_2,d_3),
$$

且 $d_i>0$。由

$$
SV=U^{-1}D
$$

可得

$$
L_SV
=\mathbb Z^3SV
=\mathbb Z^3U^{-1}D
=\mathbb Z^3D.
$$

因此右乘 $V$ 给出群同构

$$
\psi:
\mathbb Z^3/\mathbb Z^3S
\longrightarrow
\mathbb Z^3/\mathbb Z^3D,
$$

$$
[R]\longmapsto[RV].
$$

有限群坐标就是

$$
k(R)=RV\bmod(d_1,d_2,d_3).
$$

它属于

$$
\mathbb Z_{d_1}\times\mathbb Z_{d_2}\times\mathbb Z_{d_3}.
$$

于是周期平移相加严格变成逐分量模加法：

$$
k(R+T)=k(R)+k(T)\pmod D.
$$

若采用字典序混合进制，cell index 为

$$
i(k)=k_1d_2d_3+k_2d_3+k_3.
$$

一个群元素 $k$ 的原 primitive-coordinate representative 可以选择为

$$
\rho(k)=kV^{-1}.
$$

因为 $V$ unimodular，所以 $V^{-1}$ 仍是整数矩阵，并且

$$
k(\rho(k))=k.
$$

这证明完整 SNF 确实不需要 BFS、residue search 或 hash table 就能完成 finite
translation indexing。

## 4. 从 supercell atom 到 target atom 的完整链条

### 4.1 reference 构造

`StructureRelation` 仍必须根据晶格、元素和周期位置，确定每个用户 reference atom
对应的

$$
(a,R).
$$

SNF 只能把已知 $R$ 折叠为 $k(R)$，不能从近似浮点坐标和任意原子排列中推导
$a,R$。因此 Hungarian matching、结构残差和 primitive-site 识别不能由 SNF
替代。

映射表可以写为

$$
\text{atom\_by\_site\_group}[a,i(k(R))]=j.
$$

### 4.2 primitive exact-$R$ interaction realization

对于锚点 cell $T$ 和 exact interaction

$$
(a_0,0),(a_1,R_1),\ldots,(a_m,R_m),
$$

目标 reference 中的 atom index 可以由

$$
k_i=k(T)+k(R_i)\pmod D
$$

直接查表。这一部分确实比一般 reduction 的数学表达更简单。

### 4.3 从 source 到不同 target

canonical IFC 必须继续保存 exact $R_i$，不能只保存 source 的 $k_i$。原因是

$$
k(R)=k(R+nS)
$$

会丢失无限 primitive 晶格中的信息。两个 translation 在 source 中相同，不保证
在另一个 target supercell 中相同。

因此跨超胞导出必须执行：

```text
source structure geometry -> exact (a, R)
canonical IFC             -> exact (a, R)
target SNF/HNF quotient   -> target finite cell
target table              -> target atom index
```

SNF 可以统一 source 和 target 的 finite indexing API，但不能把整个链条压缩为
source group coordinate 到 target group coordinate。

## 5. reciprocal quotient 的严格推导

row-vector reciprocal fractional point记为 $q$。它与 supercell 相容的条件是

$$
qS^T\in\mathbb Z^3,
$$

等价地，对所有 $n\in\mathbb Z^3$，$(nS)q^T$ 为整数。

用 SNF group coordinate $\ell$ 枚举 reciprocal characters：

$$
\ell_i\in\{0,1,\ldots,d_i-1\}.
$$

定义

$$
q(\ell)=\ell D^{-1}V^T\pmod 1.
$$

因为

$$
V^TS^T=(SV)^T=(U^{-1}D)^T=DU^{-T},
$$

所以

$$
q(\ell)S^T
=\ell D^{-1}V^TS^T
=\ell U^{-T}
\in\mathbb Z^3.
$$

direct translation 与 reciprocal character 的配对是

$$
Rq(\ell)^T
=(RV)D^{-1}\ell^T.
$$

因此同一个 $(D,V)$ 足以支持 direct cell coordinate 和 reciprocal q-point，
不需要再对 $S^T$ 做第二次 SNF。prototype 已对对角、非对角和高剪切矩阵验证
SNF 与当前 `quotient_qpoints()` 的 q-point 集合完全相同，只是顺序可以不同。

这是 SNF-centered 方案最真实的架构优势。

## 6. periodic folding 与 aliasing

两个 translation 在当前 supercell 中等价，当且仅当

$$
k(R_1)=k(R_2),
$$

也就是

$$
(R_1-R_2)V=0\pmod D.
$$

高阶 interaction folding 可以逐个 site 比较锚定后的 SNF coordinates。因而 SNF
使 alias 的整数判据非常直观。

但这只简化“两个 exact key 是否折叠到同一个 finite key”的检测。它不能判断
折叠后的参数系统是否可辨识，也不能替代 realization operator 的 rank 检查。
HNF cell index 相等同样已经提供完全相同的判据。

## 7. HNF-centered 与 SNF-centered 逐步对照

| 操作 | 当前 HNF-centered | 候选 SNF-centered |
|---|---|---|
| 初始化 | 一次 SymPy HNF | 一次完整 $(D,U,V)$ SNF 和验证 |
| quotient identity | HNF remainder | $RV\bmod D$ |
| cell index | HNF remainder 的混合进制 | SNF coordinate 的混合进制 |
| representative | HNF 基本域中的小整数向量 | $kV^{-1}$，可能出现大系数 |
| addition | 先加 translation 再 reduce | 直接逐分量模加 |
| atom lookup | 稠密二维数组 | 同一个稠密二维数组 |
| hash/search | 无 | 无 |
| q-point | 对 $S^T$ 的 HNF reps 加 adjugate | 同一 $(D,V)$ 直接生成 |
| exact $R$ | 仍需保存 | 仍需保存 |
| structure matching | 仍需执行 | 仍需执行 |
| alias detection | HNF index equality | SNF coordinate equality |
| 行基不变性 | canonical row-HNF 相同 | $D$ 相同，但 $U,V,k$ 通常不同 |

### 7.1 canonicality 的关键差别

$D$ 是有限阿贝尔群的不变量，但给定 $D$ 的 $U,V$ 不唯一。对定义相同 row
sublattice 的

$$
S'=MS,
\qquad M\in GL(3,\mathbb Z),
$$

SymPy 会给出相同 invariant factors，却不保证相同 $V$ 或相同 logical coordinate。
prototype 已对此给出实际反例。

若要求同一物理子晶格具有相同 logical indexing，就必须先把 $S$ canonicalize 为
row-HNF，再对 HNF 做 SNF。这样 HNF 仍是必要步骤，SNF 不再替代它。

## 8. operation count 与实测成本

### 8.1 每个 translation 的理论操作

当前批量 HNF reduction 对三个轴顺序执行：

- 3 次整数 floor division；
- 当前 NumPy 写法约 9 次整数乘法和 9 次减法；
- 当前 debug invariant 还执行一次 $qH+r=R$ 的矩阵验证；
- cell index 再执行 3 次乘法和 2 次加法。

SNF coordinate 执行：

- $RV$：9 次整数乘法和 6 次加法；
- 3 次逐分量 modulo；
- cell index：3 次乘法和 2 次加法。

两者都是固定 $O(1)$，都不需要 hash、search 或 representative table lookup。
SNF 的优势主要来自一次 dense integer matmul 和一次 mod 可以由 NumPy 连续执行，
而当前 HNF reduction 有三个顺序 array pass。

### 8.2 一百万次 lookup 实测

在当前 CPU、NumPy 和正式 HNF 实现上：

| $S$ | HNF | SNF | SNF/HNF |
|---|---:|---:|---:|
| $\operatorname{diag}(2,2,2)$ | 0.086 s | 0.045 s | 0.52 |
| $\begin{pmatrix}2&1&0\\0&2&1\\0&0&2\end{pmatrix}$ | 0.086 s | 0.044 s | 0.52 |
| 高剪切、$|\det S|=60$ | 0.084 s | 0.040 s | 0.47 |

该结果说明 SNF modular indexing 的批量 kernel 约快两倍，但绝对差值只有约
$0.04$ 秒/百万次。去除当前 HNF 的逐调用 invariant verification 后，差距还会
进一步缩小。

### 8.3 对实际 MLFCS 的 amortized 估算

SnSe FC2+FC3+FC4 parameter packing 中，按实际 image、32 个 translation cell 和
阶数估算 atom translations 为

$$
664\times32\times2
+7280\times32\times3
+6144\times32\times4
=1{,}527{,}808.
$$

按上述 benchmark，SNF 最多节约约 $0.06$ 至 $0.08$ 秒。该案例的 design kernel
约为 598 秒。因此 lookup 加速不到总时间的 $0.02\%$，不构成生产迁移理由。

supercell matrix 只有 $3\times3$，HNF 与 SNF decomposition 的单次初始化成本
都可以忽略；代表元枚举都是 $O(|\det S|)$，内存也都是
$O(|\det S|)$ 或一个很小的 atom lookup table。

## 9. 代码复杂度和稳健性

SNF-centered 实现可以删除或简化：

- HNF 的三步 triangular reduction；
- reciprocal quotient 对 $S^T$ 的第二个 HNF quotient；
- q-point 的 adjugate/determinant 表达。

但它必须新增：

- DomainMatrix 1.14 版本边界；
- $D,U,V$ 的 exact identity、unimodularity 和 divisibility validation；
- $V^{-1}$ 及其 int64 overflow 检查；
- transformation 非 canonical 的行为说明；
- 为 structure slots 和 `cell_representatives` 生成实际 translation representative；
- 上游 transformation bug 的防护与回归测试。

在 highly skew matrix 上，SNF 的 $V$ 和 $V^{-1}$ 可能含有远大于 $S$ 或 HNF 的
系数。prototype 的一个 $|\det S|=60$ 示例已经产生 $V$ 中的 $-291$。这会增加
int64 overflow 风险，并让实际 representative 不适合结构构造、几何诊断和日志。

若同时保留 HNF 作为 real-space presentation layer，则 direct indexing 使用 SNF、
实际 representative 使用 HNF，代码会出现两套必须持续交叉验证的坐标系统。
这与“减少 mapping 路径”的目标相反。

## 10. 最终判定

### 理论结论

完整 SNF **可以**成为有限 supercell translation group 的统一抽象坐标系统：

$$
R\mapsto RV\bmod D
$$

统一 direct folding、cell addition、atom-table indexing、alias equality 和 reciprocal
characters。这个数学结论成立。

### 工程结论

对当前 MLFCS，结论为 **No-Go for production replacement**：

- 它不能替代 structure matching 或 exact-$R$ identity；
- 它不能替代 canonical real-space representative；
- 它不能解决 alias identifiability；
- 当前 HNF 已经是无搜索、无 hash 的 $O(1)$ batch reduction；
- 实际节约不足 $0.1$ 秒，而引入 transformation API 和双坐标验证会增加维护风险；
- $U,V$ 不 canonical，且当前 SymPy transformation 路径仍有版本和正确性风险。

### 保留建议

- HNF 继续作为唯一生产 quotient reduction、canonical representative 和 cell
  indexing 系统。
- `residue_key()` 继续只作为 exact equivalence oracle 和测试交叉验证。
- SNF prototype 保留，用于群结构诊断、direct/reciprocal 公式验证，以及未来若要
  进行 FFT 因子化时研究 $\mathbb Z_{d_1}\times\mathbb Z_{d_2}\times\mathbb Z_{d_3}$。
- 不新增 SNF 运行时依赖或用户 API。
- 如果未来 q-point/FFT 需要显式循环群坐标，可单独考虑“SNF 仅服务 reciprocal
  factorization”，但不应因此替换整个 HNF mapping 链。

核心问题的答案是：

$$
\boxed{\text{完整 SNF 能统一有限群坐标，但不能更简单、更稳健地统一整个结构与 exact-}R\text{ 映射链。}}
$$
