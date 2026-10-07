# 超胞平移商：$\mathbb Z^3/\mathbb Z^3 S$ 与商标签约定

`ClusterMap` 的原子匹配要回答一个问题：给定原胞位点 $i$ 与原胞平移
$\mathbf n$，它对应超胞里的哪个原子；以及反过来，哪些 $\mathbf n$ 对应
**同一个**超胞原子。答案由一个有限商群决定，而这个商群的记号在行向量约定
下有一个容易写错的细节，本文把它固定下来。

## 行向量约定

本代码库全部使用行向量：晶胞以格矢为**行**，超胞关系为

$$
A_{\text{super}} = S\, A_{\text{prim}},
$$

平移 $\mathbf n \in \mathbb Z^3$ 是行向量，作用为右乘 $\mathbf n\,A_{\text{prim}}$。

代码中的 ClusterMap、prepare_supercell_data 和 quotient_label 都采用这
套行约定：传入的超胞矩阵满足 $A_{\text{super}}=S A_{\text{prim}}$，平移按行
向量参与运算，标签按 $\mathbf n\operatorname{adj}(S)$ 计算。因此这里不存在一
部分代码按行、另一部分代码按列的混用。

若把同一计算改写为列向量形式，应同时转置坐标和超胞矩阵：
$\mathbf n_{\text{col}}=\mathbf n^T$、
$S_{\text{col}}=S^T$，且列基满足
$A_{\text{super,col}}=A_{\text{prim,col}}S_{\text{col}}$。对应标签为
$\operatorname{adj}(S_{\text{col}})\mathbf n_{\text{col}}$，它正好是行约定
标签的转置。两种实现的乘法顺序不同，但表示的是同一个商类。只转置平移而
不转置矩阵，会把子格换成另一个子格。

## 等价子群是 $\mathbb Z^3 S$，不是 $S\mathbb Z^3$

超胞的格矢集合是 $\{\mathbf m\,A_{\text{super}}\} = \{\mathbf m S\,A_{\text{prim}}\}$。
于是两个原胞平移把同一位点映到同一超胞原子，当且仅当

$$
\mathbf n_1 - \mathbf n_2 = \mathbf m S,\qquad \mathbf m \in \mathbb Z^3.
$$

等价子群是 $S$ 的整数**行**生成格 $\mathbb Z^3 S$，商群为

$$
\mathbb Z^3 / \mathbb Z^3 S,
$$

其阶等于 $|\det S|$。若改用列向量表示平移，则同一行格的转置是
$S^T\mathbb Z^3$。若列向量约定将超胞矩阵定义为
$S_{\mathrm{col}}=S^T$，该格也可写作 $S_{\mathrm{col}}\mathbb Z^3$。
换用约定时应同时转换超胞矩阵，不能只把行向量改写为列向量而沿用同一个
矩阵乘法方向。

## 商标签及其正确性

`quotient_label` 用

$$
q(\mathbf n) = \mathbf n\,\operatorname{adj}(S) \bmod |\det S|
$$

把每个平移编码为商类的典范非负标签。它的核恰好是 $\mathbb Z^3 S$：

- 若 $\mathbf n = \mathbf m S$，则
  $\mathbf n\,\operatorname{adj}(S) = \mathbf m\,S\operatorname{adj}(S) = \det(S)\,\mathbf m \equiv 0 \pmod{|\det S|}$；
- 反之若 $\mathbf n\,\operatorname{adj}(S) = |\det S|\,\mathbf k$，右乘 $S$ 得
  $\mathbf n\,\det(S) = |\det S|\,\mathbf k S$，即 $\mathbf n = \pm\mathbf k S$。

令 $d=|\det S|$。公式中的标签实际属于 $(\mathbb Z/d\mathbb Z)^3$，因此
$q$ 是从 $\mathbb Z^3$ 到该剩余向量群的同态，且 $\ker q=\mathbb Z^3S$。
它的像只有 $d$ 个元素（而不是整个含 $d^3$ 个元素的剩余向量群），由第一
同构定理，商群 $\mathbb Z^3/\mathbb Z^3S$ 与 $\operatorname{im}(q)$ 同构。
实现中用 $\{0,\dots,d-1\}^3$ 中的非负向量表示这些像元素。

## $\mathbb Z^3 S$ 与 $S\mathbb Z^3$ 可以是不同的格

对非对称的 $S$，两者一般**不相等**。二维反例，
$S = \begin{pmatrix}2 & 1\\ 0 & 2\end{pmatrix}$：

- $\mathbb Z^2 S = \{(2a,\ a+2b)\}$ 包含 $(2,1)$；而
  $S\mathbb Z^2 = \{(2a+b,\ 2b)\}$ 的第二分量恒为偶数，故
  $(2,1) \notin S\mathbb Z^2$；
- 反之 $(1,2)=S(0,1) \in S\mathbb Z^2$，但 $(1,2)\notin\mathbb Z^2S$，
  因为 $\mathbb Z^2S$ 中向量的第一分量必为偶数。并且
  $q((1,2))=(2,3)\bmod 4=(2,3)\neq\mathbf 0$，这与
  $\ker q=\mathbb Z^2S$ 一致，也表明 $\ker q\neq S\mathbb Z^2$。

两个子格的指标同为 $|\det S|$。它们的商群因 $S$ 与 $S^T$ 的 Smith 标准形
相同而同构，但作为 $\mathbb Z^2$ 的子格不同，商标签的坐标表示也不同。

在三维代码中可将此二维例子嵌入为
$\begin{pmatrix}2&1&0\\0&2&0\\0&0&1\end{pmatrix}$，结论不变。

## 非对角超胞为什么仍然正确

非对角矩阵表示超胞基矢是原胞基矢的整数线性组合；它可以剪切平移子格，
改变其基矢方向，但不改变商标签证明所依赖的恒等式
$S\operatorname{adj}(S)=\det(S)I$。只要 $S$ 是非奇异整数矩阵，标签的核
仍严格等于超胞平移子格，商群大小仍为 $|\det S|$。算法不假设每个方向
分别按对角元素取模，也不需要超胞边界与原胞坐标轴对齐。

例如取
$S=\begin{pmatrix}2&1&0\\0&2&0\\0&0&1\end{pmatrix}$，则
$|\det S|=4$、
$\operatorname{adj}(S)=\begin{pmatrix}2&-1&0\\0&2&0\\0&0&4\end{pmatrix}$。
四个平移 $(0,0,0)$、$(1,0,0)$、$(0,1,0)$、$(1,1,0)$ 的标签分别为
$(0,0,0)$、$(2,3,0)$、$(0,2,0)$、$(2,1,0)$，彼此不同，因而对应四个不同
商类。更多整数平移会重复这些标签，当且仅当它们相差一个超胞平移。

实现中，prepare_supercell_data 先检查输入晶胞是否满足
$A_{\text{super}}=S A_{\text{prim}}$，再将每个原子匹配到原胞位点和整数
平移，并计算其商标签。它要求每个“原胞位点 × 商类”键唯一；同时原子数必须
为 $n_{\text{primitive}}|\det S|$。标签的核证明保证这些唯一键就是完整的
商类覆盖，而不是依赖对角超胞中特有的逐轴枚举。之后 map_labels 对查询
平移使用同一个标签公式查找超胞原子，所以非对角矩阵和对角矩阵走同一条逻辑。

## 为什么此前未暴露

教程与测试中的所有超胞矩阵都是对角的（$7\times7\times1$、
$\operatorname{diag}(2,2,3)$、$\operatorname{diag}(2,2,2)$ 等）。对角
$S$ 满足 $\mathbb Z^3 S = S\mathbb Z^3$，两种记号给出同一个子格，混淆只有
在一般（非对称、剪切）超胞矩阵下才会显形。这正是 API 必须对任意
$S$ 正确的原因。

## 在代码中的位置

- `mapping/periodic.py` 的 `quotient_label(translation, adjugate, modulus)`
  实现商标签；
- `mapping/supercell.py` 的 `prepare_supercell_data` 验证
  *primitive site × 商类* 恰好被超胞原子一一覆盖：每个原子恰有一个匹配、
  原子数必为 $n_{\text{primitive}}\,|\det S|$、且
  $(i,\text{label})$ 二元组无重复；
- `ClusterMap.map_labels` 以 $(i, [\mathbf n + \mathbf t])$ 为键查回超胞
  原子序号。

## 约定条款

本文档与代码统一使用行约定记号 $\mathbb Z^3/\mathbb Z^3 S$。外部文献常以
列约定写 $S\mathbb Z^3$，阅读时注意换算。
