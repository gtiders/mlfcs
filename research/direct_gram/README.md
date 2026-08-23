# Direct-Gram 研究原型

这里保存 Direct-Gram / matrix-free design 构造的独立研究代码。它不属于
`mlfcs` 运行时，也不会被拟合器自动调用；正式实现仍然使用现有 design
构造和 dense SYRK 路径。

## 运行

使用 `uv` 在仓库根目录运行。例如：

```text
uv run python research/direct_gram/prototype.py --case si --orders 2 --frames 2
uv run python research/direct_gram/prototype.py --case si --orders 2 3 --frames 2
uv run python research/direct_gram/prototype.py --case si --orders 2 3 4 --frames 2
uv run python research/direct_gram/prototype.py --case snse --orders 2 3 4 --frames 181 --metadata-only
```

小型 Si 运行会收集 bounded design tiles，并逐项比较显式 $X^T X$、$X^T y$
与 tile-pair Direct-Gram 结果。SnSe 的完整 tile-pair contraction 只在专门
的受控实验中运行；默认先使用 `--metadata-only` 输出 design 规模，避免误申请
平方级临时内存。

完整数学推导、成本估计、数值结果和 No-Go 判定见
`docs/zh/开发/研究/Direct-Gram与matrix-freedesign研究.md`。
