# 项目准则

1. 所有 Markdown（`.md`）文件中的数学公式必须使用美元符号：行内公式使用 `$...$`，独立公式使用 `$$...$$`；不得使用 `\(...\)`、`\[...\]` 或其他数学定界符。
2. 教学拟合由文档中的教程 notebook 承担：notebook 的完整执行输出（含图、指标与文本输出）是拟合任务的结果记录，必须纳入 Git，且不得被 `.gitignore` 排除；重新生成输出时必须逐个串行执行 notebook。教学拟合不得依赖公用日志包装器。
3. `src/mlfcs/tools` 包不编写常驻测试：工具的正确性在实现时以一次性临时验证代替，验证脚本及其输出不纳入版本库；测试套件不保留工具包的依赖方向门禁。
4. Every function and class must have an English docstring, including private members. The level of detail, however, must match the semantic role of the object rather than follow a uniform template.

For public APIs and classes or functions that define a mathematical or physical object, the docstring must explain the object from the user's perspective before describing implementation details. It should include, where relevant:

- the mathematical or physical object being represented or computed;
- the meaning of important parameters and return values;
- non-obvious shape, type, domain, or dimensional constraints;
- conventions required to interpret the result correctly, such as units, coordinate basis, tensor-index order, row/column-vector convention, normalization, periodicity, or boundary conditions;
- mathematically meaningful truncation, symmetry, invariance, or equivalence rules;
- exceptions whose conditions are part of the public contract;
- literature references or equation numbers when the definition is taken from an external source.

Do not mechanically document information that is already obvious from type annotations or the implementation. In particular, dtype, array shape, readonly status, allocation strategy, copying behavior, internal indexing details, and validation steps should only appear when they are necessary to correctly interpret or use the API.

For private helper functions, the docstring should concisely state the function's role in the algorithm and any non-obvious mathematical or numerical convention it relies on. Private helpers are not required to repeat full parameter tables, return schemas, or exception lists when these are already defined by the public interface.

Docstrings must prioritize semantic meaning over implementation mechanics. For physics-related code, prefer descriptions such as "number of distinct atomic positions involved" over storage-oriented descriptions such as "number of unique lattice addresses" unless the latter distinction is itself part of the physical definition.

Inline comments may explain local implementation choices, but must not replace a docstring.
