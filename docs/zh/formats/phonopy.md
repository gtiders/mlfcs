# Phonopy 与 phono3py

phonopy FC2 文本是稠密表示，因此必须使用相容的目标超胞。phono3py HDF5 是外部 schema，只有明确目标
phono3py 版本和配套超胞时才应写出。原生 MLFCS HDF5 仍是所有阶数的无损源文件。

尽可能使用后处理软件自己的 primitive 和 supercell。不能直接比较两个独立选择的超胞表示中的稠密数组下标。
