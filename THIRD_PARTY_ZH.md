# 第三方来源说明

## thirdorder

MLFCS 参考并借鉴了 [thirdorder](https://gitlab.com/sousaw/thirdorder) 的算法和工作流，
包括有限位移三阶力常数的对称性约化、周期镜像几何约定，以及用于生成 ShengBTE 兼容
数据的有序 `sow`/`reap` 流程。

MLFCS 在这一技术来源上进一步设计了不同的阶数参数化 ASE/JAX 架构。MLFCS 的新增工作
包括二阶至任意阶的统一路径、递归中心差分模板、稀疏团簇存储、受约束声学求和规则
求解、CPU/GPU 张量运算、与计算器解耦的 API 和多种导出格式。这些新增内容不消除或
替代对 thirdorder 的来源说明。

开发过程中参考的 thirdorder 源码版本为
`7cb4ef0d2e036941165b016ba1b4f23bdd0e81c7`。其源码声明列出的版权所有者为：

- Copyright (C) 2012–2018 Wu Li
- Copyright (C) 2012–2018 Jesús Carrete Montaña
- Copyright (C) 2012–2018 Natalio Mingo Bisquert
- Copyright (C) 2014–2018 Antti J. Karttunen
- Copyright (C) 2016–2018 Genadi Naydenov

thirdorder 采用 GNU 通用公共许可证第 3 版或更高版本。MLFCS 采用相同许可证系列，详见
[LICENSE](LICENSE)。MLFCS 不主张 thirdorder 的作者身份；同样，thirdorder 作者也不
因此被表述为 MLFCS 新增内容的作者。

## ALM FCSXML writer

ALAMODE XML 适配器参考并改编自 [ttadano/ALM](https://github.com/ttadano/ALM) 仓库
`f1d668f210d3e95355643132144f3fd1ec10d4d7` 版本中的纯 Python
`alm.fcsxml.Fcsxml` writer。该实现的版权所有者为 Terumasa Tadano，采用 MIT
许可证。MLFCS 保留其 XML 布局、单位换算和 27 周期像最近镜像约定，同时以 MLFCS
控制的原子与平移映射替代原胞重新识别。改编模块在源码头部保留了来源声明。
完整上游许可证保存在 [third_party_licenses/ALM-MIT.txt](third_party_licenses/ALM-MIT.txt)。

## 科学参考数据

仓库再分发的第三方测试输入和参考数据在文件旁保留其上游声明与许可证。生成的测试
夹具在本地 README 中记录上游版本、哈希和再生成步骤。科学论文引用不能替代对相应
软件或数据许可证的遵守。
