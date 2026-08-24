---
title: 版本策略
audience:
  - user
status: stable
code_verified: 4.0.0a5
---

# 版本策略

## 用途

定义公共 API 稳定性、alpha 状态、HDF5 schema 兼容政策和文档验证版本。

## 稳定性约定

Reference 只描述当前公开行为。参数默认值、单位、返回对象和异常必须与已验证版本一致；planned 或 research 功能不会出现在稳定接口中。

## 诊断顺序

先阅读异常消息和同一调用产生的诊断，再检查输入结构与数据形状。若问题涉及物理近似，应转到 Theory；若涉及具体流程，应转到 How-to。

## 版本与兼容性

内部模块路径不属于公共兼容承诺。用户代码应从 `mlfcs` 顶层导入公开对象，并在保存结果时记录 MLFCS 版本和 schema。
