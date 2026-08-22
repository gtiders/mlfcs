---
title: 文档规范
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 文档规范

定义页面模板、术语、状态元数据、双语镜像、代码验证、链接和美元符号数学公式规范。

## 权威内容源

- 两种语言首页维护项目介绍，并生成 README 核心区域。
- Theory 维护推导，Concepts 维护对象模型，Tutorials 维护学习工作流，How-to 维护具体任务。
- Reference 维护公共签名，Examples 维护可复现证据，Roadmap 维护未实现工作。

## 页面模板

Theory 页面使用“动机、定义、推导、在 MLFCS 中的实现、数值注意、相关页面、参考文献”。Tutorial 使用“目标、前置条件、准备、步骤、结果、解释、常见问题、下一步”。How-to 使用“问题、解决方法、解释、限制、相关页面”。

## 数学与语言

Markdown 行内数学使用 `$...$`，独立公式使用 `$$...$$`。中英文页面共享路径、标题层级、公式、代码、状态和案例语义；翻译不得改变数值约定。

## 维护

公共 API 修改必须在同一提交中更新 Reference 和所有受影响工作流。算法修改必须在发布前更新 Theory、Validation 和对应案例证据。
