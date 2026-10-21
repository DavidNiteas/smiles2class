# smiles2class 概览

**中文** | [English](../en/index.md)

`smiles2class` 把 SMILES 转成 **426 个分子类别的概率**。它提供两种**独立**实现：

- **main**：UniMol encoder + 426 个类查询解码器，单模型输出全部类别；
- **vice**：426 个逐类 AutoGluon 分类器，输入为主模型输出的 512 维 embedding。

## 核心特性

- **等长保序**：输出行与输入行一一对应，重复 SMILES 不折叠；
- **显式状态**：非法 SMILES、2D 降级、超时都有 `status`，不静默；
- **两种实现独立**：可以只用 main，也可以额外安装 `[vice]`；
- **可 pip 安装**：hatchling 构建，提供 `smiles2class` 命令行入口；
- **CPU 可用**：默认在 CPU 上推理，`device="cuda"` 可切 GPU。

## 文档地图

| 页面 | 你会在里面找到 |
|---|---|
| [快速开始](getting-started.md) | 安装、模型目录、最小示例、CLI |
| [API](api.md) | 类与函数签名、参数、返回结构、状态模型 |
| [模型说明](models.md) | 结构、标签、概率语义、运行时与限制 |
| [评测](evaluation.md) | 逐类指标、AUROC/AP 口径、评测报告 |
| [CLI 与 FAQ](cli-and-faq.md) | 命令行、常见问题、开发与测试 |

## 版本与许可

当前版本 `0.1.1`；代码 MIT。类名来自 HMDB `Alternative Parents`（ClassyFire 术语），426 个类的选择与顺序是模型的一部分。
