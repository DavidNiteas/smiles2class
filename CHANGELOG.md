# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- 文档明确说明：主模型编码器来自 `unimol-lite`（Uni-Mol 的轻量修改版实现），不是官方 Uni-Mol；依赖也不引入官方实现。
- 文档重构：README 改为中英双语主页，文档索引并入主页，移除 `docs/README.md` 与 `docs/zh|en/index.md`。
- 项目元数据只保留 `authors`，移除 `maintainers`；README 统一使用“作者”。
- 文档主页新增“只想要结果？”路径，明确本项目不提供预训练模型、需先用自有数据训练；新增 `agent.md`，覆盖安装、数据获取、清洗、训练与预测。
- 修正训练与评测文档中的命令：`training.main.prepare --labels`、`--smiles-column SMILES`。
- `.gitignore` 忽略 `data/`、`work/`、`runs/`，避免提交数据与训练产物。
- 文档重构为编号式 `docs/{en,zh-CN}/`，增加语言索引页、每页 Level/Prerequisites/Next 与 `AGENTS.md`。
- 新增 `AUTHORS`、`NOTICE`、`CITATION.cff`；README 改为英文优先的双语结构。
- 包元数据改用 `license = { file = "LICENSE" }`，补充 classifiers、项目主页与 sdist 文件清单。
- 公开文档移除“内部工单/开发笔记”等内部过程表述，改为面向用户的“你需要自备”；移除公开 README 中的 CI 说明。
- 统一 Python 版本说明为 `>=3.10,<3.12`，并注明与 `unimol-lite` 的组合安装范围一致。

## [0.1.1] - 2026-10-22

### Changed

- 安装说明指向 `v0.1.1`。
- `unimol-lite` 依赖固定为 `v0.2.1`。

## [0.1.0] - 2026-10-21

### Added

- `MainClassifier`：UniMol-Lite encoder + 426 类查询解码器，输出 `(N, 426)` 概率（同时给出 512 维 embedding）。
- `ViceClassifier`：复用主模型 embedding，走 426 个 AutoGluon 逐类 predictor，输出 `(N, 426)` 概率。
- 行对齐：输出与输入等长保序；重复 SMILES 不折叠；非法 SMILES 该行全 `NaN` + `status="invalid"`。
- CLI：`smiles2class --model main|vice`。
- `LABELS` / `NUM_LABELS`（426 类，顺序与模型语义绑定）。
- 中英双语文档：`docs/zh-CN/`、`docs/en/`。
- hatchling 构建；`pip install -e ".[dev,vice]"` 可用。

### Notes

- 副模型 426 个 predictor 顺序执行；首次调用可能较慢。
- 模型文件不随仓库分发，`from_pretrained` 指向本地模型目录。
