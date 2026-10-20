# Changelog

## [0.1.0] - 2026-10-21

### Added

- `MainClassifier`：UniMol encoder + 426 类查询解码器，输出 `(N, 426)` 概率（内部同时给出 512 维 embedding）。
- `ViceClassifier`：复用主模型 embedding，走 426 个 AutoGluon 逐类 predictor，输出 `(N, 426)` 概率。
- 行对齐：输出与输入等长保序；重复 SMILES 不折叠；非法 SMILES 该行全 `NaN` + `status="invalid"`。
- CLI：`smiles2class predict --model main|vice`。
- `LABELS` / `NUM_LABELS`（426 类，顺序与模型语义绑定）。
- 中英双语文档：`docs/zh/`、`docs/en/`。
- hatchling 构建；`pip install -e ".[dev,vice]"` 可用。

### Notes

- 副模型 426 个 predictor 顺序执行；机械盘冷启动为分钟级。
- 模型文件不随仓库分发，`from_pretrained` 指向本地模型目录。
