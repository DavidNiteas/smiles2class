# 模型说明

**中文** | [English](../en/05-models.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [快速开始](./02-quickstart.md)。

## 主模型（main）

> 编码器来自配套库 UniMol-Lite，不是官方 Uni-Mol。UniMol-Lite 是 Uni-Mol 的轻量修改版实现。

- Uni-Mol 风格的 15 层 encoder（512 维，来自 UniMol-Lite）+ 3 层 TransformerDecoder；
- 426 个类查询 token，每个 token 经过 `Linear(512, 2)` 输出该类的“是/否”概率；
- 因此输出是 **426 个独立的二分类概率**，不是 426 类互斥 softmax，跨类不归一；
- 同时给出 `(N, 512)` 的 CLS embedding，供副模型使用。

加载方式：`main_model/config.json` + `main_model/model.safetensors`，`strict` 校验全部张量。

## 副模型（vice）

- 426 个 `autogluon.tabular.TabularPredictor`，每个类一个；
- 每个 predictor 内含 13 个基模型 + 1 个 `WeightedEnsemble`；
- 输入是主模型的 512 维 embedding，列名固定 `vec_0..vec_511`；
- 输出 `(N, 426)` 逐类概率；
- 默认顺序执行：426 个集成逐个推理。

## 标签空间

- 426 个类名来自 HMDB `Alternative Parents`（ClassyFire 术语）；
- 426 个类的**选择与顺序**是模型语义的一部分：`label_map.json` 第 k 行对应输出第 k 列；
- 重排标签、增删类别都必须重训，不能只改 JSON。

## 概率语义

- 两套实现都输出**逐类独立概率**，不做跨类归一；
- 主模型与副模型的训练分布不同（自然分布 vs 平衡采样），**概率口径不同**。

## 运行时与限制

| 项 | 说明 |
|---|---|
| 主模型 | 可在 CPU 或 GPU 上运行 |
| 副模型 | 426 个集成顺序推理；首次调用可能较慢 |
| 副模型目录 | 体积较大，宜放在 SSD/NVMe |
| 模型文件 | 不随仓库分发；需要自行训练，`from_pretrained` 指向训练产出的模型目录 |

- 副模型的主要成本是 426 个集成的计算；
- 服务启动时预热或使用本地缓存，可降低首次调用延迟；
- 耗时取决于硬件、模型目录和输入分布。

**下一页：** [API](./06-api.md)
