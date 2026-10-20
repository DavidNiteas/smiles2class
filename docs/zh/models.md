# 模型说明

**中文** | [English](../en/models.md)

## 主模型（main）

- Uni-Mol 15 层 encoder（512 维）+ 3 层 TransformerDecoder；
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

## 性能与限制

| 项 | 参考值 |
|---|---|
| 主模型单分子（CPU） | 约 75 ms |
| 副模型单次预测（CPU） | 约 47 秒 |
| 副模型首次调用 | 可能更久，取决于磁盘与缓存 |
| 副模型目录大小 | 约 15 GB |

- 副模型的主要成本是 426 个集成的计算；
- 建议把副模型目录放在 SSD/NVMe，或在服务启动时预热；
- 模型文件不随仓库分发，`from_pretrained` 指向本地模型目录。
