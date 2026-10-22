# 安装

**中文** | [English](../en/01-installation.md) · [返回主页](../../README.md#zh-cn)

**层级：** 1 — 开始

**前置条件：** Python 3.10–3.11 与 `pip`。

## 从 Git 安装

只使用主模型：

```shell
python -m pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

同时使用副模型：

```shell
python -m pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

已配置 SSH 密钥时，也可使用 `git+ssh://git@github.com/DavidNiteas/smiles2class.git@v0.1.1`。

安装时会一并安装 `unimol-lite@v0.2.1`。UniMol-Lite 是 Uni-Mol 的轻量修改版实现，不是官方 Uni-Mol。

## 开发安装

```shell
git clone https://github.com/DavidNiteas/smiles2class.git
cd smiles2class
python -m pip install -e ".[dev]"
```

需要副模型时：

```shell
python -m pip install -e ".[dev,vice]"
```

## 可选依赖

| extra | 用途 |
|---|---|
| `vice` | 426 个 AutoGluon 逐类 predictor |
| `data` | HMDB / LIPID MAPS 数据管线（`pyarrow`、`PyYAML`） |
| `train` | 主模型与副模型训练（PyTorch Lightning、AutoGluon） |
| `eval` | 逐类评测（scikit-learn） |
| `dev` | 测试与开发工具 |

## 验证安装

```shell
python -c "import smiles2class; print(smiles2class.__version__)"
smiles2class --help
```

## 模型文件

本项目不提供预训练模型，也不随仓库分发模型权重。模型目录需要按[训练](04-training.md)流程自行训练生成：

```text
<model_dir>/
├── label_map.json
├── main_model/
└── vice_model/        # 仅 vice 需要
```

## 构建 wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/smiles2class-*.whl
```

## 常见安装问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: autogluon` | 安装 `.[vice]` 或 `.[train,vice]` |
| `ModuleNotFoundError: rdkit` | 检查 Python 版本与网络，重新安装依赖 |
| Python 3.12 及以上 | 使用 Python 3.10 或 3.11 |
| 构建失败 | 升级 `pip`，确认 `hatchling` 可用 |

**下一页：** [快速开始](./02-quickstart.md)
