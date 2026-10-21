# evaluation

逐类评测工具，不随 wheel 安装。需要 `pip install -e ".[eval]"`（scikit-learn）。

- `metrics.py`：逐类 precision / recall / F1 / AP / ROC AUC / balanced accuracy，macro（排除 `NaN`）与 micro（展平混淆矩阵）汇总；
- `evaluate.py`：`evaluate_predictions()` 与命令行入口，读真值表 + 已有预测或直接加载 `main` / `vice` 模型，写 `<prefix>_per_class.csv` 与 `<prefix>_summary.json`。

```bash
python -m evaluation.evaluate --input labels.csv --predictions preds.csv --output report
python -m evaluation.evaluate --input labels.csv --model main --model-path MODEL_DIR --output report
PYTHONPATH=src:. pytest tests/evaluation/ -q
```

AUROC 与 AP 使用原始概率，不用阈值化后的 0/1；零正例类记 `NaN` 并在 macro 中排除。完整口径与报告字段见 `docs/zh/evaluation.md`、`docs/en/evaluation.md`。运行产物放 `reports/`（gitignore）。
