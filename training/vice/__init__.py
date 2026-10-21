"""Vice-model training: one AutoGluon predictor per label over main-model embeddings.

Run from a clone (not part of the installed wheel)::

    python -m training.vice.prepare --input labels.parquet --model-dir <model_dir> \
        --output vice.parquet
    python -m training.vice.train --dataset vice.parquet --label-map <model_dir>/label_map.json \
        --output <model_dir>/vice_model
"""