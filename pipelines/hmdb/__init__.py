"""HMDB / LIPID MAPS data preparation entry points.

Modules:

- :mod:`pipelines.hmdb.clean`  -- wash HMDB, convert LIPID MAPS, merge;
- :mod:`pipelines.hmdb.labels` -- 426-label multi-hot table;
- :mod:`pipelines.hmdb.split`  -- multi-label greedy train/vali/test split.

Run them as ``python -m pipelines.hmdb.<module> --help`` from the repo root.
"""
