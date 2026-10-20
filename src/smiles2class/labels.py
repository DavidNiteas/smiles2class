"""Label vocabulary (426 classes).

The order is part of the model artifact: index ``i`` must stay aligned with the
weights in ``main_model/`` and the predictor directories in ``vice_model/``.
"""

from __future__ import annotations

import json
from pathlib import Path

LABELS: list[str] = json.loads(Path(__file__).with_name("labels.json").read_text())
NUM_LABELS = len(LABELS)

__all__ = ["LABELS", "NUM_LABELS"]
