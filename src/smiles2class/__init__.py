"""smiles2class: SMILES → molecular class probabilities.

Two independent implementations are exposed:

- :class:`MainClassifier` — UniMol encoder + 426 class-query decoder;
- :class:`ViceClassifier` — 426 per-class AutoGluon predictors over the main
  model embedding.
"""

from .classifier import MainClassifier, ViceClassifier
from .labels import LABELS, NUM_LABELS

__version__ = "0.1.1"

__all__ = ["MainClassifier", "ViceClassifier", "LABELS", "NUM_LABELS", "__version__"]
