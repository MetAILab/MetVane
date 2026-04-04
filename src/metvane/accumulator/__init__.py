"""Chunked accumulators for large-scale evaluation.

Classes
-------
ContinuousAccumulator
    Accumulates regression statistics (RMSE, MAE, …) chunk by chunk.
ContingencyAccumulator
    Accumulates TP/FP/FN/TN for categorical metrics chunk by chunk.
"""

from .continuous_acc import ContinuousAccumulator
from .contingency_acc import ContingencyAccumulator

__all__ = ["ContinuousAccumulator", "ContingencyAccumulator"]
