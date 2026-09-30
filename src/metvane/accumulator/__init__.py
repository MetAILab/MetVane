"""Chunked accumulators for large-scale evaluation.

Classes
-------
ContinuousAccumulator
    Accumulates regression statistics (RMSE, MAE, bias, ACC, per-sample ACC) chunk by chunk.
ContingencyAccumulator
    Accumulates exact int64 TP/FP/FN/TN for categorical metrics chunk by chunk.
FSSAccumulator
    Accumulates pooled Fractions Skill Score sums.

All accumulators: ``axis=`` *or* ``preserve_axes=``; chunks may only be split along reduced
axes (checked); atomic ``update``; ``merge``; ``state_dict`` / ``load_state_dict``; ``stats``.
"""

from .continuous_acc import ContinuousAccumulator
from .contingency_acc import ContingencyAccumulator, FSSAccumulator

__all__ = ["ContinuousAccumulator", "ContingencyAccumulator", "FSSAccumulator"]
