"""Spatial verification metrics.

Functions
---------
fss                  — Fractions Skill Score (pooled over leading axes)
fss_components       — additive sums (num, den, base-rate counts) for pooling / multiple
                       thresholds and windows
fss_from_components  — FSS from (pooled) sums
"""

from ._impl import fss, fss_components, fss_from_components

__all__ = ["fss", "fss_components", "fss_from_components"]
