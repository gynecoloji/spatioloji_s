"""Shared data structures for immune infiltration scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class InfiltrationResult:
    """Container for immune infiltration scoring results.

    Attributes:
        distances: Signed distance per cell to the interface contour.
        cell_classifications: Series with values ``"infiltrating"``,
            ``"resident"``, or ``"other"`` per cell.
        per_type_metrics: DataFrame with rows=immune cell types and
            columns ``median_depth``, ``max_depth``, ``density_slope``,
            ``density_pvalue``, ``infiltration_fraction``,
            ``n_infiltrating``, ``n_resident``.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
        target_region: Which region immune cells infiltrate into.
    """

    distances: pd.Series
    cell_classifications: pd.Series
    per_type_metrics: pd.DataFrame
    region_a: str | list[str]
    region_b: str | list[str]
    target_region: str
