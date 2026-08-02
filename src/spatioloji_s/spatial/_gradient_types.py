"""Shared data structures for gradient analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GradientResult:
    """Container for spatial gradient analysis results.

    Attributes:
        distances: Signed distance per cell to the interface contour.
            Positive = region A side, negative = region B side.
        gene_gradients: DataFrame with rows=genes and columns
            ``coef``, ``pvalue``, ``r2``, ``trend``.
        program_gradients: DataFrame with rows=programs and columns
            ``coef``, ``pvalue``, ``r2``, ``trend``.
        program_scores: DataFrame (cells × programs) with per-cell
            mean expression of each gene program.
        bins: Long-form DataFrame with columns ``distance_bin``,
            ``gene``, ``mean_expr``, ``std_expr`` for plotting.
        region_a: Region A label(s) used.
        region_b: Region B label(s) used.
    """

    distances: pd.Series
    gene_gradients: pd.DataFrame
    program_gradients: pd.DataFrame
    program_scores: pd.DataFrame
    bins: pd.DataFrame
    region_a: str | list[str]
    region_b: str | list[str]
