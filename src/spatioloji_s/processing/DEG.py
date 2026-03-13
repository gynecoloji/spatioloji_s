"""
DEG.py - Differentially Expressed Gene analysis for spatial transcriptomics

Supports five statistical methods:
  - Wilcoxon rank-sum test (scipy.stats.mannwhitneyu)
  - Student's t-test (scipy.stats.ttest_ind)
  - MAST-inspired hurdle model (statsmodels; optional [deg])
  - Negative-binomial GLM (statsmodels; optional [deg])
  - DESeq2 pseudobulk (pydeseq2; optional [deg])

Scalability design
------------------
  Wilcoxon / t-test   Gene-chunked loops; t-test is fully vectorized via
                      scipy axis=0. Wilcoxon uses per-gene ThreadPoolExecutor.
  NB-GLM / MAST       Per-gene model fitting parallelized via ThreadPoolExecutor.
  DESeq2              Pseudobulk aggregation collapses millions of cells to
                      n_replicates × n_genes before calling pydeseq2.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _n_workers(n_jobs: int) -> int:
    """Resolve n_jobs to a positive thread count (matches normalization.py pattern)."""
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _get_X(spatioloji_obj, layer: str | None):
    """Return expression matrix without unnecessary copies.

    This is an intentional per-module copy of the pattern established in
    normalization.py. Each processing module defines its own private `_get_X`
    rather than importing across module boundaries, following the project's
    convention of keeping private helpers module-local.
    """
    if layer is None:
        if spatioloji_obj.expression.is_sparse:
            return spatioloji_obj.expression.get_sparse()
        return spatioloji_obj.expression.get_dense()
    return spatioloji_obj.get_layer(layer)
