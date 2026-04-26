"""
annotation.py - Automated cell-type annotation for spatioloji_s

Wraps the `CellTypist <https://www.celltypist.org/>`_ family of pre-trained
logistic-regression models, applied to a spatioloji object.  CellTypist
expects log1p-normalized counts and uses gene symbols, so this module
operates on a layer (default ``'log_normalized'``) and assumes the gene
index already holds symbols (true for Xenium / CosMx panels).

Optional dependency
-------------------
``celltypist`` and ``anndata`` are not declared in ``pyproject.toml``.
Install separately::

    pip install celltypist

Function-local ``import`` raises a helpful ``ImportError`` if missing.

Quick start
-----------
>>> # 1. (One-time) download a model
>>> sj.processing.download_celltypist_models("Immune_All_Low.pkl")
>>>
>>> # 2. Annotate cells against the downloaded model
>>> sj.processing.celltypist_annotate(
...     sp,
...     model       = "Immune_All_Low.pkl",
...     layer       = "log_normalized",
...     output_column = "celltypist_label",
... )
>>>
>>> # 3. New columns are added to sp.cell_meta:
>>> sp.cell_meta[["celltypist_label", "celltypist_score", "celltypist_majority"]].head()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Optional-dep import helpers
# ---------------------------------------------------------------------------


def _import_celltypist():
    """Import celltypist + anndata, raising a friendly ImportError otherwise."""
    try:
        import anndata  # noqa: F401
        import celltypist
    except ImportError as err:
        raise ImportError(
            "celltypist (and anndata) are required for annotation. Install with: "
            "`pip install celltypist`. CellTypist is not declared as a "
            "spatioloji_s dependency — install it manually in this env."
        ) from err
    return celltypist


# ---------------------------------------------------------------------------
# Model management — thin pass-throughs
# ---------------------------------------------------------------------------


def list_celltypist_models(refresh: bool = False) -> pd.DataFrame:
    """List CellTypist's pre-trained models.

    Args:
        refresh: If True, re-fetch the model index from the CellTypist server.

    Returns:
        DataFrame with one row per available model and columns describing the
        species, tissue, and tag set used to train it.
    """
    celltypist = _import_celltypist()
    desc = celltypist.models.models_description(on_the_fly=refresh)
    # Recent celltypist returns a DataFrame; older versions return a dict.
    if isinstance(desc, pd.DataFrame):
        return desc
    return pd.DataFrame(desc)


def download_celltypist_models(
    model: str | list[str] | None = None,
    force: bool = False,
) -> None:
    """Download one or all CellTypist models into the local model cache.

    Args:
        model: Specific model name (e.g. ``'Immune_All_Low.pkl'``), a list of
            names, or ``None`` to download every model in the index.
        force: If True, redownload even if the file is already cached.
    """
    celltypist = _import_celltypist()
    if model is None:
        celltypist.models.download_models(force_update=force)
    else:
        names = [model] if isinstance(model, str) else list(model)
        celltypist.models.download_models(model=names, force_update=force)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_celltypist_anndata(spatioloji_obj, layer: str | None):
    """Construct a minimal AnnData containing only what CellTypist needs.

    Cells are indexed by ``spatioloji_obj.cell_index`` (as strings, since
    CellTypist requires string ``obs_names``); genes by
    ``spatioloji_obj.gene_index``; ``adata.X`` is the requested layer.
    Sparse storage is preserved.
    """
    import anndata

    if layer is None:
        if spatioloji_obj.expression.is_sparse:
            X = spatioloji_obj.expression.get_sparse()
        else:
            X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)

    if not sparse.issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    obs = pd.DataFrame(index=spatioloji_obj.cell_index.astype(str))
    var = pd.DataFrame(index=spatioloji_obj.gene_index.astype(str))
    return anndata.AnnData(X=X, obs=obs, var=var)


def _check_layer_for_celltypist(spatioloji_obj, layer: str | None) -> None:
    """Warn (don't raise) if the chosen layer name doesn't look log-normalized."""
    import warnings

    if layer is None:
        warnings.warn(
            "celltypist_annotate: layer=None uses the main expression matrix "
            "(typically raw counts). CellTypist models are trained on "
            "log1p(CP10k)-normalized data — pass layer='log_normalized' "
            "(or your equivalent) for accurate predictions.",
            UserWarning,
            stacklevel=3,
        )
        return

    name = layer.lower()
    if "log" not in name:
        warnings.warn(
            f"celltypist_annotate: layer='{layer}' doesn't look log-normalized. "
            "CellTypist models expect log1p(CP10k) input. Run "
            "sj.processing.normalize_total + log_transform first if needed.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def celltypist_annotate(
    spatioloji_obj,
    model: str | Any,
    layer: str | None = "log_normalized",
    majority_voting: bool = True,
    over_clustering: str | None = None,
    p_thres: float = 0.5,
    mode: str = "best match",
    output_column: str = "celltypist_label",
    output_score_column: str = "celltypist_score",
    output_majority_column: str = "celltypist_majority",
    inplace: bool = True,
    **annotate_kwargs: Any,
) -> pd.DataFrame | None:
    """Annotate cells using a pre-trained CellTypist logistic-regression model.

    For each cell, CellTypist computes class probabilities under a one-vs-rest
    logistic-regression model trained on a reference single-cell dataset, then
    optionally refines per-cluster majorities ("majority voting") to suppress
    isolated misclassifications.

    Args:
        spatioloji_obj: spatioloji object with expression data.
        model: A model identifier — one of:

            - ``str`` ending in ``.pkl`` — name of a downloaded CellTypist model
              (e.g. ``'Immune_All_Low.pkl'``).  Use
              :func:`list_celltypist_models` to browse available names.
            - Absolute file path to a custom-trained ``.pkl``.
            - A loaded ``celltypist.Model`` instance (skips disk load).
        layer: Expression layer used as input.  CellTypist expects log1p(CP10k);
            default ``'log_normalized'`` matches the package convention.
            A UserWarning is emitted when the name doesn't look log-normalized.
        majority_voting: If True, refine each cell's prediction by majority vote
            within its over-clustering bin.  Smooths out per-cell flickering at
            the cost of one extra clustering pass.
        over_clustering: Column in ``cell_meta`` providing the over-clustering
            for majority voting (e.g. an existing ``'leiden'`` column).  When
            ``None`` and ``majority_voting=True``, CellTypist computes its own
            over-clustering on the fly.
        p_thres: Minimum probability for a confident call.  Cells whose top
            probability is below ``p_thres`` are labelled ``"Unassigned"``.
        mode: One of ``'best match'`` (single label per cell) or
            ``'prob match'`` (multi-label string when several classes pass
            ``p_thres``).
        output_column: Name of the cell_meta column written with the per-cell
            label (always written).
        output_score_column: Column written with the predicted-class probability
            for the assigned label.
        output_majority_column: Column written with the majority-voted label
            (only present when ``majority_voting=True``).
        inplace: If True, write columns to ``sp.cell_meta`` and return ``None``.
            If False, return a DataFrame indexed by ``cell_index`` with all
            three columns.
        **annotate_kwargs: Forwarded verbatim to ``celltypist.annotate``.

    Returns:
        ``None`` (when ``inplace=True``) or a DataFrame.

    Raises:
        ImportError: If celltypist / anndata are not installed.
        ValueError: If ``over_clustering`` is given but the column isn't in
            ``cell_meta``.

    Examples:
        >>> # Default — Immune Low-resolution model on log-normalized counts
        >>> sj.processing.celltypist_annotate(sp, model="Immune_All_Low.pkl")
        >>>
        >>> # Use existing leiden as the majority-voting over-clustering
        >>> sj.processing.celltypist_annotate(
        ...     sp, model="Immune_All_Low.pkl",
        ...     over_clustering="leiden",
        ... )
        >>>
        >>> # Custom-trained model from disk, no majority voting
        >>> sj.processing.celltypist_annotate(
        ...     sp, model="/path/to/my_panel.pkl",
        ...     majority_voting=False,
        ... )
    """
    celltypist = _import_celltypist()

    _check_layer_for_celltypist(spatioloji_obj, layer)

    if over_clustering is not None and over_clustering not in spatioloji_obj.cell_meta.columns:
        raise ValueError(
            f"over_clustering column '{over_clustering}' not found in cell_meta. "
            f"Available: {list(spatioloji_obj.cell_meta.columns)}"
        )

    print(
        f"\nCellTypist annotation "
        f"(model={model!r}, majority_voting={majority_voting}, "
        f"over_clustering={over_clustering!r})"
    )

    adata = _build_celltypist_anndata(spatioloji_obj, layer)

    # Attach an over-clustering column on the AnnData so CellTypist can
    # re-use it for majority voting instead of computing its own.
    if over_clustering is not None:
        adata.obs["over_clustering"] = (
            spatioloji_obj.cell_meta[over_clustering].astype(str).values
        )
        ov_arg: str | bool = "over_clustering"
    else:
        ov_arg = True if majority_voting else False  # CellTypist default

    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=majority_voting,
        over_clustering=ov_arg,
        p_thres=p_thres,
        mode=mode,
        **annotate_kwargs,
    )

    pred_df = predictions.predicted_labels  # DataFrame indexed by obs_names
    n_cells = adata.n_obs

    # `predicted_labels` is the column with the per-cell call.
    label_series = pred_df["predicted_labels"].astype(str)

    # Per-cell probability for the assigned label (max across classes).
    prob_matrix = predictions.probability_matrix
    if prob_matrix is not None and not prob_matrix.empty:
        score_series = prob_matrix.max(axis=1).astype(float)
    else:
        score_series = pd.Series(np.nan, index=pred_df.index)

    if majority_voting and "majority_voting" in pred_df.columns:
        majority_series = pred_df["majority_voting"].astype(str)
    else:
        majority_series = None

    n_unique = label_series.nunique()
    n_unassigned = int((label_series == "Unassigned").sum())
    print(f"  ✓ Annotated {n_cells:,} cells into {n_unique} cell types "
          f"({n_unassigned:,} unassigned at p<{p_thres}).")
    if majority_series is not None:
        print(f"    Majority-voted unique types: {majority_series.nunique()}")

    out_df = pd.DataFrame(
        {
            output_column: label_series.values,
            output_score_column: score_series.values,
        },
        index=spatioloji_obj.cell_index,
    )
    if majority_series is not None:
        out_df[output_majority_column] = majority_series.values

    if inplace:
        for col in out_df.columns:
            spatioloji_obj._cell_meta[col] = out_df[col].values
        return None
    return out_df
