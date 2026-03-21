# src/spatioloji_s/spatial/_gradient.py
"""Spatial gradient analysis.

Shared by both point and polygon modules. Uses only cell centroid
coordinates — no polygon geometry or spatial graph required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from spatioloji_s.spatial._distance_utils import signed_distance_to_interface
from spatioloji_s.spatial._gradient_types import GradientResult
from spatioloji_s.spatial._interface_types import InterfaceResult


def _fit_gradient(values: np.ndarray, distances: np.ndarray) -> dict:
    """Fit linear regression of values ~ distances.

    Args:
        values: Expression values (1-D array).
        distances: Signed distances (1-D array, same length).

    Returns:
        Dict with keys coef, pvalue, r2, trend.
    """
    mask = np.isfinite(values) & np.isfinite(distances)
    if mask.sum() < 3:
        return {"coef": np.nan, "pvalue": np.nan, "r2": np.nan, "trend": "flat"}

    result = linregress(distances[mask], values[mask])
    trend = "flat"
    if result.pvalue < 0.05:
        trend = "increasing_toward_a" if result.slope > 0 else "increasing_toward_b"
    return {
        "coef": result.slope,
        "pvalue": result.pvalue,
        "r2": result.rvalue**2,
        "trend": trend,
    }


def _bin_expression(
    expr_df: pd.DataFrame,
    distances: pd.Series,
    genes: list[str],
    n_bins: int,
) -> pd.DataFrame:
    """Bin expression by distance for plotting.

    Args:
        expr_df: Expression DataFrame (cells x genes).
        distances: Signed distances indexed by cell ID.
        genes: Gene names to include.
        n_bins: Number of equal-width bins.

    Returns:
        Long-form DataFrame: distance_bin, gene, mean_expr, std_expr.
    """
    bin_edges = np.linspace(distances.min(), distances.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_labels = np.digitize(distances.values, bin_edges[1:-1])  # 0-indexed bins

    rows = []
    for gene in genes:
        gene_vals = expr_df[gene].values
        for b in range(n_bins):
            mask = bin_labels == b
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "distance_bin": bin_centers[b],
                    "gene": gene,
                    "mean_expr": float(np.mean(gene_vals[mask])),
                    "std_expr": float(np.std(gene_vals[mask])),
                }
            )
    return pd.DataFrame(rows)


def _discover_programs_nmf(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via NMF.

    Args:
        expr_matrix: Dense expression matrix (cells x genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for auto_programs='nmf'. Install with: pip install scikit-learn"
        ) from err

    X = np.maximum(expr_matrix, 0)
    n_programs = min(n_programs, min(X.shape))
    model = NMF(n_components=n_programs, random_state=42, max_iter=500)
    model.fit(X)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(component)[::-1][:n_top]
        programs[f"NMF_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def _discover_programs_pca(expr_matrix: np.ndarray, gene_names: list[str], n_programs: int) -> dict[str, list[str]]:
    """Discover gene programs via PCA loadings.

    Args:
        expr_matrix: Dense expression matrix (cells x genes).
        gene_names: Gene names.
        n_programs: Number of programs.

    Returns:
        Dict of {program_name: [top_gene_names]}.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for auto_programs='pca'. Install with: pip install scikit-learn"
        ) from err

    n_programs = min(n_programs, min(expr_matrix.shape))
    model = PCA(n_components=n_programs, random_state=42)
    model.fit(expr_matrix)

    programs = {}
    n_top = max(3, len(gene_names) // n_programs)
    for i, component in enumerate(model.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:n_top]
        programs[f"PC_{i}"] = [gene_names[j] for j in top_idx]
    return programs


def compute_gradient(
    sp,
    interface_result: InterfaceResult,
    genes: list[str] | None = None,
    features: list[str] | None = None,
    layer: str | None = None,
    cells: str | list[str] | None = None,
    cell_types: list[str] | None = None,
    cell_type_col: str | None = None,
    programs: dict[str, list[str]] | None = None,
    n_bins: int = 20,
    auto_programs: str | None = None,
    n_auto_programs: int = 5,
    n_jobs: int = 1,
    coord_type: str = "global",
    unsigned: bool = False,
) -> GradientResult:
    """Compute spatial expression gradient relative to an interface.

    Fits linear regression of gene expression (or cell_meta features) vs
    signed distance from the interface contour.

    Works identically for point-based and polygon-based data — only
    cell centroid coordinates are used.

    Args:
        sp: spatioloji object.
        interface_result: Result from ``identify_interface``.
        genes: List of gene names to analyze. ``None`` = all genes
            (only when ``features`` is also ``None``).
        features: List of numeric ``cell_meta`` column names to include
            in the gradient analysis (e.g. ``['morph_area',
            'aucell_EMT']``).  These are appended alongside genes.
        layer: Expression layer to use (e.g. ``'log_normalized'``,
            ``'scaled'``).  If ``None``, uses raw expression.
        cells: Which cells to include in the gradient fit.  Can be:

            - ``None`` (default) — all cells.
            - ``'regions_only'`` — only cells in region A or B
              (excludes "other" cells).
            - ``'interface_only'`` — only cells labeled as interface.
            - A list of cell IDs — explicit cell subset.
        cell_types: List of cell type labels to include (e.g.
            ``['Tumor', 'Stroma', 'CD8_T']``).  Only cells matching
            these types are used for the gradient fit.  Requires
            ``cell_type_col``.
        cell_type_col: Column in ``cell_meta`` containing cell type
            labels.  Required when ``cell_types`` is set.
        programs: Dict of ``{name: [gene_list]}`` for user-defined
            gene modules. ``None`` = skip user programs.
        n_bins: Number of equal-width distance bins for curves.
        auto_programs: ``"nmf"`` or ``"pca"`` to auto-discover gene
            programs. ``None`` = skip auto-discovery.
        n_auto_programs: Number of programs to discover (default 5).
        n_jobs: Number of parallel workers for gradient fitting.
            ``1`` = sequential, ``-1`` = all CPUs.  Useful when
            analyzing many genes/features.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        unsigned: If True, use absolute distances.

    Returns:
        GradientResult with gene/program gradients and binned data.

    Raises:
        ValueError: If no valid genes or features remain after filtering.

    Examples:
        >>> result = compute_gradient(sp, interface, genes=['EPCAM', 'VIM'],
        ...     layer='log_normalized')
    """
    if auto_programs is not None and auto_programs not in ("nmf", "pca"):
        raise ValueError(f"auto_programs must be 'nmf', 'pca', or None, got '{auto_programs}'")

    distances = signed_distance_to_interface(
        sp,
        interface_result,
        coord_type=coord_type,
        unsigned=unsigned,
    )

    if layer is not None:
        from scipy import sparse as sp_sparse

        X = sp.get_layer(layer)
        if sp_sparse.issparse(X):
            X = X.toarray()
        expr_df = pd.DataFrame(X, index=sp.cell_index, columns=sp.gene_index)
    else:
        expr_df = sp.expression.to_dataframe()

    # --- Cell filtering ---
    if cell_types is not None:
        if cell_type_col is None:
            raise ValueError("cell_type_col is required when cell_types is set")
        if cell_type_col not in sp.cell_meta.columns:
            raise ValueError(f"'{cell_type_col}' not found in cell_meta")
        type_mask = sp.cell_meta[cell_type_col].isin(cell_types)
        keep_ids = sp.cell_index[type_mask]
        print(f"  Filtering to {len(keep_ids)} cells of types {cell_types}")
        expr_df = expr_df.loc[keep_ids]
        distances = distances.loc[keep_ids]
    elif isinstance(cells, str):
        labels = interface_result.cell_labels
        if cells == "regions_only":
            keep = labels.isin(["region_a_interface", "region_b_interface", "interior_a", "interior_b"])
            keep_ids = labels.index[keep]
            print(f"  Filtering to {len(keep_ids)} region A + B cells (excluding 'other')")
        elif cells == "interface_only":
            keep = labels.isin(["region_a_interface", "region_b_interface"])
            keep_ids = labels.index[keep]
            print(f"  Filtering to {len(keep_ids)} interface cells only")
        else:
            raise ValueError(f"cells must be None, 'regions_only', 'interface_only', or a list, got '{cells}'")
        expr_df = expr_df.reindex(keep_ids).dropna(how="all")
        distances = distances.reindex(keep_ids).dropna()
    elif isinstance(cells, list):
        keep_ids = pd.Index(cells).intersection(expr_df.index)
        print(f"  Filtering to {len(keep_ids)} cells from provided list")
        expr_df = expr_df.loc[keep_ids]
        distances = distances.loc[keep_ids]

    import os
    import warnings

    all_genes = list(expr_df.columns)

    # Resolve which genes to analyze — skip missing with warning
    if genes is None and features is None:
        genes = all_genes
    elif genes is None:
        genes = []
    else:
        missing = [g for g in genes if g not in all_genes]
        if missing:
            warnings.warn(
                f"Skipping {len(missing)} genes not found in expression: {missing}",
                UserWarning,
                stacklevel=2,
            )
            genes = [g for g in genes if g in all_genes]

    # Append cell_meta features — skip missing with warning
    if features is not None:
        missing_feat = [f for f in features if f not in sp.cell_meta.columns]
        if missing_feat:
            warnings.warn(
                f"Skipping {len(missing_feat)} features not found in cell_meta: {missing_feat}",
                UserWarning,
                stacklevel=2,
            )
            features = [f for f in features if f in sp.cell_meta.columns]
        for feat in features:
            feat_vals = sp.cell_meta[feat].reindex(expr_df.index).values.astype(float)
            expr_df[feat] = feat_vals
        all_targets = genes + features
    else:
        all_targets = genes

    if not all_targets:
        raise ValueError("No valid genes or features remaining after filtering.")

    # Fit gradients — parallel when n_jobs != 1
    dist_arr = distances.values

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and len(all_targets) > 1:
        from concurrent.futures import ProcessPoolExecutor

        print(f"  Fitting {len(all_targets)} gradients with {n_jobs} workers...")

        # Prepare data chunks for parallel workers
        val_arrays = [expr_df[t].values for t in all_targets]

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            fit_results = list(pool.map(_fit_gradient, val_arrays, [dist_arr] * len(all_targets)))

        gene_rows = [{"gene": t, **r} for t, r in zip(all_targets, fit_results, strict=True)]
    else:
        gene_rows = []
        for target in all_targets:
            vals = expr_df[target].values
            gene_rows.append({"gene": target, **_fit_gradient(vals, dist_arr)})

    gene_gradients = pd.DataFrame(gene_rows).set_index("gene")

    all_programs = dict(programs) if programs else {}

    if auto_programs == "nmf":
        # Use only gene columns for program discovery (not cell_meta features)
        auto = _discover_programs_nmf(
            expr_df[all_genes].values,
            all_genes,
            n_auto_programs,
        )
        all_programs.update(auto)
    elif auto_programs == "pca":
        auto = _discover_programs_pca(
            expr_df[all_genes].values,
            all_genes,
            n_auto_programs,
        )
        all_programs.update(auto)

    program_rows = []
    program_score_dict = {}
    for pname, pgenes in all_programs.items():
        valid = [g for g in pgenes if g in all_genes]
        if not valid:
            continue
        scores = expr_df[valid].mean(axis=1).values
        program_score_dict[pname] = scores
        program_rows.append({"program": pname, **_fit_gradient(scores, dist_arr)})

    if program_rows:
        program_gradients = pd.DataFrame(program_rows).set_index("program")
        program_scores = pd.DataFrame(program_score_dict, index=expr_df.index)
    else:
        program_gradients = pd.DataFrame(columns=["coef", "pvalue", "r2", "trend"])
        program_scores = pd.DataFrame(index=expr_df.index)

    bins = _bin_expression(expr_df, distances, all_targets, n_bins)

    if program_score_dict:
        prog_df = pd.DataFrame(program_score_dict, index=expr_df.index)
        prog_bins = _bin_expression(prog_df, distances, list(program_score_dict.keys()), n_bins)
        bins = pd.concat([bins, prog_bins], ignore_index=True)

    return GradientResult(
        distances=distances,
        gene_gradients=gene_gradients,
        program_gradients=program_gradients,
        program_scores=program_scores,
        bins=bins,
        region_a=interface_result.region_a,
        region_b=interface_result.region_b,
    )
