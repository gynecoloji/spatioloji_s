"""
feature_selection.py - Feature selection methods for spatial transcriptomics

Provides methods for selecting highly variable genes before dimensionality
reduction and clustering.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

# Methods grouped by the type of input they require
_RAW_COUNT_METHODS = {"deviance", "pearson_residuals", "cell_ranger"}
_LOG_NORM_METHODS = {"seurat", "seurat_v3", "spatial_moran"}


def _validate_layer_for_method(method: str, layer: str | None) -> None:
    """Warn when *layer* looks incompatible with *method*'s expected input.

    Parameters
    ----------
    method : str
        HVG method name.
    layer : str or None
        Layer name passed by the caller (None = main expression matrix).
    """
    layer_lower = (layer or "").lower()
    is_log = any(kw in layer_lower for kw in ("log",))
    is_scaled = "scaled" in layer_lower

    if method in _RAW_COUNT_METHODS:
        if is_log or is_scaled:
            warnings.warn(
                f"Method '{method}' requires raw counts, but layer='{layer}' "
                "appears to be log-transformed or scaled. "
                "Pass the raw count layer (e.g. layer=None or layer='raw_counts').",
                UserWarning,
                stacklevel=3,
            )

    elif method in _LOG_NORM_METHODS:
        if layer is None:
            warnings.warn(
                f"Method '{method}' requires log-normalised data, but layer=None "
                "uses the main expression matrix which is typically raw counts. "
                "Pass layer='log_normalized' (or the name of your log layer).",
                UserWarning,
                stacklevel=3,
            )
        elif not is_log:
            warnings.warn(
                f"Method '{method}' requires log-normalised data, but layer='{layer}' "
                "does not appear to be log-transformed. "
                "Consider using a log-normalised layer (e.g. 'log_normalized').",
                UserWarning,
                stacklevel=3,
            )


def highly_variable_genes(
    spatioloji_obj,
    layer: str | None = None,
    n_top_genes: int = 2000,
    method: Literal["seurat", "cell_ranger", "seurat_v3", "pearson_residuals", "deviance", "spatial_moran"] = "seurat",
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "pearson_residuals", "deviance", "spatial_moran"] = "seurat",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_bins: int = 20,
    n_spatial_neighbors: int = 15,
    output_column: str = "highly_variable",
    inplace: bool = True,
):
    """
    Identify highly variable genes (HVGs) using one of six methods.

    Selects genes that show high variability across cells, which typically carry
    the most biological signal. Run before PCA and clustering.

    Args:
        spatioloji_obj: Spatioloji object with expression data.
        layer: Expression layer to use. If None, uses the main expression matrix.
            **Layer requirements by method:**

            - Raw counts required (integer or CPM before log):
              ``'deviance'``, ``'pearson_residuals'``, ``'cell_ranger'``
            - Log-normalised data required (after ``log_transform``):
              ``'seurat'``, ``'seurat_v3'``, ``'spatial_moran'``

            A ``UserWarning`` is raised when the layer name looks incompatible.
        n_top_genes: Number of highly variable genes to select, by default 2000.
        method: HVG selection method. One of:

            - ``'seurat'`` / ``'cell_ranger'``: Mean-variance dispersion with
              MAD-normalised bins (Seurat v1/v2).
            - ``'seurat_v3'``: Variance-stabilising transformation via polynomial
              regression on the log-log mean-variance curve (Hafemeister & Satija 2019).
            - ``'pearson_residuals'``: Analytic Pearson residuals under a Poisson
              null model; selects by residual variance (Lause et al. 2021).
            - ``'deviance'``: Binomial deviance from the depth-only null; higher
              deviance = more information beyond sequencing depth (Townes et al. 2019).
            - ``'spatial_moran'``: Selects spatially variable genes by Moran's I
              on a k-NN spatial weight matrix. Requires spatial coordinates.

        flavor: Alias for method (scanpy compatibility).
        min_mean: Minimum mean expression cutoff for ``'seurat'``, by default 0.0125.
        max_mean: Maximum mean expression cutoff for ``'seurat'``, by default 3.0.
        min_disp: Minimum normalised dispersion for ``'seurat'``, by default 0.5.
        n_bins: Number of mean-expression bins for ``'seurat'``, by default 20.
        n_spatial_neighbors: k for the k-NN spatial weight matrix used by
            ``'spatial_moran'``, by default 15.
        output_column: Column name written to gene_meta, by default 'highly_variable'.
        inplace: If True, writes results to gene_meta and returns None.
            If False, returns a DataFrame with per-gene statistics.

    Returns:
        None if inplace=True; otherwise a pd.DataFrame with per-gene statistics
        and a boolean ``'highly_variable'`` column.

    Raises:
        ValueError: If an unknown method is specified, or if ``'spatial_moran'`` is
            requested but spatial coordinates cannot be found.

    Example:
        >>> highly_variable_genes(sp, layer='normalized_counts', method='seurat_v3')
        >>> highly_variable_genes(sp, method='pearson_residuals', n_top_genes=1500)
        >>> hvg_genes = sp.gene_meta[sp.gene_meta['highly_variable']].index

    References:
        Seurat: Stuart et al. (2019) Cell
        Seurat v3 VST: Hafemeister & Satija (2019) Genome Biology
        Pearson residuals: Lause et al. (2021) Genome Biology
        Binomial deviance: Townes et al. (2019) Genome Biology
    """
    if flavor != "seurat":
        method = flavor

    _validate_layer_for_method(method, layer)
    print(f"\nIdentifying highly variable genes (method={method})")

    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    n_cells, n_genes = X.shape
    mean = X.mean(axis=0)
    var = X.var(axis=0)

    hvg_df = pd.DataFrame({"means": mean, "variances": var}, index=spatioloji_obj.gene_index)

    if method in ("seurat", "cell_ranger"):
        print(f"  Using Seurat method (n_top_genes={n_top_genes})")

        mean_nonzero = mean.copy()
        mean_nonzero[mean_nonzero == 0] = 1e-12
        dispersion = var / mean_nonzero

        hvg_df["dispersions"] = dispersion
        hvg_df["mean_bin"] = pd.cut(mean, bins=n_bins, labels=False)

        disp_norm = np.zeros(n_genes)
        for bin_idx in range(n_bins):
            bin_mask = hvg_df["mean_bin"] == bin_idx
            if bin_mask.sum() > 1:
                bin_disp = dispersion[bin_mask]
                disp_median = np.median(bin_disp)
                disp_mad = np.median(np.abs(bin_disp - disp_median))
                disp_norm[bin_mask] = (bin_disp - disp_median) / (disp_mad + 1e-12)

        hvg_df["dispersions_norm"] = disp_norm
        gene_subset = (mean > min_mean) & (mean < max_mean) & (disp_norm > min_disp)
        hvg_df["highly_variable"] = False

        if gene_subset.sum() > n_top_genes:
            ranks = rankdata(-disp_norm[gene_subset])
            top_genes_mask = ranks <= n_top_genes
            subset_indices = np.where(gene_subset)[0]
            hvg_indices = subset_indices[top_genes_mask]
            hvg_df.iloc[hvg_indices, hvg_df.columns.get_loc("highly_variable")] = True
        else:
            hvg_df.loc[gene_subset, "highly_variable"] = True

    elif method == "seurat_v3":
        print(f"  Using Seurat v3 VST method (n_top_genes={n_top_genes})")
        # Fit polynomial regression log10(var) ~ poly(log10(mean), 2) to model the
        # mean-variance trend, then normalise variance by the expected value from the fit.
        valid = mean > 0
        log_mean = np.zeros(n_genes)
        log_var = np.zeros(n_genes)
        log_mean[valid] = np.log10(mean[valid])
        log_var[valid] = np.log10(np.clip(var[valid], 1e-12, None))

        if valid.sum() > 10:
            coeffs = np.polyfit(log_mean[valid], log_var[valid], deg=2)
            log_var_expected = np.polyval(coeffs, log_mean)
        else:
            log_var_expected = log_mean  # fallback: Poisson

        var_expected = 10.0**log_var_expected
        var_norm = var / (var_expected + 1e-12)
        var_norm = np.clip(var_norm, 0.0, float(np.sqrt(n_cells)))

        hvg_df["variances_norm"] = var_norm
        hvg_df["highly_variable"] = False
        top_indices = np.argsort(-var_norm)[:n_top_genes]
        hvg_df.iloc[top_indices, hvg_df.columns.get_loc("highly_variable")] = True

    elif method == "pearson_residuals":
        print(f"  Using Pearson residuals method (n_top_genes={n_top_genes})")
        # Analytic Pearson residuals under Poisson null: μ_gc = n_c * p_g
        # Processed in gene batches to limit peak memory usage.
        X_f = X.astype(np.float32)
        cell_totals = X_f.sum(axis=1)  # (n_cells,)
        total_sum = float(cell_totals.sum())
        gene_fractions = X_f.sum(axis=0) / (total_sum + 1e-10)  # (n_genes,)
        clip_val = float(np.sqrt(n_cells))
        resid_var = np.zeros(n_genes, dtype=np.float32)

        batch_size = 256
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            X_b = X_f[:, start:end]
            mu_b = cell_totals[:, np.newaxis] * gene_fractions[np.newaxis, start:end]
            r_b = (X_b - mu_b) / np.sqrt(mu_b + 1e-10)
            r_b = np.clip(r_b, -clip_val, clip_val)
            resid_var[start:end] = r_b.var(axis=0)

        hvg_df["pearson_residual_variance"] = resid_var.astype(float)
        hvg_df["highly_variable"] = False
        top_indices = np.argsort(-resid_var)[:n_top_genes]
        hvg_df.iloc[top_indices, hvg_df.columns.get_loc("highly_variable")] = True

    elif method == "deviance":
        print(f"  Using binomial deviance method (n_top_genes={n_top_genes})")
        # Binomial deviance (Townes et al. 2019): how much information each gene
        # carries beyond the depth-only null model μ_gc = n_c * p_g.
        X_f = X.astype(np.float32)
        cell_totals = X_f.sum(axis=1)  # (n_cells,)
        total_sum = float(cell_totals.sum())
        gene_fractions = X_f.sum(axis=0) / (total_sum + 1e-10)  # (n_genes,)
        deviance = np.zeros(n_genes, dtype=np.float64)

        batch_size = 256
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            X_b = X_f[:, start:end].astype(np.float64)
            p_b = gene_fractions[start:end].astype(np.float64)
            ct = cell_totals[:, np.newaxis].astype(np.float64)
            mu_b = ct * p_b[np.newaxis, :]
            n_minus_x = ct - X_b
            n_minus_mu = ct - mu_b
            with np.errstate(divide="ignore", invalid="ignore"):
                term1 = np.where(X_b > 0, X_b * np.log(np.where(X_b > 0, X_b / (mu_b + 1e-10), 1.0)), 0.0)
                term2 = np.where(
                    n_minus_x > 0,
                    n_minus_x * np.log(np.where(n_minus_x > 0, n_minus_x / (n_minus_mu + 1e-10), 1.0)),
                    0.0,
                )
            deviance[start:end] = 2.0 * (term1 + term2).sum(axis=0)

        hvg_df["deviance"] = deviance
        hvg_df["highly_variable"] = False
        top_indices = np.argsort(-deviance)[:n_top_genes]
        hvg_df.iloc[top_indices, hvg_df.columns.get_loc("highly_variable")] = True

    elif method == "spatial_moran":
        print(f"  Using spatial Moran's I method (n_top_genes={n_top_genes})")
        from scipy.sparse import csr_matrix
        from scipy.spatial import KDTree

        # Detect spatial coordinates from cell_meta (tries common column name conventions)
        coord_candidates = [
            ("x_centroid", "y_centroid"),
            ("CenterX_local_px", "CenterY_local_px"),
            ("x_local_px", "y_local_px"),
            ("x_global_px", "y_global_px"),
            ("x", "y"),
        ]
        coords = None
        for xcol, ycol in coord_candidates:
            if xcol in spatioloji_obj.cell_meta.columns and ycol in spatioloji_obj.cell_meta.columns:
                coords = spatioloji_obj.cell_meta[[xcol, ycol]].values.astype(np.float32)
                print(f"    Spatial coordinates: '{xcol}', '{ycol}'")
                break
        if coords is None:
            try:
                sp_data = spatioloji_obj.spatial
                coords = np.column_stack([sp_data.x, sp_data.y]).astype(np.float32)
                print("    Spatial coordinates: sp.spatial")
            except (AttributeError, TypeError) as err:
                raise ValueError(
                    "spatial_moran requires spatial coordinates. "
                    "Expected 'x_centroid'/'y_centroid' columns in cell_meta "
                    "or a valid sp.spatial attribute."
                ) from err

        # Build sparse row-normalised k-NN weight matrix
        tree = KDTree(coords)
        _, nn_indices = tree.query(coords, k=n_spatial_neighbors + 1)

        rows_idx, cols_idx = [], []
        for i in range(n_cells):
            for j in nn_indices[i, 1:]:  # skip self
                rows_idx.append(i)
                cols_idx.append(int(j))
        data_vals = np.ones(len(rows_idx), dtype=np.float32)
        W = csr_matrix((data_vals, (rows_idx, cols_idx)), shape=(n_cells, n_cells))
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W_norm = W.multiply(1.0 / row_sums[:, np.newaxis]).tocsr()

        # Vectorised Moran's I: I_g = (z_g @ W_norm @ z_g) / (z_g @ z_g)
        # With row-normalised W, S0 = n_cells so (n/S0) = 1.
        Z = X.astype(np.float32)
        gene_means_arr = Z.mean(axis=0)
        gene_stds_arr = Z.std(axis=0)
        gene_stds_arr[gene_stds_arr == 0] = 1.0
        Z = (Z - gene_means_arr[np.newaxis, :]) / gene_stds_arr[np.newaxis, :]
        WZ = W_norm @ Z  # (n_cells, n_genes)
        numerator = (Z * WZ).sum(axis=0)  # (n_genes,)
        denominator = (Z * Z).sum(axis=0)  # (n_genes,)
        moran_i = numerator / (denominator + 1e-12)

        hvg_df["moran_i"] = moran_i.astype(float)
        hvg_df["highly_variable"] = False
        top_indices = np.argsort(-moran_i)[:n_top_genes]
        hvg_df.iloc[top_indices, hvg_df.columns.get_loc("highly_variable")] = True

    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Choose from: seurat, cell_ranger, seurat_v3, pearson_residuals, deviance, spatial_moran"
        )

    n_hvg = int(hvg_df["highly_variable"].sum())
    print(f"  ✓ Selected {n_hvg} highly variable genes")
    if n_hvg > 0:
        hvg_mask = hvg_df["highly_variable"].values
        print(f"    Mean expression range: {mean[hvg_mask].min():.3f} - {mean[hvg_mask].max():.3f}")

    if inplace:
        for col in hvg_df.columns:
            spatioloji_obj._gene_meta[col] = hvg_df[col]
        return None
    else:
        return hvg_df


def select_genes_by_pattern(
    spatioloji_obj,
    patterns: str | list[str],
    method: Literal["startswith", "endswith", "contains", "regex"] = "startswith",
    case_sensitive: bool = False,
    output_column: str = "pattern_selected",
    inplace: bool = True,
):
    """
    Select genes based on name patterns.

    Useful for selecting/excluding specific gene families (e.g., mitochondrial,
    ribosomal genes).

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object
    patterns : str or list of str
        Pattern(s) to match gene names
    method : {'startswith', 'endswith', 'contains', 'regex'}, optional
        Matching method, by default 'startswith'
    case_sensitive : bool, optional
        Case-sensitive matching, by default False
    output_column : str, optional
        Column name for selection results, by default 'pattern_selected'
    inplace : bool, optional
        Add selection to gene_meta, by default True

    Returns
    -------
    np.ndarray or None
        Boolean mask if inplace=False, otherwise None

    Examples
    --------
    >>> # Select mitochondrial genes (often start with "MT-" or "mt-")
    >>> sp.processing.select_genes_by_pattern(
    ...     sp,
    ...     patterns=['MT-', 'mt-'],
    ...     method='startswith',
    ...     output_column='mito'
    ... )
    >>>
    >>> # Calculate % mitochondrial counts
    >>> mito_genes = sp.gene_meta[sp.gene_meta['mito']].index
    >>> mito_counts = sp.get_expression(gene_names=mito_genes).sum(axis=1)
    >>> total_counts = sp.get_expression().sum(axis=1)
    >>> sp.cell_meta['pct_mito'] = (mito_counts / total_counts) * 100
    >>>
    >>> # Select ribosomal genes
    >>> sp.processing.select_genes_by_pattern(
    ...     sp,
    ...     patterns=['RPS', 'RPL'],
    ...     method='startswith',
    ...     output_column='ribo'
    ... )
    """
    print(f"\nSelecting genes by pattern (method={method})")

    if isinstance(patterns, str):
        patterns = [patterns]

    gene_names = spatioloji_obj.gene_index.astype(str)

    if not case_sensitive:
        gene_names = gene_names.str.lower()
        patterns = [p.lower() for p in patterns]

    # Initialize mask
    mask = np.zeros(len(gene_names), dtype=bool)

    for pattern in patterns:
        if method == "startswith":
            mask |= gene_names.str.startswith(pattern).values
        elif method == "endswith":
            mask |= gene_names.str.endswith(pattern).values
        elif method == "contains":
            mask |= gene_names.str.contains(pattern, regex=False).values
        elif method == "regex":
            mask |= gene_names.str.contains(pattern, regex=True).values
        else:
            raise ValueError(f"Unknown method: {method}")

    n_selected = mask.sum()
    print(f"  ✓ Selected {n_selected} genes matching patterns: {patterns}")

    if inplace:
        spatioloji_obj._gene_meta[output_column] = mask
        return None
    else:
        return mask


def compare_hvg_methods(
    spatioloji_obj,
    methods: list[str] | None = None,
    n_top_genes: int = 2000,
    layers: list[str | None] | None = None,
    n_spatial_neighbors: int = 15,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compare HVG gene lists across multiple selection methods.

    Each method is run with its matched layer from *layers*, then pairwise
    Jaccard similarity between the resulting gene lists is reported and a
    consensus set is identified.  Methods can be run in parallel via
    *n_jobs*.

    **Default layer assignment (when** ``layers=None`` **):**

    - ``None`` (main expression matrix / raw counts):
      ``'deviance'``, ``'pearson_residuals'``, ``'cell_ranger'``
    - ``'log_normalized'``:
      ``'seurat'``, ``'seurat_v3'``, ``'spatial_moran'``

    Args:
        spatioloji_obj: Spatioloji object with expression data.
        methods: List of HVG methods to compare. Defaults to
            ``['seurat', 'seurat_v3', 'pearson_residuals', 'deviance']``.
            Add ``'spatial_moran'`` to include spatially variable genes
            (requires spatial coordinates).
        n_top_genes: Number of top HVGs per method, by default 2000.
        layers: List of layer names, one per method (same order as *methods*).
            Each element is a layer name string or ``None`` (main expression).
            When ``None`` (default), layers are assigned automatically based on
            each method's input requirements (raw counts vs log-normalised).
        n_spatial_neighbors: k for spatial weight matrix if ``'spatial_moran'``
            is included, by default 15.
        n_jobs: Number of parallel worker threads, by default 1 (serial).
            ``-1`` uses all available CPU cores.  Methods are independent and
            safe to run concurrently.

    Returns:
        pd.DataFrame indexed by gene name with:
            - One boolean column per method (True = selected as HVG).
            - ``n_methods_selected``: How many methods selected each gene.
            - ``consensus`` (bool): True if selected by the majority of methods.

    Raises:
        ValueError: If *layers* is provided but its length does not match
            *methods*.

    Example:
        >>> # Auto layer assignment, serial
        >>> df = compare_hvg_methods(sp, methods=['seurat', 'pearson_residuals'])
        >>>
        >>> # Explicit layers matched to methods, parallel
        >>> df = compare_hvg_methods(
        ...     sp,
        ...     methods=['seurat', 'seurat_v3', 'pearson_residuals', 'deviance'],
        ...     layers=['log_normalized', 'log_normalized', None, None],
        ...     n_jobs=4,
        ... )
        >>> consensus_genes = df.index[df['consensus']].tolist()
    """
    if methods is None:
        methods = ["seurat", "seurat_v3", "pearson_residuals", "deviance"]

    # ── Resolve layers list ──────────────────────────────────────────────────
    if layers is None:
        layers = [None if m in _RAW_COUNT_METHODS else "log_normalized" for m in methods]
    elif len(layers) != len(methods):
        raise ValueError(
            f"len(layers)={len(layers)} does not match len(methods)={len(methods)}. "
            "Provide one layer entry per method, in the same order."
        )

    # ── Worker function ──────────────────────────────────────────────────────
    def _run(method: str, layer: str | None) -> tuple[str, np.ndarray | None]:
        try:
            hvg_df = highly_variable_genes(
                spatioloji_obj,
                layer=layer,
                n_top_genes=n_top_genes,
                method=method,
                n_spatial_neighbors=n_spatial_neighbors,
                inplace=False,
            )
            return method, hvg_df["highly_variable"].values.astype(bool)
        except Exception as exc:
            warnings.warn(f"Method '{method}' failed and will be skipped: {exc}", stacklevel=2)
            return method, None

    # ── Dispatch (serial or parallel) ────────────────────────────────────────
    gene_index = spatioloji_obj.gene_index
    n_workers = max(1, ((os.cpu_count() or 1) + 1 + n_jobs) if n_jobs < 0 else n_jobs)

    print(f"\nComparing {len(methods)} HVG methods (n_jobs={n_workers})")
    for m, l in zip(methods, layers):
        print(f"  {m:<28} layer='{l}'")

    ordered_results: list[tuple[str, np.ndarray | None]] = [None] * len(methods)  # type: ignore[list-item]

    if n_workers == 1:
        for i, (method, layer) in enumerate(zip(methods, layers)):
            print(f"\n{'=' * 55}\nRunning: {method}  (layer='{layer}')")
            ordered_results[i] = _run(method, layer)
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for i, (method, layer) in enumerate(zip(methods, layers)):
                fut = ex.submit(_run, method, layer)
                futures[fut] = i
            for fut in as_completed(futures):
                i = futures[fut]
                ordered_results[i] = fut.result()

    results: dict[str, np.ndarray] = {method: arr for method, arr in ordered_results if arr is not None}

    if not results:
        raise RuntimeError("All methods failed. Cannot produce a comparison.")

    method_list = list(results.keys())
    df = pd.DataFrame(results, index=gene_index, dtype=bool)
    df["n_methods_selected"] = df[method_list].sum(axis=1)
    majority = len(method_list) // 2 + 1
    df["consensus"] = df["n_methods_selected"] >= majority

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("HVG Method Comparison Summary")
    print(f"{'=' * 60}")
    print(f"  n_top_genes per method : {n_top_genes}")
    print(f"  Methods run            : {method_list}\n")

    print("  Genes selected per method:")
    for m in method_list:
        print(f"    {m:<28}  {int(df[m].sum()):>5} genes")

    n_consensus = int(df["consensus"].sum())
    n_all = int((df["n_methods_selected"] == len(method_list)).sum())
    n_unique = int((df["n_methods_selected"] == 1).sum())
    print(f"\n  Consensus (≥ {majority}/{len(method_list)} methods): {n_consensus} genes")
    print(f"  Selected by ALL methods: {n_all} genes")
    print(f"  Selected by only 1 method: {n_unique} genes")

    # ── Pairwise Jaccard similarity ──────────────────────────────────────────
    print("\n  Pairwise Jaccard similarity:")
    col_w = max(len(m) for m in method_list) + 2
    header = " " * col_w + "  ".join(f"{m[:10]:>10}" for m in method_list)
    print(f"    {header}")
    for m_i in method_list:
        row_vals = []
        for m_j in method_list:
            a = df[m_i].values
            b = df[m_j].values
            inter = int((a & b).sum())
            union = int((a | b).sum())
            row_vals.append(f"{inter / union if union else 1.0:.3f}")
        print(f"    {m_i:<{col_w}}{'  '.join(f'{v:>10}' for v in row_vals)}")

    # ── Per-gene overlap count distribution ─────────────────────────────────
    print("\n  Genes selected by exactly N methods:")
    counts = df["n_methods_selected"].value_counts().sort_index()
    for n_sel, cnt in counts.items():
        bar = "█" * min(int(cnt / max(counts) * 30), 30)
        print(f"    N={n_sel}: {cnt:>5} genes  {bar}")

    return df
