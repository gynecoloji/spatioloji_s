"""
feature_selection.py - Feature selection methods for spatial transcriptomics

Provides methods for selecting highly variable genes before dimensionality
reduction and clustering.

GPU acceleration
----------------
``highly_variable_genes`` accepts ``device='auto' | 'cpu' | 'gpu'``.  When
``'auto'`` and RAPIDS is importable, the heavy per-gene mean / var /
Pearson-residual / deviance reductions (the cell-axis sums that dominate
runtime on Xenium-scale data) run on cupy.  Small per-bin / polynomial
fits stay on CPU.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from ._gpu import Device, _resolve_device, _to_cupy, _to_numpy, _warn_fallback

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


def _hvg_mean_var_gpu(X, n_cells, n_genes):
    """Compute (mean, var, gene_sums, cell_totals) on GPU. Returns numpy.

    Uses ``_to_cupy`` so integer raw-counts inputs (the typical Xenium /
    CosMx case) are up-promoted to ``float32`` before crossing onto the
    device — ``cupyx.scipy.sparse`` only supports float / bool / complex
    dtypes.
    """
    import cupy as cp

    X_gpu = _to_cupy(X)

    if sparse.issparse(X):
        gene_sums = cp.asarray(X_gpu.sum(axis=0)).ravel().astype(cp.float64)
        gene_sumsq = cp.asarray(X_gpu.power(2).sum(axis=0)).ravel().astype(cp.float64)
        cell_totals = cp.asarray(X_gpu.sum(axis=1)).ravel().astype(cp.float64)
        mean = gene_sums / n_cells
        var = gene_sumsq / n_cells - mean**2
    else:
        mean = X_gpu.mean(axis=0)
        var = X_gpu.var(axis=0)
        gene_sums = X_gpu.sum(axis=0).astype(cp.float64)
        cell_totals = X_gpu.sum(axis=1).astype(cp.float64)

    return _to_numpy(mean), _to_numpy(var), _to_numpy(gene_sums), _to_numpy(cell_totals)


def _hvg_pearson_residual_var_gpu(X, cell_totals, gene_fractions, clip_val, batch_size: int = 512):
    """Compute per-gene Pearson residual variance on GPU.

    Gene-chunked to bound device memory to ``n_cells × batch_size × float32``
    per iteration (≈ 1 GB at 500k cells × 512 genes), avoiding the
    multi-tens-of-GB temporary that a one-shot densification produces on
    Xenium-scale data.  Returns numpy.
    """
    import cupy as cp

    n_cells, n_genes = X.shape
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X = X.tocsr()

    ct = cp.asarray(cell_totals, dtype=cp.float32)[:, None]
    gf_full = cp.asarray(gene_fractions, dtype=cp.float32)
    out = cp.empty(n_genes, dtype=cp.float32)

    for s in range(0, n_genes, batch_size):
        e = min(s + batch_size, n_genes)
        if is_sparse:
            X_b_cpu = X[:, s:e].toarray()
        else:
            X_b_cpu = X[:, s:e]
        X_b = cp.asarray(X_b_cpu, dtype=cp.float32)
        gf_b = gf_full[s:e][None, :]
        mu_b = ct * gf_b
        r_b = (X_b - mu_b) / cp.sqrt(mu_b + cp.float32(1e-10))
        cp.clip(r_b, -clip_val, clip_val, out=r_b)
        out[s:e] = r_b.var(axis=0)
        # let cupy's caching allocator reclaim the slice temporaries
        del X_b, mu_b, r_b

    return _to_numpy(out)


def _hvg_deviance_gpu(X, cell_totals, gene_fractions, batch_size: int = 256):
    """Compute per-gene binomial deviance on GPU.

    Gene-chunked, like ``_hvg_pearson_residual_var_gpu``, to keep device
    memory bounded.  Uses ``float64`` per the CPU implementation to avoid
    overflow in ``cell_totals - X`` for cells with very high transcript
    counts.  Returns numpy.
    """
    import cupy as cp

    n_cells, n_genes = X.shape
    is_sparse = sparse.issparse(X)
    if is_sparse:
        X = X.tocsr()

    ct = cp.asarray(cell_totals, dtype=cp.float64)[:, None]
    gf_full = cp.asarray(gene_fractions, dtype=cp.float64)
    out = cp.empty(n_genes, dtype=cp.float64)
    eps = cp.float64(1e-10)
    one = cp.float64(1.0)

    for s in range(0, n_genes, batch_size):
        e = min(s + batch_size, n_genes)
        if is_sparse:
            X_b_cpu = X[:, s:e].toarray()
        else:
            X_b_cpu = X[:, s:e]
        X_b = cp.asarray(X_b_cpu, dtype=cp.float64)
        p_b = gf_full[s:e][None, :]
        mu_b = ct * p_b
        n_minus_x = ct - X_b
        n_minus_mu = ct - mu_b

        safe_x = cp.where(X_b > 0, X_b, one)
        term1 = cp.where(X_b > 0, X_b * cp.log(safe_x / (mu_b + eps)), 0.0)
        safe_nx = cp.where(n_minus_x > 0, n_minus_x, one)
        term2 = cp.where(n_minus_x > 0, n_minus_x * cp.log(safe_nx / (n_minus_mu + eps)), 0.0)
        out[s:e] = 2.0 * (term1 + term2).sum(axis=0)

        del X_b, mu_b, n_minus_x, n_minus_mu, safe_x, safe_nx, term1, term2

    return _to_numpy(out)


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
    device: Device = "auto",
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
        device: Compute backend (``'auto' | 'cpu' | 'gpu'``).  ``'auto'``
            (default) routes the heavy per-gene mean/var reduction and the
            ``pearson_residuals`` / ``deviance`` matrix math through cupy
            when RAPIDS is importable; small per-bin / polynomial fits and
            ``spatial_moran``'s KDTree stay on CPU.

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

    # Keep sparse on the GPU path; only densify on the CPU path or when sparse
    # input is unavoidable (CPU path has always done so).
    if layer is None:
        X_raw = (
            spatioloji_obj.expression.get_sparse()
            if spatioloji_obj.expression.is_sparse
            else spatioloji_obj.expression.get_dense()
        )
    else:
        X_raw = spatioloji_obj.get_layer(layer)

    n_cells, n_genes = X_raw.shape

    backend = _resolve_device(device, "highly_variable_genes")

    cell_totals = None  # only computed if needed (pearson_residuals / deviance)

    if backend == "gpu":
        try:
            mean, var, gene_sums, cell_totals = _hvg_mean_var_gpu(X_raw, n_cells, n_genes)
            print("  (mean/var reductions on GPU)")
            X = X_raw  # keep original, may still be sparse — needed for GPU helpers below
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("highly_variable_genes", exc)
            backend = "cpu"

    if backend == "cpu":
        if sparse.issparse(X_raw):
            X = X_raw.toarray()
        else:
            X = X_raw
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
        clip_val = float(np.sqrt(n_cells))

        if backend == "gpu":
            if cell_totals is None:
                cell_totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64) if sparse.issparse(X) \
                    else X.sum(axis=1).astype(np.float64)
            total_sum = float(cell_totals.sum())
            gene_fractions = (
                np.asarray(X.sum(axis=0)).ravel().astype(np.float64) if sparse.issparse(X)
                else X.sum(axis=0).astype(np.float64)
            ) / (total_sum + 1e-10)
            try:
                resid_var = _hvg_pearson_residual_var_gpu(X, cell_totals, gene_fractions, clip_val)
                resid_var = resid_var.astype(np.float32)
            except Exception as exc:
                if device == "gpu":
                    raise
                _warn_fallback("highly_variable_genes (pearson)", exc)
                backend = "cpu"

        if backend == "cpu":
            # Analytic Pearson residuals under Poisson null: μ_gc = n_c * p_g
            # Processed in gene batches to limit peak memory usage.
            X_f = X.astype(np.float32) if not sparse.issparse(X) else X.toarray().astype(np.float32)
            cell_totals = X_f.sum(axis=1)
            total_sum = float(cell_totals.sum())
            gene_fractions = X_f.sum(axis=0) / (total_sum + 1e-10)
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

        if backend == "gpu":
            if cell_totals is None:
                cell_totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float64) if sparse.issparse(X) \
                    else X.sum(axis=1).astype(np.float64)
            total_sum = float(cell_totals.sum())
            gene_fractions = (
                np.asarray(X.sum(axis=0)).ravel().astype(np.float64) if sparse.issparse(X)
                else X.sum(axis=0).astype(np.float64)
            ) / (total_sum + 1e-10)
            try:
                deviance = _hvg_deviance_gpu(X, cell_totals, gene_fractions)
            except Exception as exc:
                if device == "gpu":
                    raise
                _warn_fallback("highly_variable_genes (deviance)", exc)
                backend = "cpu"

        if backend == "cpu":
            # Binomial deviance (Townes et al. 2019): how much information each gene
            # carries beyond the depth-only null model μ_gc = n_c * p_g.
            X_f = X.astype(np.float32) if not sparse.issparse(X) else X.toarray().astype(np.float32)
            cell_totals = X_f.sum(axis=1)
            total_sum = float(cell_totals.sum())
            gene_fractions = X_f.sum(axis=0) / (total_sum + 1e-10)
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
        # spatial_moran has no GPU helper; densify here if we took the GPU
        # path on mean/var (X may still be sparse).
        if sparse.issparse(X):
            X = X.toarray()
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
    device: Device = "auto",
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
                device=device,
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
