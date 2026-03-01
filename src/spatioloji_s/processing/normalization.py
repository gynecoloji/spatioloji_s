"""
normalization.py - Normalization methods for spatial transcriptomics data

Provides various normalization strategies optimized for spatioloji objects.
"""

from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.preprocessing import RobustScaler, StandardScaler


def normalize_total(
    spatioloji_obj,
    target_sum: float = 1e4,
    layer: str | None = None,
    output_layer: str = "normalized_counts",
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    inplace: bool = False,
    copy: bool = False,
):
    """
    Normalize counts per cell (library size normalization).

    Each cell is normalized to have the same total count (library size).
    This is typically followed by log1p transformation.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    target_sum : float, optional
        Target sum for each cell, by default 1e4
    layer : str, optional
        Which layer to normalize. If None, uses main expression matrix
    output_layer : str, optional
        Name for output layer, by default 'normalized_counts'
    exclude_highly_expressed : bool, optional
        Exclude highly expressed genes from normalization, by default False
    max_fraction : float, optional
        If exclude_highly_expressed=True, exclude genes that make up more
        than max_fraction of total counts, by default 0.05
    inplace : bool, optional
        Modify the spatioloji object in place (adds layer), by default False
    copy : bool, optional
        Return a copy of spatioloji object, by default False

    Returns
    -------
    spatioloji or None
        If copy=True, returns new spatioloji object with normalized layer
        If inplace=True, returns None (modifies object in place)
        Otherwise returns normalized expression matrix

    Examples
    --------
    >>> # Normalize and add as layer
    >>> sp.processing.normalize_total(sp, target_sum=1e4, inplace=True)
    >>>
    >>> # Get normalized matrix without modifying object
    >>> norm_expr = sp.processing.normalize_total(sp, target_sum=1e4)
    >>>
    >>> # Create new object with normalized data
    >>> sp_norm = sp.processing.normalize_total(sp, target_sum=1e4, copy=True)
    """
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
        is_sparse = spatioloji_obj.expression.is_sparse
    else:
        X = spatioloji_obj.get_layer(layer)
        is_sparse = sparse.issparse(X)
        if is_sparse:
            X = X.toarray()

    print(f"\nNormalizing expression to target sum: {target_sum:,.0f}")

    # Calculate counts per cell
    if is_sparse and sparse.issparse(X):
        counts_per_cell = np.array(X.sum(axis=1)).flatten()
    else:
        counts_per_cell = X.sum(axis=1)

    print(f"  Counts per cell - mean: {counts_per_cell.mean():.0f}, " f"median: {np.median(counts_per_cell):.0f}")

    # Exclude highly expressed genes if requested
    if exclude_highly_expressed:
        gene_subset = np.ones(X.shape[1], dtype=bool)

        # Calculate fraction of counts per gene across all cells
        counts_per_gene = X.sum(axis=0)
        if sparse.issparse(counts_per_gene):
            counts_per_gene = np.array(counts_per_gene).flatten()

        gene_fractions = counts_per_gene / counts_per_gene.sum()
        gene_subset = gene_fractions < max_fraction

        n_excluded = (~gene_subset).sum()
        if n_excluded > 0:
            print(f"  Excluding {n_excluded} highly expressed genes (>{max_fraction*100:.1f}% of total)")

            # Recalculate counts without highly expressed genes
            counts_per_cell = X[:, gene_subset].sum(axis=1)
            if sparse.issparse(counts_per_cell):
                counts_per_cell = np.array(counts_per_cell).flatten()

    # Normalize
    counts_per_cell_nonzero = counts_per_cell.copy()
    counts_per_cell_nonzero[counts_per_cell_nonzero == 0] = 1  # Avoid division by zero

    scaling_factors = target_sum / counts_per_cell_nonzero

    # Apply normalization
    if is_sparse and sparse.issparse(X):
        # For sparse matrices, multiply by scaling factors
        X_norm = X.multiply(scaling_factors[:, np.newaxis])
    else:
        X_norm = X * scaling_factors[:, np.newaxis]

    print(f"  ✓ Normalized {X.shape[0]:,} cells × {X.shape[1]:,} genes")

    # Return based on mode
    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_norm, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_norm, overwrite=True)
        return None
    else:
        return X_norm


def log_transform(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "log_normalized",
    base: float | None = None,
    offset: float = 1.0,
    inplace: bool = False,
    copy: bool = False,
):
    """
    Apply log transformation to expression data.

    Typically applied after library size normalization.
    Uses log1p (log(x + 1)) by default to handle zeros.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to transform. If None, uses main expression matrix
    output_layer : str, optional
        Name for output layer, by default 'log_normalized'
    base : float, optional
        Logarithm base. If None, uses natural log (np.log1p)
    offset : float, optional
        Offset added before log, by default 1.0 (for log1p)
    inplace : bool, optional
        Modify the spatioloji object in place, by default False
    copy : bool, optional
        Return a copy of spatioloji object, by default False

    Returns
    -------
    spatioloji, np.ndarray, or None
        Depends on copy/inplace parameters

    Examples
    --------
    >>> # Log transform normalized counts
    >>> sp.processing.normalize_total(sp, inplace=True)
    >>> sp.processing.log_transform(sp, layer='normalized_counts', inplace=True)
    >>>
    >>> # Combine normalization and log in one step
    >>> norm = sp.processing.normalize_total(sp)
    >>> log_norm = sp.processing.log_transform(sp, layer='normalized_counts')
    """
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    print(f"\nApplying log transformation (base={'e' if base is None else base})")

    # Apply transformation
    if base is None:
        # Use natural log (log1p)
        X_log = np.log1p(X) if offset == 1.0 else np.log(X + offset)
    else:
        X_log = np.log(X + offset) / np.log(base)

    print(f"  ✓ Transformed {X.shape[0]:,} cells × {X.shape[1]:,} genes")

    # Return based on mode
    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_log, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_log, overwrite=True)
        return None
    else:
        return X_log


def scale(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "scaled",
    max_value: float | None = 10.0,
    method: Literal["standard", "robust"] = "standard",
    zero_center: bool = True,
    inplace: bool = False,
    copy: bool = False,
):
    """
    Scale expression data (z-score normalization per gene).

    Centers and scales each gene to unit variance. Useful for PCA and
    other dimensional reduction techniques.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to scale. If None, uses main expression matrix
    output_layer : str, optional
        Name for output layer, by default 'scaled'
    max_value : float, optional
        Clip values to [-max_value, max_value], by default 10.0
        Set to None to disable clipping
    method : {'standard', 'robust'}, optional
        Scaling method:
        - 'standard': (X - mean) / std
        - 'robust': (X - median) / IQR (robust to outliers)
    zero_center : bool, optional
        Center data to zero mean, by default True
    inplace : bool, optional
        Modify the spatioloji object in place, by default False
    copy : bool, optional
        Return a copy of spatioloji object, by default False

    Returns
    -------
    spatioloji, np.ndarray, or None
        Depends on copy/inplace parameters

    Examples
    --------
    >>> # Standard scaling
    >>> sp.processing.scale(sp, layer='log_normalized', inplace=True)
    >>>
    >>> # Robust scaling (better for data with outliers)
    >>> sp.processing.scale(sp, method='robust', inplace=True)
    """
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    print(f"\nScaling expression data (method: {method})")

    # Apply scaling
    if method == "standard":
        scaler = StandardScaler(with_mean=zero_center, with_std=True)
    elif method == "robust":
        scaler = RobustScaler(with_centering=zero_center, with_scaling=True)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_scaled = scaler.fit_transform(X)

    # Clip values if requested
    if max_value is not None:
        X_scaled = np.clip(X_scaled, -max_value, max_value)
        print(f"  Clipped values to [{-max_value}, {max_value}]")

    print(f"  ✓ Scaled {X.shape[0]:,} cells × {X.shape[1]:,} genes")

    # Return based on mode
    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, X_scaled, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
        return None
    else:
        return X_scaled


def normalize_pearson_residuals(
    spatioloji_obj,
    layer: str | None = None,
    output_layer: str = "pearson_residuals",
    theta: float = 100.0,
    clip: float | None = None,
    inplace: bool = False,
    copy: bool = False,
):
    """
    Normalize using Pearson residuals (analytic method).

    Normalizes counts using a negative binomial model without fitting,
    following the analytical approach from Lause et al. (2021) Genome Biology.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with raw counts
    layer : str, optional
        Which layer to use. If None, uses main expression matrix
    output_layer : str, optional
        Name for output layer, by default 'pearson_residuals'
    theta : float, optional
        Overdispersion parameter for negative binomial, by default 100.0
    clip : float, optional
        Clip residuals to [-clip, clip], by default None (uses sqrt(n))
    inplace : bool, optional
        Modify the spatioloji object in place, by default False
    copy : bool, optional
        Return a copy of spatioloji object, by default False

    Returns
    -------
    spatioloji, np.ndarray, or None
        Depends on copy/inplace parameters

    References
    ----------
    Lause, J., Berens, P. & Kobak, D. Analytic Pearson residuals for
    normalization of single-cell RNA-seq UMI data. Genome Biol 22, 258 (2021).

    Examples
    --------
    >>> # Normalize raw counts using Pearson residuals
    >>> sp.processing.normalize_pearson_residuals(sp, inplace=True)
    """
    # Get raw counts
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    print(f"\nNormalizing using Pearson residuals (theta={theta})")

    # Calculate library sizes
    lib_size = X.sum(axis=1, keepdims=True)

    # Calculate gene means
    gene_means = X.mean(axis=0, keepdims=True)

    # Calculate expected counts under NB model
    mu = lib_size * gene_means / lib_size.mean()

    # Calculate Pearson residuals
    residuals = (X - mu) / np.sqrt(mu + mu**2 / theta)

    # Clip if requested
    if clip is None:
        clip = np.sqrt(X.shape[0])

    residuals = np.clip(residuals, -clip, clip)

    print(f"  Clipped residuals to [{-clip:.2f}, {clip:.2f}]")
    print(f"  ✓ Calculated residuals for {X.shape[0]:,} cells × {X.shape[1]:,} genes")

    # Return based on mode
    if copy:
        import copy as copy_module

        sp_new = copy_module.deepcopy(spatioloji_obj)
        sp_new.add_layer(output_layer, residuals, overwrite=True)
        return sp_new
    elif inplace:
        spatioloji_obj.add_layer(output_layer, residuals, overwrite=True)
        return None
    else:
        return residuals


# Convenience function for standard workflow
def normalize_standard_workflow(
    spatioloji_obj,
    target_sum: float = 1e4,
    log_transform: bool = True,
    scale: bool = False,
    output_layer: str = "normalized",
    inplace: bool = True,
):
    """
    Apply standard normalization workflow.

    Performs: library size normalization → log transformation → (optional) scaling

    This is the most common normalization workflow for scRNA-seq data.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with raw counts
    target_sum : float, optional
        Target sum for library size normalization, by default 1e4
    log_transform : bool, optional
        Apply log1p transformation, by default True
    scale : bool, optional
        Apply z-score scaling, by default False
    output_layer : str, optional
        Name for final output layer, by default 'normalized'
    inplace : bool, optional
        Modify the spatioloji object in place, by default True

    Returns
    -------
    None or np.ndarray
        If inplace=True, returns None. Otherwise returns normalized matrix.

    Examples
    --------
    >>> # Standard workflow (library size + log)
    >>> sp.processing.normalize_standard_workflow(sp)
    >>>
    >>> # With scaling (for PCA/clustering)
    >>> sp.processing.normalize_standard_workflow(sp, scale=True)
    """
    print("\n" + "=" * 70)
    print("Standard Normalization Workflow")
    print("=" * 70)

    # Step 1: Library size normalization
    print("\nStep 1: Library size normalization")
    X_norm = normalize_total(spatioloji_obj, target_sum=target_sum, output_layer="temp_counts")

    # Step 2: Log transformation
    if log_transform:
        print("\nStep 2: Log transformation")
        spatioloji_obj.add_layer("temp_counts", X_norm, overwrite=True)
        X_norm = log_transform(spatioloji_obj, layer="temp_counts", output_layer="temp_log")
        spatioloji_obj.remove_layer("temp_counts")

    # Step 3: Scaling (optional)
    if scale:
        print("\nStep 3: Scaling")
        spatioloji_obj.add_layer("temp_log", X_norm, overwrite=True)
        X_norm = scale(spatioloji_obj, layer="temp_log", output_layer="temp_scaled")
        spatioloji_obj.remove_layer("temp_log")

    print("\n" + "=" * 70)
    print("✓ Normalization complete")
    print("=" * 70)

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_norm, overwrite=True)
        return None
    else:
        return X_norm
