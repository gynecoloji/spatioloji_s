"""
feature_selection.py - Feature selection methods for spatial transcriptomics

Provides methods for selecting highly variable genes before dimensionality
reduction and clustering.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def highly_variable_genes(
    spatioloji_obj,
    layer: str | None = None,
    n_top_genes: int = 2000,
    method: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',
    flavor: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',  # scanpy compatibility
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    n_bins: int = 20,
    output_column: str = 'highly_variable',
    inplace: bool = True
):
    """
    Identify highly variable genes (HVGs).

    Selects genes that show high variability across cells, which typically
    contain the most biological signal. This should be done BEFORE PCA and
    clustering.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to use. If None, uses main expression matrix
        Use 'normalized_counts' for Seurat method (before log transform)
    n_top_genes : int, optional
        Number of highly variable genes to select, by default 2000
    method : {'seurat', 'cell_ranger', 'seurat_v3'}, optional
        Method for identifying HVGs:
        - 'seurat': Mean-variance relationship (Seurat v1/v2)
        - 'cell_ranger': Similar to Seurat
        - 'seurat_v3': Variance stabilizing transformation (Seurat v3)
        By default 'seurat'
    flavor : str, optional
        Alias for method (for scanpy compatibility)
    min_mean : float, optional
        Minimum mean expression for Seurat method, by default 0.0125
    max_mean : float, optional
        Maximum mean expression for Seurat method, by default 3.0
    min_disp : float, optional
        Minimum dispersion for Seurat method, by default 0.5
    n_bins : int, optional
        Number of bins for Seurat method, by default 20
    output_column : str, optional
        Column name in gene_meta for HVG labels, by default 'highly_variable'
    inplace : bool, optional
        Add HVG labels to gene_meta, by default True

    Returns
    -------
    pd.DataFrame or None
        DataFrame with HVG statistics if inplace=False, otherwise None

    Examples
    --------
    >>> # Standard Seurat method (most common)
    >>> sp.processing.highly_variable_genes(
    ...     sp,
    ...     layer='normalized_counts',  # Before log transform
    ...     n_top_genes=2000,
    ...     method='seurat'
    ... )
    >>>
    >>> # Seurat v3 method (for large datasets)
    >>> sp.processing.highly_variable_genes(
    ...     sp,
    ...     layer='normalized_counts',
    ...     method='seurat_v3',
    ...     n_top_genes=3000
    ... )
    >>>
    >>> # Then subset to HVGs before clustering
    >>> hvg_genes = sp.gene_meta[sp.gene_meta['highly_variable']].index
    >>> sp_hvg = sp.subset_by_genes(hvg_genes)

    References
    ----------
    Seurat: Stuart et al. (2019) Cell
    Seurat v3: Hafemeister & Satija (2019) Genome Biology
    """
    # Handle flavor alias
    if flavor != 'seurat':
        method = flavor

    print(f"\nIdentifying highly variable genes (method={method})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    n_cells, n_genes = X.shape

    # Calculate mean and variance per gene
    mean = X.mean(axis=0)
    var = X.var(axis=0)

    # Initialize results dataframe
    hvg_df = pd.DataFrame({
        'means': mean,
        'variances': var
    }, index=spatioloji_obj.gene_index)

    if method == 'seurat' or method == 'cell_ranger':
        print(f"  Using Seurat method (n_top_genes={n_top_genes})")

        # Calculate dispersion (normalized variance)
        mean_nonzero = mean.copy()
        mean_nonzero[mean_nonzero == 0] = 1e-12  # Avoid division by zero
        dispersion = var / mean_nonzero

        # Log transform
        hvg_df['dispersions'] = dispersion
        hvg_df['mean_bin'] = pd.cut(mean, bins=n_bins, labels=False)

        # Calculate normalized dispersion per bin
        disp_norm = np.zeros(n_genes)

        for bin_idx in range(n_bins):
            bin_mask = hvg_df['mean_bin'] == bin_idx
            if bin_mask.sum() > 1:
                bin_disp = dispersion[bin_mask]
                disp_median = np.median(bin_disp)
                # MAD (median absolute deviation)
                disp_mad = np.median(np.abs(bin_disp - disp_median))
                disp_norm[bin_mask] = (bin_disp - disp_median) / (disp_mad + 1e-12)

        hvg_df['dispersions_norm'] = disp_norm

        # Filter by mean and dispersion
        gene_subset = (
            (mean > min_mean) &
            (mean < max_mean) &
            (disp_norm > min_disp)
        )

        hvg_df['highly_variable'] = False

        if gene_subset.sum() > n_top_genes:
            # Rank by normalized dispersion
            ranks = rankdata(-disp_norm[gene_subset])
            top_genes_mask = ranks <= n_top_genes

            # Map back to full gene list
            subset_indices = np.where(gene_subset)[0]
            hvg_indices = subset_indices[top_genes_mask]
            hvg_df.iloc[hvg_indices, hvg_df.columns.get_loc('highly_variable')] = True
        else:
            hvg_df.loc[gene_subset, 'highly_variable'] = True

    elif method == 'seurat_v3':
        print(f"  Using Seurat v3 method (n_top_genes={n_top_genes})")

        # Variance stabilizing transformation
        # Clip to avoid log(0)
        mean_clip = np.clip(mean, 1e-12, None)
        var_clip = np.clip(var, 1e-12, None)

        # Expected variance under Poisson
        var_expected = mean_clip

        # Variance ratio
        var_ratio = var_clip / var_expected

        # Standardized variance
        hvg_df['variances_norm'] = np.log10(var_ratio)

        # Select top genes
        hvg_df['highly_variable'] = False
        top_indices = np.argsort(-hvg_df['variances_norm'].values)[:n_top_genes]
        hvg_df.iloc[top_indices, hvg_df.columns.get_loc('highly_variable')] = True

    else:
        raise ValueError(f"Unknown method: {method}")

    n_hvg = hvg_df['highly_variable'].sum()
    print(f"  ✓ Selected {n_hvg} highly variable genes")
    print(f"    Mean expression range: {mean[hvg_df['highly_variable']].min():.2f} - "
          f"{mean[hvg_df['highly_variable']].max():.2f}")

    if inplace:
        # Add to gene metadata
        for col in hvg_df.columns:
            spatioloji_obj._gene_meta[col] = hvg_df[col]
        return None
    else:
        return hvg_df


def select_genes_by_pattern(
    spatioloji_obj,
    patterns: str | list[str],
    method: Literal['startswith', 'endswith', 'contains', 'regex'] = 'startswith',
    case_sensitive: bool = False,
    output_column: str = 'pattern_selected',
    inplace: bool = True
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
        if method == 'startswith':
            mask |= gene_names.str.startswith(pattern).values
        elif method == 'endswith':
            mask |= gene_names.str.endswith(pattern).values
        elif method == 'contains':
            mask |= gene_names.str.contains(pattern, regex=False).values
        elif method == 'regex':
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
