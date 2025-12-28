"""
basic_plots.py - Basic visualization functions for spatioloji objects

Provides standard single-cell analysis plots with extensive color customization options.
"""

from typing import Optional, Union, List, Tuple, Literal, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
import seaborn as sns
import warnings
from scipy import sparse


def plot_umap_by_clusters(
    spatioloji_obj,
    cluster_column: str = 'leiden',
    color_map: Optional[Union[str, List, Dict]] = None,
    palette: Optional[Union[str, List, Dict]] = None,  # Alias for color_map
    colors: Optional[Dict[str, str]] = None,  # Map specific clusters to colors
    point_size: float = 5.0,
    alpha: float = 0.7,
    legend: bool = True,
    legend_loc: str = 'right margin',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Plot UMAP colored by cluster labels with extensive color customization.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with UMAP coordinates
    cluster_column : str, optional
        Column name in cell_meta with cluster labels, by default 'leiden'
    color_map : str, list, or dict, optional
        Color specification:
        - str: Matplotlib colormap name (e.g., 'tab20', 'Set3')
        - list: List of colors (hex, named, or RGB tuples)
        - dict: Mapping of cluster labels to colors
        If None, uses 'tab20' for ≤20 clusters, 'tab20b' + 'tab20c' for more
    palette : str, list, or dict, optional
        Alias for color_map (for seaborn compatibility)
    colors : dict, optional
        Explicit mapping of cluster labels to colors (overrides color_map)
        Example: {'0': '#FF0000', '1': '#00FF00', '2': '#0000FF'}
    point_size : float, optional
        Size of points, by default 5.0
    alpha : float, optional
        Point transparency (0-1), by default 0.7
    legend : bool, optional
        Show legend, by default True
    legend_loc : str, optional
        Legend location ('right margin', 'best', 'upper right', etc.)
    title : str, optional
        Plot title. If None, uses f'UMAP - {cluster_column}'
    figsize : tuple, optional
        Figure size (width, height), by default (8, 6)
    save_path : str, optional
        Path to save figure, by default None
    dpi : int, optional
        Resolution for saved figure, by default 300
    show : bool, optional
        Display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Basic plot with default colors
    >>> sj.plotting.plot_umap_by_clusters(sp, cluster_column='leiden')
    >>> 
    >>> # Use different colormap
    >>> sj.plotting.plot_umap_by_clusters(sp, color_map='Set3')
    >>> 
    >>> # Custom color list
    >>> my_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    >>> sj.plotting.plot_umap_by_clusters(sp, color_map=my_colors)
    >>> 
    >>> # Map specific clusters to specific colors
    >>> cluster_colors = {
    ...     '0': '#FF0000',  # Red for cluster 0
    ...     '1': '#00FF00',  # Green for cluster 1
    ...     '2': '#0000FF',  # Blue for cluster 2
    ... }
    >>> sj.plotting.plot_umap_by_clusters(sp, colors=cluster_colors)
    >>> 
    >>> # Use seaborn color palette
    >>> sj.plotting.plot_umap_by_clusters(sp, palette='husl')
    """
    # Check UMAP exists
    if not hasattr(spatioloji_obj, '_embeddings') or 'X_umap' not in spatioloji_obj.embeddings:
        raise ValueError(
            "UMAP not found. Run UMAP first:\n"
            "  sj.processing.umap(sp, use_pca=True)"
        )
    
    # Check cluster column exists
    if cluster_column not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{cluster_column}' not found in cell_meta")
    
    # Get data
    umap_coords = spatioloji_obj.embeddings['X_umap']
    clusters = spatioloji_obj.cell_meta[cluster_column].astype('category')
    unique_clusters = clusters.cat.categories
    n_clusters = len(unique_clusters)
    
    # Handle color specification
    if colors is not None:
        # Use explicit color mapping
        color_list = [colors.get(str(cluster), '#808080') for cluster in unique_clusters]
    elif palette is not None:
        color_map = palette  # Use palette as alias
    
    # Generate color list
    if colors is None:
        color_list = _get_categorical_colors(n_clusters, color_map)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
        mask = clusters == cluster
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=[color_list[i]],
            s=point_size,
            alpha=alpha,
            label=str(cluster),
            edgecolors='none'
        )
    
    # Formatting
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(title or f'UMAP - {cluster_column}', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    if legend:
        if legend_loc == 'right margin':
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=False,
                markerscale=2,
                title=cluster_column
            )
        else:
            ax.legend(
                loc=legend_loc,
                frameon=False,
                markerscale=2,
                title=cluster_column
            )
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_umap_by_gene(
    spatioloji_obj,
    gene_name: str,
    layer: Optional[str] = 'log_normalized',
    color_map: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    robust: bool = False,
    point_size: float = 5.0,
    alpha: float = 0.7,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    sort_by_value: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Plot UMAP colored by gene expression with extensive color customization.
    
    [docstring same as before...]
    """
    # Check UMAP exists
    if not hasattr(spatioloji_obj, '_embeddings') or 'X_umap' not in spatioloji_obj.embeddings:
        raise ValueError(
            "UMAP not found. Run UMAP first:\n"
            "  sj.processing.umap(sp, use_pca=True)"
        )
    
    # Get data - CORRECTED APPROACH
    umap_coords = spatioloji_obj.embeddings['X_umap']
    
    # Get gene expression from specified layer
    if layer is None:
        # Use main expression matrix
        gene_expr = spatioloji_obj.get_expression(gene_names=gene_name).flatten()
    else:
        # Get layer data
        layer_data = spatioloji_obj.get_layer(layer)
        
        # Get gene index
        gene_idx = spatioloji_obj._get_gene_indices([gene_name])[0]
        
        # Extract gene expression
        if hasattr(layer_data, 'toarray'):  # Sparse matrix
            gene_expr = layer_data[:, gene_idx].toarray().flatten()
        else:  # Dense array
            gene_expr = layer_data[:, gene_idx].flatten()
    
    # Set color limits
    if robust and vmin is None and vmax is None:
        vmin = np.percentile(gene_expr, 2)
        vmax = np.percentile(gene_expr, 98)
    
    # Handle diverging colormap with center
    if vcenter is not None:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort points by expression value if requested
    if sort_by_value:
        sort_idx = np.argsort(gene_expr)
        umap_coords = umap_coords[sort_idx]
        gene_expr = gene_expr[sort_idx]
    
    # Plot
    scatter = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=gene_expr,
        s=point_size,
        alpha=alpha,
        cmap=color_map,
        norm=norm,
        edgecolors='none'
    )
    
    # Formatting
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(title or f'{gene_name} Expression', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Colorbar
    if colorbar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(colorbar_label or 'Expression', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_umap_by_feature(
    spatioloji_obj,
    feature: str,
    categorical: Optional[bool] = None,
    color_map: Optional[Union[str, List, Dict]] = None,
    palette: Optional[Union[str, List, Dict]] = None,
    colors: Optional[Dict[str, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    point_size: float = 5.0,
    alpha: float = 0.7,
    legend: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Plot UMAP colored by any cell metadata feature with color customization.
    
    Automatically detects if feature is categorical or continuous.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with UMAP coordinates
    feature : str
        Column name in cell_meta to plot
    categorical : bool, optional
        Force categorical (True) or continuous (False) treatment.
        If None, auto-detects based on dtype
    color_map : str, list, or dict, optional
        For categorical: colormap name, list of colors, or dict mapping
        For continuous: colormap name
    palette : str, list, or dict, optional
        Alias for color_map
    colors : dict, optional
        For categorical: explicit mapping of values to colors
    vmin : float, optional
        For continuous: minimum value for color scale
    vmax : float, optional
        For continuous: maximum value for color scale
    vcenter : float, optional
        For continuous: center value for diverging colormaps
    point_size : float, optional
        Size of points, by default 5.0
    alpha : float, optional
        Point transparency (0-1), by default 0.7
    legend : bool, optional
        Show legend/colorbar, by default True
    title : str, optional
        Plot title. If None, uses feature name
    figsize : tuple, optional
        Figure size (width, height), by default (8, 6)
    save_path : str, optional
        Path to save figure, by default None
    dpi : int, optional
        Resolution for saved figure, by default 300
    show : bool, optional
        Display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> # Categorical feature
    >>> sj.plotting.plot_umap_by_feature(sp, feature='sample_id')
    >>> 
    >>> # Continuous feature
    >>> sj.plotting.plot_umap_by_feature(sp, feature='total_counts')
    >>> 
    >>> # Custom colors for categorical
    >>> sample_colors = {'sample1': '#FF0000', 'sample2': '#00FF00'}
    >>> sj.plotting.plot_umap_by_feature(sp, feature='sample_id', colors=sample_colors)
    """
    # Check feature exists
    if feature not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{feature}' not found in cell_meta")
    
    # Get data
    feature_data = spatioloji_obj.cell_meta[feature]
    
    # Auto-detect categorical vs continuous
    if categorical is None:
        categorical = pd.api.types.is_categorical_dtype(feature_data) or \
                     pd.api.types.is_object_dtype(feature_data) or \
                     feature_data.nunique() < 50
    
    # Use palette as alias
    if palette is not None:
        color_map = palette
    
    # Plot based on type
    if categorical:
        fig = plot_umap_by_clusters(
            spatioloji_obj,
            cluster_column=feature,
            color_map=color_map,
            colors=colors,
            point_size=point_size,
            alpha=alpha,
            legend=legend,
            title=title or f'UMAP - {feature}',
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
            show=show
        )
    else:
        # Continuous - create custom plot
        umap_coords = spatioloji_obj.embeddings['X_umap']
        
        # Handle diverging colormap
        if vcenter is not None:
            norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=feature_data.values,
            s=point_size,
            alpha=alpha,
            cmap=color_map or 'viridis',
            norm=norm,
            edgecolors='none'
        )
        
        ax.set_xlabel('UMAP1', fontsize=12)
        ax.set_ylabel('UMAP2', fontsize=12)
        ax.set_title(title or f'UMAP - {feature}', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if legend:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(feature, fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    return fig


def plot_umap_grid(
    spatioloji_obj,
    features: List[str],
    genes: Optional[List[str]] = None,
    layer: str = 'log_normalized',
    color_maps: Optional[Dict[str, str]] = None,
    ncols: int = 3,
    point_size: float = 3.0,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """Plot multiple UMAP plots in a grid."""
    # Combine features and genes
    all_items = features.copy()
    if genes:
        all_items.extend(genes)
    
    n_plots = len(all_items)
    nrows = int(np.ceil(n_plots / ncols))
    
    # Auto-calculate figsize
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Get UMAP coords
    umap_coords = spatioloji_obj.embeddings['X_umap']
    
    # Default colormaps
    default_color_maps = color_maps or {}
    
    # Plot each item
    for idx, item in enumerate(all_items):
        ax = axes[idx]
        
        # Get colormap for this item
        cmap = default_color_maps.get(item, None)
        
        # Check if gene or feature
        is_gene = item in spatioloji_obj.gene_index
        
        if is_gene:
            # Plot gene expression - CORRECTED
            gene_expr = _get_gene_expression(spatioloji_obj, item, layer)
            
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=gene_expr,
                s=point_size,
                alpha=0.7,
                cmap=cmap or 'viridis',
                edgecolors='none'
            )
            ax.set_title(f'{item}', fontsize=11, fontweight='bold')
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Plot feature
            if item not in spatioloji_obj.cell_meta.columns:
                ax.text(0.5, 0.5, f"'{item}' not found",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            feature_data = spatioloji_obj.cell_meta[item]
            is_categorical = pd.api.types.is_categorical_dtype(feature_data) or \
                           pd.api.types.is_object_dtype(feature_data)
            
            if is_categorical:
                # Categorical
                categories = feature_data.astype('category')
                unique_cats = categories.cat.categories
                n_cats = len(unique_cats)
                
                # Get colors
                if cmap:
                    colors = _get_categorical_colors(n_cats, cmap)
                else:
                    colors = _get_categorical_colors(n_cats, 'tab20')
                
                for i, cat in enumerate(unique_cats):
                    mask = categories == cat
                    ax.scatter(
                        umap_coords[mask, 0],
                        umap_coords[mask, 1],
                        c=[colors[i]],
                        s=point_size,
                        alpha=0.7,
                        label=str(cat),
                        edgecolors='none'
                    )
                
                ax.set_title(f'{item}', fontsize=11, fontweight='bold')
                
                # Small legend if few categories
                if n_cats <= 10:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                            frameon=False, fontsize=8)
            else:
                # Continuous
                scatter = ax.scatter(
                    umap_coords[:, 0],
                    umap_coords[:, 1],
                    c=feature_data.values,
                    s=point_size,
                    alpha=0.7,
                    cmap=cmap or 'viridis',
                    edgecolors='none'
                )
                ax.set_title(f'{item}', fontsize=11, fontweight='bold')
                plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xlabel('UMAP1', fontsize=9)
        ax.set_ylabel('UMAP2', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_pca(
    spatioloji_obj,
    color_by: Optional[str] = None,
    pcs: Tuple[int, int] = (1, 2),
    color_map: Optional[Union[str, List, Dict]] = None,
    colors: Optional[Dict[str, str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    point_size: float = 5.0,
    alpha: float = 0.7,
    show_variance: bool = True,
    legend: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Plot PCA coordinates with color customization.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with PCA coordinates
    color_by : str, optional
        Cell metadata column to color points by
    pcs : tuple of int, optional
        Which PCs to plot (1-indexed), by default (1, 2)
    color_map : str, list, or dict, optional
        Color specification (see plot_umap_by_clusters for details)
    colors : dict, optional
        Explicit color mapping for categorical features
    vmin : float, optional
        Minimum value for continuous color scale
    vmax : float, optional
        Maximum value for continuous color scale
    point_size : float, optional
        Size of points, by default 5.0
    alpha : float, optional
        Point transparency (0-1), by default 0.7
    show_variance : bool, optional
        Show variance explained in axis labels, by default True
    legend : bool, optional
        Show legend/colorbar, by default True
    figsize : tuple, optional
        Figure size (width, height), by default (8, 6)
    save_path : str, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure, by default 300
    show : bool, optional
        Display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> # Basic PCA
    >>> sj.plotting.plot_pca(sp, color_by='leiden')
    >>> 
    >>> # Custom colors
    >>> sj.plotting.plot_pca(sp, color_by='leiden', color_map='Set3')
    >>> 
    >>> # Different PCs
    >>> sj.plotting.plot_pca(sp, pcs=(2, 3), color_by='sample_id')
    """
    # Check PCA exists
    if not hasattr(spatioloji_obj, '_embeddings') or 'X_pca' not in spatioloji_obj.embeddings:
        raise ValueError(
            "PCA not found. Run PCA first:\n"
            "  sj.processing.pca(sp, n_comps=50)"
        )
    
    # Get PCA coords
    pca_coords = spatioloji_obj.embeddings['X_pca']
    pc1_idx, pc2_idx = pcs[0] - 1, pcs[1] - 1
    
    # Get variance explained
    if show_variance and 'X_pca_variance_ratio' in spatioloji_obj.embeddings:
        var_ratio = spatioloji_obj.embeddings['X_pca_variance_ratio']
        pc1_var = var_ratio[pc1_idx] * 100
        pc2_var = var_ratio[pc2_idx] * 100
    else:
        pc1_var = None
        pc2_var = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    if color_by is None:
        # No coloring
        ax.scatter(
            pca_coords[:, pc1_idx],
            pca_coords[:, pc2_idx],
            s=point_size,
            alpha=alpha,
            c='steelblue',
            edgecolors='none'
        )
    else:
        # Color by feature
        if color_by not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{color_by}' not found in cell_meta")
        
        feature_data = spatioloji_obj.cell_meta[color_by]
        is_categorical = pd.api.types.is_categorical_dtype(feature_data) or \
                        pd.api.types.is_object_dtype(feature_data)
        
        if is_categorical:
            categories = feature_data.astype('category')
            unique_cats = categories.cat.categories
            n_cats = len(unique_cats)
            
            # Get colors
            if colors is not None:
                color_list = [colors.get(str(cat), '#808080') for cat in unique_cats]
            else:
                color_list = _get_categorical_colors(n_cats, color_map)
            
            for i, cat in enumerate(unique_cats):
                mask = categories == cat
                ax.scatter(
                    pca_coords[mask, pc1_idx],
                    pca_coords[mask, pc2_idx],
                    c=[color_list[i]],
                    s=point_size,
                    alpha=alpha,
                    label=str(cat),
                    edgecolors='none'
                )
            
            if legend:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         frameon=False, title=color_by)
        else:
            # Continuous
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            scatter = ax.scatter(
                pca_coords[:, pc1_idx],
                pca_coords[:, pc2_idx],
                c=feature_data.values,
                s=point_size,
                alpha=alpha,
                cmap=color_map or 'viridis',
                norm=norm,
                edgecolors='none'
            )
            
            if legend:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_by, fontsize=11)
    
    # Formatting
    xlabel = f'PC{pcs[0]}'
    ylabel = f'PC{pcs[1]}'
    if pc1_var is not None:
        xlabel += f' ({pc1_var:.1f}%)'
    if pc2_var is not None:
        ylabel += f' ({pc2_var:.1f}%)'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('PCA', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_violin(
    spatioloji_obj,
    genes: Union[str, List[str]],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: Optional[List[str]] = None,
    color_map: Optional[Union[str, List]] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    rotation: int = 45,
    ylabel: Optional[str] = 'Expression',
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Violin plot of gene expression across groups with color and order customization.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    genes : str or list of str
        Gene name(s) to plot
    group_by : str, optional
        Cell metadata column for grouping, by default 'leiden'
    layer : str, optional
        Expression layer to use, by default 'log_normalized'
    group_order : list of str, optional
        Order of groups to display on x-axis. If None, uses sorted order.
        Can be subset of all groups - only specified groups will be shown.
        Example: ['0', '2', '1', '3'] or ['Epithelial', 'T cells', 'B cells']
    color_map : str or list, optional
        Colormap name or list of colors for violins
    colors : dict, optional
        Explicit mapping of groups to colors. Keys should match group names.
        Example: {'0': '#FF0000', '1': '#00FF00', '2': '#0000FF'}
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculates based on number of genes
    rotation : int, optional
        Rotation angle for x-axis labels, by default 45
    ylabel : str, optional
        Y-axis label, by default 'Expression'
    save_path : str, optional
        Path to save figure
    dpi : int, optional
        Resolution for saved figure, by default 300
    show : bool, optional
        Display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Basic usage with default order
    >>> sj.plotting.plot_violin(sp, genes='EPCAM', group_by='leiden')
    >>> 
    >>> # Custom order of clusters
    >>> sj.plotting.plot_violin(
    ...     sp,
    ...     genes='EPCAM',
    ...     group_by='leiden',
    ...     group_order=['0', '2', '5', '1', '3', '4']
    ... )
    >>> 
    >>> # Show only specific clusters in specific order
    >>> sj.plotting.plot_violin(
    ...     sp,
    ...     genes='EPCAM',
    ...     group_by='leiden',
    ...     group_order=['0', '2', '5']  # Only show these clusters
    ... )
    >>> 
    >>> # Order cell types logically
    >>> sj.plotting.plot_violin(
    ...     sp,
    ...     genes=['EPCAM', 'CD3D'],
    ...     group_by='cell_type',
    ...     group_order=['Epithelial', 'T cells', 'B cells', 'Myeloid', 'Fibroblasts']
    ... )
    >>> 
    >>> # Combine with custom colors matching the order
    >>> my_order = ['0', '2', '1', '3']
    >>> my_colors = {
    ...     '0': '#FF6B6B',
    ...     '2': '#4ECDC4',
    ...     '1': '#45B7D1',
    ...     '3': '#96CEB4'
    ... }
    >>> sj.plotting.plot_violin(
    ...     sp,
    ...     genes=['EPCAM', 'CD3D'],
    ...     group_by='leiden',
    ...     group_order=my_order,
    ...     colors=my_colors
    ... )
    >>> 
    >>> # Order samples by experimental design
    >>> sj.plotting.plot_violin(
    ...     sp,
    ...     genes='EPCAM',
    ...     group_by='sample_id',
    ...     group_order=['Control_1', 'Control_2', 'Treatment_1', 'Treatment_2']
    ... )
    """
    
    if isinstance(genes, str):
        genes = [genes]
    
    # VALIDATE GENES - NEW
    genes = _validate_and_filter_genes(spatioloji_obj, genes, verbose=True)

    
    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    
    n_genes = len(genes)
    
    # Auto-calculate figsize
    if figsize is None:
        figsize = (max(8, n_genes * 3), 5)
    
    # Create figure
    fig, axes = plt.subplots(1, n_genes, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Get groups
    groups = spatioloji_obj.cell_meta[group_by]
    
    # Determine group order
    if group_order is not None:
        # Use specified order
        # Convert to strings to ensure matching
        group_order_str = [str(g) for g in group_order]
        
        # Check which groups actually exist in the data
        available_groups = set(groups.astype(str).unique())
        ordered_groups = [g for g in group_order_str if g in available_groups]
        
        # Warn about missing groups
        missing_groups = [g for g in group_order_str if g not in available_groups]
        if missing_groups:
            warnings.warn(
                f"The following groups in group_order are not in the data: {missing_groups}"
            )
        
        if len(ordered_groups) == 0:
            raise ValueError(
                f"None of the groups in group_order exist in '{group_by}'. "
                f"Available groups: {sorted(available_groups)}"
            )
        
        unique_groups = ordered_groups
    else:
        # Use default sorted order
        unique_groups = sorted(groups.astype(str).unique())
    
    n_groups = len(unique_groups)
    
    # Get colors
    if colors is not None:
        violin_colors = [colors.get(str(g), 'steelblue') for g in unique_groups]
    else:
        violin_colors = _get_categorical_colors(n_groups, color_map or 'tab20')
    
    for idx, gene in enumerate(genes):
        # Get expression
        gene_expr = _get_gene_expression(spatioloji_obj, gene, layer)
        
        # Prepare data in specified order
        data_dict = {}
        for group in unique_groups:
            # Convert group to match the data type
            mask = groups.astype(str) == str(group)
            if mask.sum() > 0:  # Only include if there are cells
                data_dict[group] = gene_expr[mask]
        
        # Plot
        ax = axes[idx]
        
        if len(data_dict) == 0:
            ax.text(0.5, 0.5, f"No data for {gene}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        parts = ax.violinplot(
            list(data_dict.values()),
            positions=range(len(data_dict)),
            showmeans=True,
            showextrema=True
        )
        
        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i])
            pc.set_alpha(0.7)
        
        # Formatting
        ax.set_xticks(range(len(data_dict)))
        ax.set_xticklabels(list(data_dict.keys()), rotation=rotation, ha='right')
        ax.set_ylabel(ylabel if idx == 0 else '', fontsize=11)
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_heatmap(
    spatioloji_obj,
    genes: Union[str, List[str]],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: Optional[List[str]] = None,
    gene_order: Optional[List[str]] = None,
    scale: Literal['none', 'row', 'column'] = 'row',
    cluster_genes: bool = False,
    cluster_groups: bool = False,
    show_gene_names: bool = True,
    show_group_names: bool = True,
    color_map: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = 0,
    figsize: Optional[Tuple[float, float]] = None,
    dendrogram_ratio: float = 0.1,
    cbar_pos: Optional[Tuple[float, float, float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """
    Plot heatmap of gene expression across groups.

    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    
    if isinstance(genes, str):
        genes = [genes]
    
    # VALIDATE GENES - NEW
    genes = _validate_and_filter_genes(spatioloji_obj, genes, verbose=True)
    
    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    
    # Get groups
    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    
    # Determine group order
    if cluster_groups:
        unique_groups = None  # Will be determined after clustering
    elif group_order is not None:
        # Use specified order
        available_groups = set(groups.unique())
        unique_groups = [str(g) for g in group_order if str(g) in available_groups]
        if len(unique_groups) == 0:
            raise ValueError(f"None of the groups in group_order exist in '{group_by}'")
    else:
        # Default sorted order
        unique_groups = sorted(groups.unique())
    
    # Calculate mean expression per group
    n_genes = len(genes)
    if unique_groups is not None:
        n_groups = len(unique_groups)
        expr_matrix = np.zeros((n_genes, n_groups))
        
        for i, gene in enumerate(genes):
            try:
                gene_expr = _get_gene_expression(spatioloji_obj, gene, layer)
                for j, group in enumerate(unique_groups):
                    group_mask = groups == group
                    if group_mask.sum() > 0:
                        expr_matrix[i, j] = gene_expr[group_mask].mean()
            except ValueError as e:
                # This shouldn't happen after validation, but handle it anyway
                warnings.warn(f"Skipping gene '{gene}': {e}")
                continue
    else:
        # Will calculate after getting all groups
        all_groups = sorted(groups.unique())
        n_groups = len(all_groups)
        expr_matrix = np.zeros((n_genes, n_groups))
        
        for i, gene in enumerate(genes):
            try:
                gene_expr = _get_gene_expression(spatioloji_obj, gene, layer)
                for j, group in enumerate(all_groups):
                    group_mask = groups == group
                    if group_mask.sum() > 0:
                        expr_matrix[i, j] = gene_expr[group_mask].mean()
            except ValueError as e:
                warnings.warn(f"Skipping gene '{gene}': {e}")
                continue
        
        unique_groups = all_groups
    
    # Apply gene ordering before clustering
    if not cluster_genes and gene_order is not None:
        # VALIDATE gene_order - NEW
        gene_order_valid = [g for g in gene_order if g in genes]
        if len(gene_order_valid) == 0:
            warnings.warn("None of the genes in gene_order are available, using default order")
        else:
            # Reorder genes
            gene_indices = [genes.index(g) for g in gene_order_valid]
            genes = [genes[i] for i in gene_indices]
            expr_matrix = expr_matrix[gene_indices, :]
    
    # [Rest of the function remains the same...]
    # Apply scaling
    if scale == 'row':
        # Z-score per gene (row)
        expr_matrix = (expr_matrix - expr_matrix.mean(axis=1, keepdims=True)) / \
                     (expr_matrix.std(axis=1, keepdims=True) + 1e-10)
    elif scale == 'column':
        # Z-score per group (column)
        expr_matrix = (expr_matrix - expr_matrix.mean(axis=0, keepdims=True)) / \
                     (expr_matrix.std(axis=0, keepdims=True) + 1e-10)
    
    # Hierarchical clustering
    gene_linkage = None
    group_linkage = None
    
    if cluster_genes and n_genes > 1:
        try:
            gene_distances = pdist(expr_matrix, metric='euclidean')
            gene_linkage = linkage(gene_distances, method='average')
            gene_dendro = dendrogram(gene_linkage, no_plot=True)
            gene_order_idx = gene_dendro['leaves']
            expr_matrix = expr_matrix[gene_order_idx, :]
            genes = [genes[i] for i in gene_order_idx]
        except Exception as e:
            warnings.warn(f"Gene clustering failed: {e}")
    
    if cluster_groups and n_groups > 1:
        try:
            group_distances = pdist(expr_matrix.T, metric='euclidean')
            group_linkage = linkage(group_distances, method='average')
            group_dendro = dendrogram(group_linkage, no_plot=True)
            group_order_idx = group_dendro['leaves']
            expr_matrix = expr_matrix[:, group_order_idx]
            unique_groups = [unique_groups[i] for i in group_order_idx]
        except Exception as e:
            warnings.warn(f"Group clustering failed: {e}")
    
    # Auto-calculate figsize
    if figsize is None:
        width = max(8, n_groups * 0.4 + 2)
        height = max(6, n_genes * 0.3 + 2)
        figsize = (width, height)
    
    # Set up color normalization
    if vcenter is not None:
        if vmin is None and vmax is None:
            vmax = max(abs(np.nanmin(expr_matrix) - vcenter), 
                      abs(np.nanmax(expr_matrix) - vcenter))
            vmin = vcenter - vmax
            vmax = vcenter + vmax
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure with GridSpec
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=figsize)
    
    # Calculate ratios for dendrograms
    if cluster_genes and cluster_groups:
        gs = GridSpec(2, 2, figure=fig,
                     width_ratios=[dendrogram_ratio, 1],
                     height_ratios=[dendrogram_ratio, 1],
                     hspace=0.02, wspace=0.02)
        ax_heatmap = fig.add_subplot(gs[1, 1])
        ax_gene_dendro = fig.add_subplot(gs[1, 0], sharey=ax_heatmap)
        ax_group_dendro = fig.add_subplot(gs[0, 1], sharex=ax_heatmap)
    elif cluster_genes:
        gs = GridSpec(1, 2, figure=fig,
                     width_ratios=[dendrogram_ratio, 1],
                     wspace=0.02)
        ax_heatmap = fig.add_subplot(gs[0, 1])
        ax_gene_dendro = fig.add_subplot(gs[0, 0], sharey=ax_heatmap)
        ax_group_dendro = None
    elif cluster_groups:
        gs = GridSpec(2, 1, figure=fig,
                     height_ratios=[dendrogram_ratio, 1],
                     hspace=0.02)
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_group_dendro = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
        ax_gene_dendro = None
    else:
        ax_heatmap = fig.add_subplot(111)
        ax_gene_dendro = None
        ax_group_dendro = None
    
    # Plot heatmap
    im = ax_heatmap.imshow(
        expr_matrix,
        aspect='auto',
        cmap=color_map,
        norm=norm,
        interpolation='nearest'
    )
    
    # Set ticks and labels
    ax_heatmap.set_xticks(range(n_groups))
    ax_heatmap.set_yticks(range(n_genes))
    
    if show_group_names:
        ax_heatmap.set_xticklabels(unique_groups, rotation=90, ha='center')
    else:
        ax_heatmap.set_xticklabels([])
    
    if show_gene_names:
        ax_heatmap.set_yticklabels(genes)
    else:
        ax_heatmap.set_yticklabels([])
    
    # Move x-axis to top if there's a dendrogram
    if cluster_groups:
        ax_heatmap.xaxis.tick_top()
        ax_heatmap.xaxis.set_label_position('top')
    
    # Plot dendrograms
    if ax_gene_dendro is not None and gene_linkage is not None:
        dendrogram(gene_linkage, ax=ax_gene_dendro, orientation='left')
        ax_gene_dendro.set_xticks([])
        ax_gene_dendro.set_yticks([])
        ax_gene_dendro.spines['top'].set_visible(False)
        ax_gene_dendro.spines['right'].set_visible(False)
        ax_gene_dendro.spines['bottom'].set_visible(False)
        ax_gene_dendro.spines['left'].set_visible(False)
    
    if ax_group_dendro is not None and group_linkage is not None:
        dendrogram(group_linkage, ax=ax_group_dendro, orientation='top')
        ax_group_dendro.set_xticks([])
        ax_group_dendro.set_yticks([])
        ax_group_dendro.spines['top'].set_visible(False)
        ax_group_dendro.spines['right'].set_visible(False)
        ax_group_dendro.spines['bottom'].set_visible(False)
        ax_group_dendro.spines['left'].set_visible(False)
    
    # Add colorbar
    if cbar_pos is not None:
        cbar_ax = fig.add_axes(cbar_pos)
        cbar = plt.colorbar(im, cax=cbar_ax)
    else:
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    
    if scale == 'row':
        cbar.set_label('Z-score', fontsize=11)
    elif scale == 'column':
        cbar.set_label('Z-score', fontsize=11)
    else:
        cbar.set_label('Expression', fontsize=11)
    
    # Formatting
    ax_heatmap.spines['top'].set_visible(False)
    ax_heatmap.spines['right'].set_visible(False)
    ax_heatmap.spines['bottom'].set_visible(False)
    ax_heatmap.spines['left'].set_visible(False)
    
    plt.suptitle(f'Gene Expression - {group_by}', fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    # Show
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_dotplot(
    spatioloji_obj,
    genes: Union[str, List[str]],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: Optional[List[str]] = None,
    gene_order: Optional[List[str]] = None,
    size_scale: Tuple[float, float] = (20, 200),
    color_map: str = 'Reds',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_gene_names: bool = True,
    show_group_names: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True
) -> plt.Figure:
    """Plot dot plot showing gene expression and fraction of cells expressing."""
    
    if isinstance(genes, str):
        genes = [genes]
    
    # VALIDATE GENES - NEW
    genes = _validate_and_filter_genes(spatioloji_obj, genes, verbose=True)
    
    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    
    # Get groups
    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    
    # Determine group order
    if group_order is not None:
        available_groups = set(groups.unique())
        unique_groups = [str(g) for g in group_order if str(g) in available_groups]
    else:
        unique_groups = sorted(groups.unique())
    
    # Apply gene order
    if gene_order is not None:
        # VALIDATE gene_order - NEW
        gene_order_valid = [g for g in gene_order if g in genes]
        if len(gene_order_valid) > 0:
            genes = gene_order_valid
    
    n_genes = len(genes)
    n_groups = len(unique_groups)
    
    # Calculate mean expression and fraction expressing
    mean_expr = np.zeros((n_genes, n_groups))
    frac_expr = np.zeros((n_genes, n_groups))
    
    for i, gene in enumerate(genes):
        try:
            gene_expr = _get_gene_expression(spatioloji_obj, gene, layer)
            for j, group in enumerate(unique_groups):
                group_mask = groups == group
                group_expr = gene_expr[group_mask]
                
                if len(group_expr) > 0:
                    # Mean of expressing cells
                    expressing = group_expr[group_expr > 0]
                    if len(expressing) > 0:
                        mean_expr[i, j] = expressing.mean()
                    # Fraction expressing
                    frac_expr[i, j] = (group_expr > 0).sum() / len(group_expr)
        except ValueError as e:
            warnings.warn(f"Skipping gene '{gene}': {e}")
            continue
    
    # [Rest of the function remains the same...]
    # Auto-calculate figsize
    if figsize is None:
        width = max(8, n_groups * 0.6 + 2)
        height = max(6, n_genes * 0.5 + 1)
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create meshgrid for positions
    x_pos = np.arange(n_groups)
    y_pos = np.arange(n_genes)
    X, Y = np.meshgrid(x_pos, y_pos)
    
    # Flatten for scatter
    x_flat = X.flatten()
    y_flat = Y.flatten()
    colors = mean_expr.flatten()
    sizes = frac_expr.flatten()
    
    # Scale sizes
    sizes = size_scale[0] + (sizes * (size_scale[1] - size_scale[0]))
    
    # Plot
    scatter = ax.scatter(
        x_flat,
        y_flat,
        c=colors,
        s=sizes,
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.8
    )
    
    # Set ticks
    ax.set_xticks(x_pos)
    ax.set_yticks(y_pos)
    
    if show_group_names:
        ax.set_xticklabels(unique_groups, rotation=90, ha='center')
    else:
        ax.set_xticklabels([])
    
    if show_gene_names:
        ax.set_yticklabels(genes)
    else:
        ax.set_yticklabels([])
    
    # Invert y-axis
    ax.invert_yaxis()
    
    # Add colorbars
    cbar_color = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar_color.set_label('Mean Expression', fontsize=10)
    
    # Legend for dot size
    legend_sizes = [0, 0.25, 0.5, 0.75, 1.0]
    legend_labels = ['0%', '25%', '50%', '75%', '100%']
    legend_points = []
    
    for size_val in legend_sizes:
        point_size = size_scale[0] + (size_val * (size_scale[1] - size_scale[0]))
        legend_points.append(
            plt.scatter([], [], s=point_size, c='gray', edgecolors='black', 
                       linewidths=0.5, alpha=0.8)
        )
    
    legend = ax.legend(
        legend_points,
        legend_labels,
        title='% Expressing',
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),
        frameon=False,
        scatterpoints=1,
        fontsize=9,
        title_fontsize=10
    )
    
    # Formatting
    ax.set_xlabel(group_by, fontsize=12)
    ax.set_ylabel('Genes', fontsize=12)
    ax.set_title('Gene Expression Dot Plot', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def _get_gene_expression(spatioloji_obj, gene_name: str, layer: Optional[str] = None) -> np.ndarray:
    """
    Helper function to get gene expression from any layer.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object
    gene_name : str
        Gene name
    layer : str, optional
        Layer name. If None, uses main expression matrix
    
    Returns
    -------
    np.ndarray
        Gene expression values (1D array)
    
    Raises
    ------
    ValueError
        If gene is not found in the dataset
    """
    # Check if gene exists
    if gene_name not in spatioloji_obj.gene_index:
        raise ValueError(
            f"Gene '{gene_name}' not found in dataset. "
            f"Available genes: {spatioloji_obj.n_genes}"
        )
    
    if layer is None:
        # Use main expression matrix
        return spatioloji_obj.get_expression(gene_names=gene_name).flatten()
    else:
        # Get layer data
        layer_data = spatioloji_obj.get_layer(layer)
        
        # Get gene index
        gene_indices = spatioloji_obj._get_gene_indices([gene_name])
        
        if len(gene_indices) == 0:
            raise ValueError(f"Gene '{gene_name}' not found in dataset")
        
        gene_idx = gene_indices[0]
        
        # Extract gene expression
        if sparse.issparse(layer_data):  # Sparse matrix
            return layer_data[:, gene_idx].toarray().flatten()
        else:  # Dense array
            return layer_data[:, gene_idx].flatten()


def _validate_and_filter_genes(spatioloji_obj, genes: List[str], verbose: bool = True) -> List[str]:
    """
    Validate gene list and filter out missing genes.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object
    genes : list of str
        Gene names to validate
    verbose : bool, optional
        Print warning about missing genes, by default True
    
    Returns
    -------
    list of str
        Filtered list of genes that exist in the dataset
    """
    available_genes = set(spatioloji_obj.gene_index)
    valid_genes = [g for g in genes if g in available_genes]
    missing_genes = [g for g in genes if g not in available_genes]
    
    if len(missing_genes) > 0 and verbose:
        warnings.warn(
            f"{len(missing_genes)} gene(s) not found in dataset and will be skipped: "
            f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}"
        )
    
    if len(valid_genes) == 0:
        raise ValueError(
            f"None of the specified genes were found in the dataset. "
            f"Dataset contains {spatioloji_obj.n_genes} genes."
        )
    
    return valid_genes
            
def _get_categorical_colors(
    n_categories: int,
    color_spec: Optional[Union[str, List, Dict]] = None
) -> List:
    """
    Generate list of colors for categorical data.
    
    Parameters
    ----------
    n_categories : int
        Number of categories to generate colors for
    color_spec : str, list, or dict, optional
        Color specification:
        - str: Matplotlib colormap name or seaborn palette name
        - list: List of colors (hex, named, or RGB tuples)
        - dict: Mapping (returns values in order)
        - None: Uses default (tab20 for ≤20, extended tab20 for >20)
    
    Returns
    -------
    list
        List of colors (as hex, RGB tuples, or named colors)
    """
    # If dict provided, extract values
    if isinstance(color_spec, dict):
        return list(color_spec.values())
    
    # If list provided, cycle through if needed
    if isinstance(color_spec, list):
        if len(color_spec) >= n_categories:
            return color_spec[:n_categories]
        else:
            # Cycle through list
            return [color_spec[i % len(color_spec)] for i in range(n_categories)]
    
    # If string, try as colormap or palette
    if isinstance(color_spec, str):
        # Try seaborn palette first
        try:
            import seaborn as sns
            palette = sns.color_palette(color_spec, n_categories)
            return palette
        except:
            pass
        
        # Try matplotlib colormap
        try:
            cmap = plt.get_cmap(color_spec)
            return [cmap(i / n_categories) for i in range(n_categories)]
        except:
            warnings.warn(f"Unknown colormap '{color_spec}', using default")
    
    # Default: use tab20 for ≤20, extended for >20
    if n_categories <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in np.linspace(0, 1, n_categories)]
    else:
        # Combine tab20, tab20b, tab20c for up to 60 categories
        colors = []
        for cmap_name in ['tab20', 'tab20b', 'tab20c']:
            cmap = plt.get_cmap(cmap_name)
            colors.extend([cmap(i) for i in np.linspace(0, 1, 20)])
        return colors[:n_categories]