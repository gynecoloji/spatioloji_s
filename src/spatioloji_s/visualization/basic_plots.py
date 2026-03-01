"""
basic_plots.py - Basic visualization functions for spatioloji objects

Provides standard single-cell analysis plots with extensive color customization options.

Architecture:
    Private helpers     → _finalize_plot, _get_categorical_colors, _get_gene_expression, etc.
    Core scatter engine → _scatter_embedding (handles categorical + continuous + gene)
    Public thin wrappers→ plot_umap, plot_pca, plot_umap_grid
    Specialized plots   → plot_violin, plot_heatmap, plot_dotplot
"""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy import sparse

# =============================================================================
# SECTION 1 — PRIVATE HELPERS
# =============================================================================

def _finalize_plot(fig: plt.Figure, save_path: str | None, dpi: int,
                   show: bool) -> plt.Figure:
    """Shared save / show / close logic."""
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {save_path}")
    if not show:
        plt.close(fig)
        return None      # Jupyter has no object to render
    plt.close(fig)
    return fig


def _clean_axes(ax: plt.Axes) -> None:
    """Remove top/right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _require_embedding(spatioloji_obj, key: str, hint: str) -> np.ndarray:
    """Fetch an embedding array or raise a helpful error."""
    emb = getattr(spatioloji_obj, '_embeddings', None) or {}
    if key not in emb:
        raise ValueError(f"{key} not found. {hint}")
    return emb[key]


def _resolve_color_spec(palette, color_map):
    """Let callers pass either `palette` or `color_map`; return one value."""
    if palette is not None:
        return palette
    return color_map


def _build_norm(vmin, vmax, vcenter):
    """Build a Normalize or TwoSlopeNorm."""
    if vcenter is not None:
        return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def _get_gene_expression(spatioloji_obj, gene_name: str,
                         layer: str | None = None) -> np.ndarray:
    """
    Get 1-D gene expression array from any layer.

    Raises ValueError if the gene is missing.
    """
    if gene_name not in spatioloji_obj.gene_index:
        raise ValueError(
            f"Gene '{gene_name}' not found in dataset "
            f"({spatioloji_obj.n_genes} genes available)."
        )
    if layer is None:
        return spatioloji_obj.get_expression(gene_names=gene_name).flatten()

    layer_data = spatioloji_obj.get_layer(layer)
    gene_idx = spatioloji_obj._get_gene_indices([gene_name])[0]
    if sparse.issparse(layer_data):
        return layer_data[:, gene_idx].toarray().flatten()
    return layer_data[:, gene_idx].flatten()


def _validate_and_filter_genes(spatioloji_obj, genes: list[str],
                               verbose: bool = True) -> list[str]:
    """Return only genes present in the dataset; warn about missing ones."""
    available = set(spatioloji_obj.gene_index)
    valid = [g for g in genes if g in available]
    missing = [g for g in genes if g not in available]

    if missing and verbose:
        preview = missing[:10]
        suffix = '...' if len(missing) > 10 else ''
        warnings.warn(
            f"{len(missing)} gene(s) not found and will be skipped: "
            f"{preview}{suffix}", stacklevel=2
        )
    if not valid:
        raise ValueError(
            "None of the specified genes were found in the dataset "
            f"({spatioloji_obj.n_genes} genes available)."
        )
    return valid


def _get_categorical_colors(
    n: int,
    spec: str | list | dict | None = None
) -> list:
    """
    Produce *n* colours from a flexible specification.

    spec can be:
      - dict  → values in insertion order
      - list  → cycle if too short
      - str   → try seaborn palette, then matplotlib cmap
      - None  → tab20 (≤20) or tab20+tab20b+tab20c (>20)
    """
    if isinstance(spec, dict):
        return list(spec.values())

    if isinstance(spec, list):
        return [spec[i % len(spec)] for i in range(n)] if spec else []

    if isinstance(spec, str):
        # seaborn palette
        try:
            return list(sns.color_palette(spec, n))
        except Exception:
            pass
        # matplotlib cmap
        try:
            cmap = plt.get_cmap(spec)
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        except Exception:
            warnings.warn(f"Unknown colormap '{spec}', falling back to default", stacklevel=2)

    # default
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    colors = []
    for name in ('tab20', 'tab20b', 'tab20c'):
        cm = plt.get_cmap(name)
        colors.extend(cm(i / 19) for i in range(20))
    return colors[:n]


# =============================================================================
# SECTION 2 — CORE SCATTER ENGINE
# =============================================================================

def _scatter_embedding(
    spatioloji_obj,
    coords: np.ndarray,
    *,
    # What to colour by — exactly one of these should be set
    color_by: str | None = None,        # cell_meta column
    gene: str | None = None,            # gene name
    layer: str | None = 'log_normalized',
    # Colour options
    color_map: str | list | dict | None = None,
    palette: str | list | dict | None = None,
    colors: dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    robust: bool = False,
    # Scatter options
    point_size: float = 5.0,
    alpha: float = 0.7,
    sort_by_value: bool = False,
    # Labels & legend
    xlabel: str = 'Dim 1',
    ylabel: str = 'Dim 2',
    title: str | None = None,
    legend: bool = True,
    legend_loc: str = 'right margin',
    colorbar_label: str | None = None,
    # Figure
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Unified scatter plot on any 2-D embedding.

    Handles three colouring modes automatically:
      1. *gene*       → continuous, from expression layer
      2. *color_by* (categorical column) → discrete legend
      3. *color_by* (numeric column)     → continuous colorbar
      4. neither      → uniform colour
    """
    cmap_spec = _resolve_color_spec(palette, color_map)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- gene colouring ---------------------------------------------------
    if gene is not None:
        values = _get_gene_expression(spatioloji_obj, gene, layer)
        if robust and vmin is None and vmax is None:
            vmin, vmax = np.percentile(values, [2, 98])
        norm = _build_norm(vmin, vmax, vcenter)

        order = np.argsort(values) if sort_by_value else np.arange(len(values))
        sc = ax.scatter(
            coords[order, 0], coords[order, 1],
            c=values[order], s=point_size, alpha=alpha,
            cmap=cmap_spec or 'viridis', norm=norm, edgecolors='none',
        )
        if legend:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(colorbar_label or 'Expression')
        title = title or f'{gene} Expression'

    # --- metadata colouring -----------------------------------------------
    elif color_by is not None:
        if color_by not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{color_by}' not found in cell_meta")
        data = spatioloji_obj.cell_meta[color_by]
        is_cat = (
            pd.api.types.is_categorical_dtype(data)
            or pd.api.types.is_object_dtype(data)
            or data.nunique() < 50
        )

        if is_cat:
            cats = data.astype('category')
            unique = cats.cat.categories
            if colors is not None:
                clist = [colors.get(str(c), '#808080') for c in unique]
            else:
                clist = _get_categorical_colors(len(unique), cmap_spec)
            for i, cat in enumerate(unique):
                mask = cats == cat
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[clist[i]], s=point_size, alpha=alpha,
                    label=str(cat), edgecolors='none',
                )
            if legend:
                kw = dict(frameon=False, markerscale=2, title=color_by)
                if legend_loc == 'right margin':
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', **kw)
                else:
                    ax.legend(loc=legend_loc, **kw)
        else:
            norm = _build_norm(vmin, vmax, vcenter)
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=data.values, s=point_size, alpha=alpha,
                cmap=cmap_spec or 'viridis', norm=norm, edgecolors='none',
            )
            if legend:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label(colorbar_label or color_by)
        title = title or f'{color_by}'

    # --- no colouring -----------------------------------------------------
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=point_size, alpha=alpha, c='steelblue', edgecolors='none',
        )
        title = title or ''

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    _clean_axes(ax)

    ax.set_aspect('equal', adjustable='datalim')

    if own_fig:
        return _finalize_plot(fig, save_path, dpi, show)
    return fig


# =============================================================================
# SECTION 3 — PUBLIC API (thin wrappers)
# =============================================================================

def plot_umap(
    spatioloji_obj,
    color_by: str | None = None,
    gene: str | None = None,
    layer: str | None = 'log_normalized',
    *,
    color_map: str | list | dict | None = None,
    palette: str | list | dict | None = None,
    colors: dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    robust: bool = False,
    point_size: float = 5.0,
    alpha: float = 0.7,
    sort_by_value: bool = False,
    legend: bool = True,
    legend_loc: str = 'right margin',
    colorbar_label: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Plot UMAP coloured by a metadata column, a gene, or nothing.

    This single function replaces the former ``plot_umap_by_clusters``,
    ``plot_umap_by_gene``, and ``plot_umap_by_feature``.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Object with UMAP in ``embeddings['X_umap']``.
    color_by : str, optional
        Column in ``cell_meta`` (categorical or continuous).
    gene : str, optional
        Gene name to colour by expression.
    layer : str, optional
        Expression layer for gene colouring (default ``'log_normalized'``).
    color_map / palette : str, list, or dict, optional
        Colour specification (see ``_get_categorical_colors``).
    colors : dict, optional
        Explicit ``{label: colour}`` mapping for categorical data.
    vmin, vmax, vcenter : float, optional
        Colour-scale limits (continuous only).
    robust : bool
        Clip colour scale to 2nd–98th percentile (gene only).
    point_size, alpha : float
        Scatter aesthetics.
    sort_by_value : bool
        Draw high-expression points last (gene only).
    legend / legend_loc : bool / str
        Legend control.
    title : str, optional
        Plot title (auto-generated if *None*).
    figsize, save_path, dpi, show
        Standard figure options.

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> sj.plotting.plot_umap(sp, color_by='leiden')
    >>> sj.plotting.plot_umap(sp, gene='EPCAM')
    >>> sj.plotting.plot_umap(sp, color_by='total_counts', color_map='plasma')
    >>> sj.plotting.plot_umap(sp, gene='CD3D', robust=True, sort_by_value=True)
    """
    umap = _require_embedding(
        spatioloji_obj, 'X_umap',
        "Run UMAP first:  sj.processing.umap(sp, use_pca=True)"
    )
    return _scatter_embedding(
        spatioloji_obj, umap,
        color_by=color_by, gene=gene, layer=layer,
        color_map=color_map, palette=palette, colors=colors,
        vmin=vmin, vmax=vmax, vcenter=vcenter, robust=robust,
        point_size=point_size, alpha=alpha, sort_by_value=sort_by_value,
        xlabel='UMAP1', ylabel='UMAP2',
        title=title, legend=legend, legend_loc=legend_loc,
        colorbar_label=colorbar_label,
        figsize=figsize, save_path=save_path, dpi=dpi, show=show,
    )


def plot_pca(
    spatioloji_obj,
    color_by: str | None = None,
    gene: str | None = None,
    layer: str | None = 'log_normalized',
    pcs: tuple[int, int] = (1, 2),
    *,
    color_map: str | list | dict | None = None,
    palette: str | list | dict | None = None,
    colors: dict[str, str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 5.0,
    alpha: float = 0.7,
    show_variance: bool = True,
    legend: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Plot PCA coordinates, optionally coloured by metadata or gene.

    Parameters
    ----------
    pcs : tuple of int
        1-indexed PC pair to plot, e.g. ``(1, 2)``.
    show_variance : bool
        Annotate axes with % variance explained.
    (other params) : see ``plot_umap``.

    Examples
    --------
    >>> sj.plotting.plot_pca(sp, color_by='leiden')
    >>> sj.plotting.plot_pca(sp, pcs=(2, 3), gene='CD3D')
    """
    pca = _require_embedding(
        spatioloji_obj, 'X_pca',
        "Run PCA first:  sj.processing.pca(sp, n_comps=50)"
    )
    i, j = pcs[0] - 1, pcs[1] - 1
    coords = pca[:, [i, j]]

    # Build axis labels with variance if available
    xl, yl = f'PC{pcs[0]}', f'PC{pcs[1]}'
    if show_variance:
        vr = spatioloji_obj.embeddings.get('X_pca_variance_ratio')
        if vr is not None:
            xl += f' ({vr[i] * 100:.1f}%)'
            yl += f' ({vr[j] * 100:.1f}%)'

    return _scatter_embedding(
        spatioloji_obj, coords,
        color_by=color_by, gene=gene, layer=layer,
        color_map=color_map, palette=palette, colors=colors,
        vmin=vmin, vmax=vmax,
        point_size=point_size, alpha=alpha,
        xlabel=xl, ylabel=yl,
        title=title or 'PCA', legend=legend,
        figsize=figsize, save_path=save_path, dpi=dpi, show=show,
    )


def plot_umap_grid(
    spatioloji_obj,
    features: list[str] | None = None,
    genes: list[str] | None = None,
    layer: str = 'log_normalized',
    color_maps: dict[str, str] | None = None,
    ncols: int = 3,
    point_size: float = 3.0,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a grid of UMAP panels, each coloured by a different feature or gene.

    Parameters
    ----------
    features : list of str, optional
        Columns from ``cell_meta`` to plot.
    genes : list of str, optional
        Gene names to plot (from *layer*).
    layer : str
        Expression layer for gene panels.
    color_maps : dict, optional
        ``{item_name: cmap_name}`` overrides per panel.
    ncols : int
        Columns in the grid.

    Examples
    --------
    >>> sj.plotting.plot_umap_grid(sp, features=['leiden'], genes=['EPCAM','CD3D'])
    """
    umap = _require_embedding(
        spatioloji_obj, 'X_umap',
        "Run UMAP first:  sj.processing.umap(sp, use_pca=True)"
    )

    items: list[tuple[str, str]] = []  # (name, kind)  kind ∈ {'feature', 'gene'}
    for f in (features or []):
        items.append((f, 'feature'))
    for g in (genes or []):
        items.append((g, 'gene'))

    n = len(items)
    if n == 0:
        raise ValueError("Provide at least one feature or gene to plot.")

    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (ncols * 4, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    cmaps = color_maps or {}

    for idx, (name, kind) in enumerate(items):
        ax = axes[idx]
        cmap = cmaps.get(name)

        if kind == 'gene':
            try:
                _scatter_embedding(
                    spatioloji_obj, umap, gene=name, layer=layer,
                    color_map=cmap or 'viridis', point_size=point_size, alpha=0.7,
                    xlabel='UMAP1', ylabel='UMAP2', title=name,
                    ax=ax, show=False,
                )
            except ValueError:
                ax.text(0.5, 0.5, f"'{name}' not found",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            if name not in spatioloji_obj.cell_meta.columns:
                ax.text(0.5, 0.5, f"'{name}' not found",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            _scatter_embedding(
                spatioloji_obj, umap, color_by=name,
                color_map=cmap, point_size=point_size, alpha=0.7,
                xlabel='UMAP1', ylabel='UMAP2', title=name,
                ax=ax, show=False,
            )

    # hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    return _finalize_plot(fig, save_path, dpi, show)


# =============================================================================
# SECTION 4 — SPECIALIZED PLOTS  (violin · heatmap · dotplot)
# =============================================================================

# ---- helpers shared by Section 4 -----------------------------------------

def _resolve_group_order(groups: pd.Series,
                         group_order: list[str] | None) -> list[str]:
    """
    Return an ordered list of group labels.

    If *group_order* is given, keep only labels present in *groups* (with
    a warning for missing ones).  Otherwise return sorted unique values.
    """
    available = set(groups.astype(str).unique())
    if group_order is not None:
        order = [str(g) for g in group_order]
        present = [g for g in order if g in available]
        missing = [g for g in order if g not in available]
        if missing:
            warnings.warn(f"Groups not in data and will be skipped: {missing}", stacklevel=2)
        if not present:
            raise ValueError(
                f"None of the groups in group_order exist. "
                f"Available: {sorted(available)}"
            )
        return present
    return sorted(available)


# ---- violin --------------------------------------------------------------

def plot_violin(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: list[str] | None = None,
    color_map: str | list | None = None,
    colors: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    rotation: int = 45,
    ylabel: str | None = 'Expression',
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Violin plot of gene expression across groups.

    Parameters
    ----------
    genes : str or list of str
        Gene(s) to plot (one panel per gene).
    group_by : str
        Grouping column in ``cell_meta``.
    layer : str
        Expression layer to use.
    group_order : list of str, optional
        Display order (and optional subset) of groups.
    color_map / colors
        Colour specification (see ``_get_categorical_colors``).
    rotation : int
        X-tick label rotation.

    Examples
    --------
    >>> sj.plotting.plot_violin(sp, genes='EPCAM', group_by='leiden')
    >>> sj.plotting.plot_violin(sp, genes=['EPCAM','CD3D'],
    ...     group_order=['0','2','1'], colors={'0':'red','1':'blue','2':'green'})
    """
    if isinstance(genes, str):
        genes = [genes]
    genes = _validate_and_filter_genes(spatioloji_obj, genes)

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")

    groups = spatioloji_obj.cell_meta[group_by]
    unique_groups = _resolve_group_order(groups, group_order)
    n_groups = len(unique_groups)

    # colours
    if colors is not None:
        violin_colors = [colors.get(str(g), 'steelblue') for g in unique_groups]
    else:
        violin_colors = _get_categorical_colors(n_groups, color_map or 'tab20')

    n_genes = len(genes)
    if figsize is None:
        figsize = (max(8, n_genes * 3), 5)

    fig, axes = plt.subplots(1, n_genes, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, gene in enumerate(genes):
        ax = axes[idx]
        gene_expr = _get_gene_expression(spatioloji_obj, gene, layer)

        data_list, labels = [], []
        for g in unique_groups:
            mask = groups.astype(str) == str(g)
            if mask.sum() > 0:
                data_list.append(gene_expr[mask])
                labels.append(g)

        if not data_list:
            ax.text(0.5, 0.5, f"No data for {gene}",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        parts = ax.violinplot(data_list, positions=range(len(data_list)),
                              showmeans=True, showextrema=True)
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(violin_colors[i])
            body.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotation, ha='right')
        ax.set_ylabel(ylabel if idx == 0 else '', fontsize=11)
        ax.set_title(gene, fontsize=12, fontweight='bold')
        _clean_axes(ax)
        ax.grid(axis='y', alpha=0.3)

    return _finalize_plot(fig, save_path, dpi, show)


# ---- heatmap -------------------------------------------------------------

def plot_heatmap(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: list[str] | None = None,
    gene_order: list[str] | None = None,
    scale: Literal['none', 'row', 'column'] = 'row',
    cluster_genes: bool = False,
    cluster_groups: bool = False,
    show_gene_names: bool = True,
    show_group_names: bool = True,
    color_map: str = 'RdBu_r',
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = 0,
    figsize: tuple[float, float] | None = None,
    dendrogram_ratio: float = 0.1,
    cbar_pos: tuple[float, float, float, float] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Heatmap of mean gene expression per group.

    Parameters
    ----------
    scale : {'none', 'row', 'column'}
        Z-score normalisation direction.
    cluster_genes / cluster_groups : bool
        Hierarchical clustering with dendrograms.
    dendrogram_ratio : float
        Relative width/height of dendrogram panels.

    Examples
    --------
    >>> sj.plotting.plot_heatmap(sp, genes=marker_genes, group_by='leiden')
    """
    from matplotlib.gridspec import GridSpec
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    if isinstance(genes, str):
        genes = [genes]
    genes = _validate_and_filter_genes(spatioloji_obj, genes)

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")

    groups = spatioloji_obj.cell_meta[group_by].astype(str)

    # group order (may be overridden by clustering later)
    if not cluster_groups:
        unique_groups = _resolve_group_order(groups, group_order)
    else:
        unique_groups = sorted(groups.unique())

    # gene order (before clustering)
    if not cluster_genes and gene_order is not None:
        ordered = [g for g in gene_order if g in genes]
        if ordered:
            genes = ordered

    # --- build mean-expression matrix (genes × groups) --------------------
    n_genes, n_groups = len(genes), len(unique_groups)
    expr = np.zeros((n_genes, n_groups))
    for i, gene in enumerate(genes):
        vals = _get_gene_expression(spatioloji_obj, gene, layer)
        for j, grp in enumerate(unique_groups):
            mask = groups == grp
            if mask.sum():
                expr[i, j] = vals[mask].mean()

    # --- scaling ----------------------------------------------------------
    if scale == 'row':
        std = expr.std(axis=1, keepdims=True) + 1e-10
        expr = (expr - expr.mean(axis=1, keepdims=True)) / std
    elif scale == 'column':
        std = expr.std(axis=0, keepdims=True) + 1e-10
        expr = (expr - expr.mean(axis=0, keepdims=True)) / std

    # --- hierarchical clustering ------------------------------------------
    gene_link = group_link = None
    if cluster_genes and n_genes > 1:
        try:
            gene_link = linkage(pdist(expr), method='average')
            leaves = dendrogram(gene_link, no_plot=True)['leaves']
            expr = expr[leaves]
            genes = [genes[k] for k in leaves]
        except Exception as e:
            warnings.warn(f"Gene clustering failed: {e}", stacklevel=2)

    if cluster_groups and n_groups > 1:
        try:
            group_link = linkage(pdist(expr.T), method='average')
            leaves = dendrogram(group_link, no_plot=True)['leaves']
            expr = expr[:, leaves]
            unique_groups = [unique_groups[k] for k in leaves]
        except Exception as e:
            warnings.warn(f"Group clustering failed: {e}", stacklevel=2)

    # --- figure layout ----------------------------------------------------
    if figsize is None:
        figsize = (max(8, n_groups * 0.4 + 2), max(6, n_genes * 0.3 + 2))

    norm = _build_norm(vmin, vmax, vcenter)
    # auto-symmetric limits for z-scored data with vcenter
    if vcenter is not None and vmin is None and vmax is None:
        lim = max(abs(np.nanmin(expr) - vcenter), abs(np.nanmax(expr) - vcenter))
        norm = _build_norm(vcenter - lim, vcenter + lim, vcenter)

    fig = plt.figure(figsize=figsize)

    has_gd = cluster_genes and gene_link is not None
    has_cd = cluster_groups and group_link is not None

    if has_gd and has_cd:
        gs = GridSpec(2, 2, figure=fig,
                      width_ratios=[dendrogram_ratio, 1],
                      height_ratios=[dendrogram_ratio, 1],
                      hspace=0.02, wspace=0.02)
        ax_heat = fig.add_subplot(gs[1, 1])
        ax_gd = fig.add_subplot(gs[1, 0], sharey=ax_heat)
        ax_cd = fig.add_subplot(gs[0, 1], sharex=ax_heat)
    elif has_gd:
        gs = GridSpec(1, 2, figure=fig,
                      width_ratios=[dendrogram_ratio, 1], wspace=0.02)
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_gd = fig.add_subplot(gs[0, 0], sharey=ax_heat)
        ax_cd = None
    elif has_cd:
        gs = GridSpec(2, 1, figure=fig,
                      height_ratios=[dendrogram_ratio, 1], hspace=0.02)
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_cd = fig.add_subplot(gs[0, 0], sharex=ax_heat)
        ax_gd = None
    else:
        ax_heat = fig.add_subplot(111)
        ax_gd = ax_cd = None

    # --- draw heatmap -----------------------------------------------------
    im = ax_heat.imshow(expr, aspect='auto', cmap=color_map, norm=norm,
                        interpolation='nearest')
    ax_heat.set_xticks(range(n_groups))
    ax_heat.set_yticks(range(n_genes))
    ax_heat.set_xticklabels(unique_groups if show_group_names else [],
                            rotation=90, ha='center')
    ax_heat.set_yticklabels(genes if show_gene_names else [])
    if has_cd:
        ax_heat.xaxis.tick_top()
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # dendrograms
    def _draw_dendro(ax, link, orientation):
        dendrogram(link, ax=ax, orientation=orientation)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    if ax_gd is not None:
        _draw_dendro(ax_gd, gene_link, 'left')
    if ax_cd is not None:
        _draw_dendro(ax_cd, group_link, 'top')

    # colorbar
    if cbar_pos is not None:
        cbar = plt.colorbar(im, cax=fig.add_axes(cbar_pos))
    else:
        cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('Z-score' if scale in ('row', 'column') else 'Expression')

    plt.suptitle(f'Gene Expression - {group_by}', fontsize=14,
                 fontweight='bold', y=0.98)

    return _finalize_plot(fig, save_path, dpi, show)


# ---- dotplot --------------------------------------------------------------

def plot_dotplot(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = 'leiden',
    layer: str = 'log_normalized',
    group_order: list[str] | None = None,
    gene_order: list[str] | None = None,
    size_scale: tuple[float, float] = (20, 200),
    color_map: str = 'Reds',
    vmin: float | None = None,
    vmax: float | None = None,
    show_gene_names: bool = True,
    show_group_names: bool = True,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Dot plot: dot size = fraction expressing, colour = mean expression.

    Examples
    --------
    >>> sj.plotting.plot_dotplot(sp, genes=markers, group_by='leiden')
    """
    if isinstance(genes, str):
        genes = [genes]
    genes = _validate_and_filter_genes(spatioloji_obj, genes)

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")

    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    unique_groups = _resolve_group_order(groups, group_order)

    # optional gene reorder
    if gene_order is not None:
        valid = [g for g in gene_order if g in genes]
        if valid:
            genes = valid

    n_genes, n_groups = len(genes), len(unique_groups)

    mean_expr = np.zeros((n_genes, n_groups))
    frac_expr = np.zeros((n_genes, n_groups))

    for i, gene in enumerate(genes):
        vals = _get_gene_expression(spatioloji_obj, gene, layer)
        for j, grp in enumerate(unique_groups):
            mask = groups == grp
            sub = vals[mask]
            if len(sub):
                expressing = sub[sub > 0]
                if len(expressing):
                    mean_expr[i, j] = expressing.mean()
                frac_expr[i, j] = (sub > 0).sum() / len(sub)

    # --- figure -----------------------------------------------------------
    if figsize is None:
        figsize = (max(8, n_groups * 0.6 + 2), max(6, n_genes * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)

    X, Y = np.meshgrid(np.arange(n_groups), np.arange(n_genes))
    sizes = size_scale[0] + frac_expr.flatten() * (size_scale[1] - size_scale[0])

    sc = ax.scatter(
        X.flatten(), Y.flatten(),
        c=mean_expr.flatten(), s=sizes,
        cmap=color_map, vmin=vmin, vmax=vmax,
        edgecolors='black', linewidths=0.5, alpha=0.8,
    )

    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_genes))
    ax.set_xticklabels(unique_groups if show_group_names else [], rotation=90, ha='center')
    ax.set_yticklabels(genes if show_gene_names else [])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(n_genes - 0.5, -0.5)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Mean Expression', fontsize=10)

    # size legend
    _dot_size_legend(ax, size_scale)

    ax.set_xlabel(group_by, fontsize=12)
    ax.set_ylabel('Genes', fontsize=12)
    ax.set_title('Gene Expression Dot Plot', fontsize=14, fontweight='bold')
    _clean_axes(ax)
    ax.grid(False)

    return _finalize_plot(fig, save_path, dpi, show)


def _dot_size_legend(ax, size_scale):
    """Add a dot-size legend for fraction-expressing."""
    fracs = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['0%', '25%', '50%', '75%', '100%']
    handles = [
        ax.scatter([], [], s=size_scale[0] + f * (size_scale[1] - size_scale[0]),
                   c='gray', edgecolors='black', linewidths=0.5, alpha=0.8)
        for f in fracs
    ]
    ax.legend(handles, labels, title='% Expressing',
              loc='center left', bbox_to_anchor=(1.15, 0.5),
              frameon=False, scatterpoints=1, fontsize=9, title_fontsize=10)


# =============================================================================
# BACKWARD-COMPATIBLE ALIASES  (can be removed in a future version)
# =============================================================================

def plot_umap_by_clusters(spatioloji_obj, cluster_column='leiden', **kwargs):
    """Deprecated: use ``plot_umap(sp, color_by=...)`` instead."""
    warnings.warn(
        "plot_umap_by_clusters is deprecated; use plot_umap(sp, color_by=...) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return plot_umap(spatioloji_obj, color_by=cluster_column, **kwargs)


def plot_umap_by_gene(spatioloji_obj, gene_name, **kwargs):
    """Deprecated: use ``plot_umap(sp, gene=...)`` instead."""
    warnings.warn(
        "plot_umap_by_gene is deprecated; use plot_umap(sp, gene=...) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return plot_umap(spatioloji_obj, gene=gene_name, **kwargs)


def plot_umap_by_feature(spatioloji_obj, feature, **kwargs):
    """Deprecated: use ``plot_umap(sp, color_by=...)`` instead."""
    warnings.warn(
        "plot_umap_by_feature is deprecated; use plot_umap(sp, color_by=...) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return plot_umap(spatioloji_obj, color_by=feature, **kwargs)
