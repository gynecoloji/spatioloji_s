"""
interactive_plots.py - Interactive visualization functions for spatioloji objects

Uses Plotly for browser-based charts with zoom, pan, hover, and dropdown controls.
Requires: pip install plotly  (or: pip install spatioloji_s[interactive])

Public API
----------
Embedding plots:
    iplot_umap              – Interactive UMAP coloured by metadata or gene
    iplot_pca               – Interactive PCA scatter
    iplot_umap_grid         – Grid of UMAP panels

Distribution plots:
    iplot_violin            – Violin plot across groups
    iplot_heatmap           – Expression heatmap (groups × genes)
    iplot_dotplot           – Dot plot (size = % expressing, colour = mean expr)

Spatial dot plots:
    iplot_global_dots       – Global spatial scatter coloured by feature
    iplot_local_dots        – Per-FOV spatial scatter with FOV dropdown
    iplot_global_dots_gene  – Global spatial scatter coloured by gene expression
    iplot_local_dots_gene   – Per-FOV gene expression scatter with FOV dropdown

Spatial polygon plots:
    iplot_global_polygon        – Global polygon map coloured by feature
    iplot_local_polygon         – Per-FOV polygon map with FOV dropdown
    iplot_global_polygon_gene   – Global polygon map coloured by gene expression
    iplot_local_polygon_gene    – Per-FOV gene expression polygon map with FOV dropdown
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Lazy Plotly import — import fails only when a function is actually called
# ---------------------------------------------------------------------------
_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    pass


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Interactive plots require plotly. "
            "Install with: pip install plotly  or  pip install spatioloji_s[interactive]"
        )


# =============================================================================
# SECTION 1 — PRIVATE HELPERS
# =============================================================================


def _ifinalize(fig, save_path: str | None, show: bool):
    """Save (HTML or image) and optionally display the figure."""
    if save_path is not None:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            try:
                fig.write_image(save_path)
            except Exception as exc:
                warnings.warn(
                    f"Could not write image ({exc}). Use a .html path for guaranteed output.",
                    stacklevel=3,
                )
        print(f"Saved to {save_path}")
    if show:
        fig.show()
    return fig


def _get_expr_vector(sp, gene: str, layer: str | None) -> np.ndarray:
    """Return a 1-D expression array aligned to sp.cell_index."""
    if gene not in sp.gene_index:
        raise ValueError(f"Gene '{gene}' not found in dataset ({sp.n_genes} genes available).")
    if layer is None:
        return sp.get_expression(gene_names=gene).flatten()
    layer_data = sp.get_layer(layer)
    gene_idx = sp._get_gene_indices([gene])[0]
    if sparse.issparse(layer_data):
        return layer_data[:, gene_idx].toarray().flatten()
    return layer_data[:, gene_idx].flatten()


def _resolve_feature(sp, feature: str, feature_df, feature_column: str | None):
    """
    Resolve feature to (Series aligned to sp.cell_index, column_name).

    Priority: feature_df > feature="metadata" column > cell_meta column.
    """
    cell_id = sp.config.cell_id_col
    meta = sp.cell_meta.reset_index()

    if feature_df is not None:
        if cell_id not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain '{cell_id}' and '{feature}' columns")
        s = feature_df.set_index(cell_id)[feature].reindex(sp.cell_index.astype(str))
        return s, feature

    if feature == "metadata":
        if feature_column is None or feature_column not in meta.columns:
            raise ValueError(f"feature_column='{feature_column}' not found in cell_meta")
        s = meta.set_index(cell_id)[feature_column].reindex(sp.cell_index.astype(str))
        return s, feature_column

    if feature not in meta.columns:
        raise ValueError(f"Feature '{feature}' not found in cell_meta")
    s = meta.set_index(cell_id)[feature].reindex(sp.cell_index.astype(str))
    return s, feature


def _cat_color_map(cats: list, palette) -> dict:
    """Map category labels → 'rgb(r,g,b)' strings."""
    import seaborn as sns

    n = len(cats)
    if isinstance(palette, dict):
        return {str(k): palette.get(k, "#aaaaaa") for k in cats}
    if isinstance(palette, list):
        return {str(k): palette[i % len(palette)] for i, k in enumerate(cats)}
    pal = sns.color_palette(palette or "tab20", n)
    return {
        str(k): f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})" for k, c in zip(cats, pal, strict=False)
    }


def _mpl_to_plotly_colorscale(cmap_name: str, n: int = 12) -> list:
    """Convert matplotlib colormap name to a Plotly colorscale list."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    scale = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b = (int(cmap(t)[c] * 255) for c in range(3))
        scale.append([t, f"rgb({r},{g},{b})"])
    return scale


def _scatter_traces_categorical(x, y, values, color_dict, hover_text=None, point_size=4, alpha=0.7):
    """One go.Scattergl trace per category value."""
    traces = []
    cats = sorted(values.dropna().unique().tolist(), key=str)
    for cat in cats:
        mask = values == cat
        ht = hover_text[mask] if hover_text is not None else None
        traces.append(
            go.Scattergl(
                x=x[mask],
                y=y[mask],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color_dict.get(str(cat), "#aaaaaa"),
                    opacity=alpha,
                ),
                name=str(cat),
                text=ht,
                hovertemplate="%{text}<extra>" + str(cat) + "</extra>" if ht is not None else None,
                showlegend=True,
            )
        )
    return traces


def _scatter_trace_continuous(
    x, y, values, colorscale, vmin, vmax, feat_name, hover_text=None, point_size=4, alpha=0.7
):
    """Single go.Scattergl trace with continuous colorscale."""
    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax
    return go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=point_size,
            color=values,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            opacity=alpha,
            colorbar=dict(title=feat_name, thickness=14),
            showscale=True,
        ),
        name=feat_name,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>" if hover_text is not None else None,
        showlegend=False,
    )


def _polygon_traces_categorical(gdf, feat_col, color_dict, alpha=0.85):
    """One go.Scatter trace per category with None-separated polygon rings."""
    traces = []
    cats = sorted(gdf[feat_col].dropna().unique().tolist(), key=str)
    for cat in cats:
        subset = gdf[gdf[feat_col].astype(str) == str(cat)]
        if subset.empty:
            continue
        xs: list = []
        ys: list = []
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            coords = np.array(geom.exterior.coords)
            xs.extend(coords[:, 0].tolist() + [None])
            ys.extend(coords[:, 1].tolist() + [None])
        if not xs:
            continue
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                line=dict(width=0.3, color="rgba(0,0,0,0.15)"),
                fillcolor=color_dict.get(str(cat), "#aaaaaa"),
                opacity=alpha,
                name=str(cat),
                hoverinfo="name",
            )
        )
    return traces


def _polygon_traces_continuous(gdf, feat_col, cmap_name, vmin, vmax, alpha=0.85, n_bins=30):
    """Binned polygon traces for continuous data + invisible colorbar marker trace."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    vmin = float(gdf[feat_col].min()) if vmin is None else vmin
    vmax = float(gdf[feat_col].max()) if vmax is None else vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    gdf = gdf.copy()
    gdf["_bin"] = np.clip(np.digitize(gdf[feat_col].values, bin_edges[1:]), 0, n_bins - 1)

    traces = []
    for bin_idx in range(n_bins):
        subset = gdf[gdf["_bin"] == bin_idx]
        if subset.empty:
            continue
        bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        rgba = cmap(norm(bin_center))
        color_str = f"rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{alpha})"
        xs: list = []
        ys: list = []
        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            coords = np.array(geom.exterior.coords)
            xs.extend(coords[:, 0].tolist() + [None])
            ys.extend(coords[:, 1].tolist() + [None])
        if not xs:
            continue
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                line=dict(width=0),
                fillcolor=color_str,
                name=f"{bin_center:.3g}",
                hoverinfo="name",
                showlegend=False,
            )
        )

    # Invisible scatter purely for the colorbar
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0,
                colorscale=_mpl_to_plotly_colorscale(cmap_name),
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                colorbar=dict(title=feat_col, thickness=14),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return traces


def _fov_dropdown_layout(fig, fov_traces_counts: dict, default_fov: str) -> go.Figure:
    """
    Add a dropdown menu to a figure whose traces are ordered FOV-by-FOV.

    fov_traces_counts : {fov_id: n_traces}   (insertion order = trace order)
    default_fov       : FOV whose traces should be visible by default
    """
    total = sum(fov_traces_counts.values())
    buttons = []
    offset = 0
    for fov_id, n in fov_traces_counts.items():
        vis = [False] * total
        for i in range(offset, offset + n):
            vis[i] = True
        buttons.append(dict(label=f"FOV {fov_id}", method="update", args=[{"visible": vis}]))
        offset += n

    # Set initial visibility: default_fov
    init_offset = 0
    for fov_id, n in fov_traces_counts.items():
        if str(fov_id) == str(default_fov):
            break
        init_offset += n
    for i, trace in enumerate(fig.data):
        trace.visible = init_offset <= i < init_offset + fov_traces_counts[default_fov]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.06,
                yanchor="top",
            )
        ]
    )
    return fig


# =============================================================================
# SECTION 2 — EMBEDDING PLOTS
# =============================================================================


def iplot_umap(
    spatioloji_obj,
    color_by: str | None = None,
    gene: str | None = None,
    layer: str | None = "log_normalized",
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 4.0,
    alpha: float = 0.7,
    hover_cols: list[str] | None = None,
    title: str | None = None,
    width: int = 700,
    height: int = 550,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive UMAP coloured by a metadata column or gene expression.

    Args:
        spatioloji_obj: spatioloji object with ``embeddings['X_umap']``.
        color_by: Column in ``cell_meta`` to colour by.
        gene: Gene name to colour by expression.
        layer: Expression layer for gene colouring.
        palette: Colour spec for categorical data (str, list, or dict).
        cmap: Matplotlib colormap name for continuous data.
        vmin, vmax: Colour-scale limits (continuous only).
        point_size: Marker size in pixels.
        alpha: Marker opacity.
        hover_cols: Extra ``cell_meta`` columns to include in hover tooltips.
        title: Plot title (auto-generated if None).
        width, height: Figure dimensions in pixels.
        save_path: Path to save (.html or image with kaleido).
        show: Call ``fig.show()`` if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_umap(sp, color_by="leiden")
        >>> sj.visualization.iplot_umap(sp, gene="EPCAM")
    """
    _require_plotly()

    emb = getattr(spatioloji_obj, "_embeddings", {})
    if "X_umap" not in emb:
        raise ValueError("UMAP not found. Run sj.processing.umap(sp) first.")
    coords = emb["X_umap"]
    x, y = coords[:, 0], coords[:, 1]

    meta = spatioloji_obj.cell_meta.reset_index()
    cell_ids = spatioloji_obj.cell_index.astype(str)

    # Build hover text
    hover_parts = [f"Cell: {cid}" for cid in cell_ids]
    for col in hover_cols or []:
        if col in meta.columns:
            for i, val in enumerate(meta[col].values):
                hover_parts[i] += f"<br>{col}: {val}"
    hover_text = np.array(hover_parts)

    fig = go.Figure()

    if gene is not None:
        values = _get_expr_vector(spatioloji_obj, gene, layer)
        colorscale = _mpl_to_plotly_colorscale(cmap)
        for i, val in enumerate(values):
            hover_text[i] += f"<br>{gene}: {val:.3f}"
        fig.add_trace(
            _scatter_trace_continuous(x, y, values, colorscale, vmin, vmax, gene, hover_text, point_size, alpha)
        )
        auto_title = f"{gene} — UMAP"

    elif color_by is not None:
        if color_by not in meta.columns:
            raise ValueError(f"Column '{color_by}' not found in cell_meta")
        series = meta[color_by]
        is_cat = not pd.api.types.is_numeric_dtype(series) or series.nunique() < 50
        for i, val in enumerate(series.values):
            hover_text[i] += f"<br>{color_by}: {val}"

        if is_cat:
            cats = sorted(series.dropna().unique().tolist(), key=str)
            color_dict = _cat_color_map(cats, palette)
            for trace in _scatter_traces_categorical(x, y, series, color_dict, hover_text, point_size, alpha):
                fig.add_trace(trace)
        else:
            colorscale = _mpl_to_plotly_colorscale(cmap)
            fig.add_trace(
                _scatter_trace_continuous(
                    x, y, series.values, colorscale, vmin, vmax, color_by, hover_text, point_size, alpha
                )
            )
        auto_title = f"{color_by} — UMAP"

    else:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=point_size, color="steelblue", opacity=alpha),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        auto_title = "UMAP"

    fig.update_layout(
        title=title or auto_title,
        xaxis_title="UMAP1",
        yaxis_title="UMAP2",
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_pca(
    spatioloji_obj,
    color_by: str | None = None,
    gene: str | None = None,
    layer: str | None = "log_normalized",
    pcs: tuple[int, int] = (1, 2),
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 4.0,
    alpha: float = 0.7,
    hover_cols: list[str] | None = None,
    title: str | None = None,
    width: int = 700,
    height: int = 550,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive PCA scatter, optionally coloured by metadata or gene expression.

    Args:
        spatioloji_obj: spatioloji object with ``embeddings['X_pca']``.
        pcs: 1-indexed PC pair to plot, e.g. ``(1, 2)``.
        (other args): see :func:`iplot_umap`.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_pca(sp, color_by="leiden")
        >>> sj.visualization.iplot_pca(sp, pcs=(1, 3), gene="CD3D")
    """
    _require_plotly()

    emb = getattr(spatioloji_obj, "_embeddings", {})
    if "X_pca" not in emb:
        raise ValueError("PCA not found. Run sj.processing.pca(sp) first.")
    pca = emb["X_pca"]
    i, j = pcs[0] - 1, pcs[1] - 1
    x, y = pca[:, i], pca[:, j]

    # Variance labels
    xl, yl = f"PC{pcs[0]}", f"PC{pcs[1]}"
    vr = spatioloji_obj.embeddings.get("X_pca_variance_ratio")
    if vr is not None:
        xl += f" ({vr[i] * 100:.1f}%)"
        yl += f" ({vr[j] * 100:.1f}%)"

    meta = spatioloji_obj.cell_meta.reset_index()
    cell_ids = spatioloji_obj.cell_index.astype(str)
    hover_parts = [f"Cell: {cid}" for cid in cell_ids]
    for col in hover_cols or []:
        if col in meta.columns:
            for k, val in enumerate(meta[col].values):
                hover_parts[k] += f"<br>{col}: {val}"
    hover_text = np.array(hover_parts)

    fig = go.Figure()

    if gene is not None:
        values = _get_expr_vector(spatioloji_obj, gene, layer)
        colorscale = _mpl_to_plotly_colorscale(cmap)
        for k, val in enumerate(values):
            hover_text[k] += f"<br>{gene}: {val:.3f}"
        fig.add_trace(
            _scatter_trace_continuous(x, y, values, colorscale, vmin, vmax, gene, hover_text, point_size, alpha)
        )
        auto_title = f"{gene} — PCA"
    elif color_by is not None:
        if color_by not in meta.columns:
            raise ValueError(f"Column '{color_by}' not found in cell_meta")
        series = meta[color_by]
        is_cat = not pd.api.types.is_numeric_dtype(series) or series.nunique() < 50
        for k, val in enumerate(series.values):
            hover_text[k] += f"<br>{color_by}: {val}"
        if is_cat:
            cats = sorted(series.dropna().unique().tolist(), key=str)
            color_dict = _cat_color_map(cats, palette)
            for trace in _scatter_traces_categorical(x, y, series, color_dict, hover_text, point_size, alpha):
                fig.add_trace(trace)
        else:
            colorscale = _mpl_to_plotly_colorscale(cmap)
            fig.add_trace(
                _scatter_trace_continuous(
                    x, y, series.values, colorscale, vmin, vmax, color_by, hover_text, point_size, alpha
                )
            )
        auto_title = f"{color_by} — PCA"
    else:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=point_size, color="steelblue", opacity=alpha),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        auto_title = "PCA"

    fig.update_layout(
        title=title or auto_title,
        xaxis_title=xl,
        yaxis_title=yl,
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_umap_grid(
    spatioloji_obj,
    features: list[str] | None = None,
    genes: list[str] | None = None,
    layer: str = "log_normalized",
    ncols: int = 3,
    point_size: float = 3.0,
    alpha: float = 0.7,
    cmap: str = "viridis",
    subplot_size: int = 300,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Grid of interactive UMAP panels, each coloured by a different feature or gene.

    Args:
        spatioloji_obj: spatioloji object with ``embeddings['X_umap']``.
        features: List of ``cell_meta`` columns to plot.
        genes: List of gene names to plot.
        layer: Expression layer for gene panels.
        ncols: Number of columns in the grid.
        point_size: Marker size.
        alpha: Marker opacity.
        cmap: Matplotlib colormap name for continuous data.
        subplot_size: Pixel size of each subplot panel.
        save_path: Path to save (.html or image).
        show: Display the figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_umap_grid(sp, features=["leiden"], genes=["EPCAM", "CD3D"])
    """
    _require_plotly()

    emb = getattr(spatioloji_obj, "_embeddings", {})
    if "X_umap" not in emb:
        raise ValueError("UMAP not found. Run sj.processing.umap(sp) first.")
    coords = emb["X_umap"]

    items: list[tuple[str, str]] = []
    for f in features or []:
        items.append((f, "feature"))
    for g in genes or []:
        items.append((g, "gene"))
    if not items:
        raise ValueError("Provide at least one feature or gene.")

    n = len(items)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=False, shared_yaxes=False)

    meta = spatioloji_obj.cell_meta.reset_index()
    colorscale = _mpl_to_plotly_colorscale(cmap)

    for idx, (name, kind) in enumerate(items):
        row = idx // ncols + 1
        col = idx % ncols + 1
        try:
            if kind == "gene":
                values = _get_expr_vector(spatioloji_obj, name, layer)
                trace = _scatter_trace_continuous(
                    coords[:, 0], coords[:, 1], values, colorscale, None, None, name, point_size=point_size, alpha=alpha
                )
                trace.showlegend = False
                fig.add_trace(trace, row=row, col=col)
            else:
                if name not in meta.columns:
                    raise ValueError(f"Column '{name}' not found")
                series = meta[name]
                is_cat = not pd.api.types.is_numeric_dtype(series) or series.nunique() < 50
                if is_cat:
                    cats = sorted(series.dropna().unique().tolist(), key=str)
                    color_dict = _cat_color_map(cats, None)
                    for i, trace in enumerate(
                        _scatter_traces_categorical(
                            coords[:, 0], coords[:, 1], series, color_dict, point_size=point_size, alpha=alpha
                        )
                    ):
                        trace.showlegend = idx == 0 and i == 0
                        fig.add_trace(trace, row=row, col=col)
                else:
                    trace = _scatter_trace_continuous(
                        coords[:, 0],
                        coords[:, 1],
                        series.values,
                        colorscale,
                        None,
                        None,
                        name,
                        point_size=point_size,
                        alpha=alpha,
                    )
                    trace.showlegend = False
                    fig.add_trace(trace, row=row, col=col)
        except ValueError as exc:
            fig.add_annotation(
                text=str(exc), row=row, col=col, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )

        fig.update_xaxes(title_text=name, row=row, col=col)

    fig.update_layout(
        width=subplot_size * ncols,
        height=subplot_size * nrows,
        template="simple_white",
        showlegend=False,
    )
    return _ifinalize(fig, save_path, show)


# =============================================================================
# SECTION 3 — DISTRIBUTION PLOTS
# =============================================================================


def iplot_violin(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = "leiden",
    layer: str = "log_normalized",
    group_order: list[str] | None = None,
    palette: str | list | dict | None = None,
    show_box: bool = True,
    show_points: bool = False,
    title: str | None = None,
    width: int | None = None,
    height: int = 500,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive violin plot of gene expression across groups.

    Args:
        spatioloji_obj: spatioloji object.
        genes: Gene name(s) — one subplot per gene.
        group_by: Column in ``cell_meta`` to group by.
        layer: Expression layer.
        group_order: Ordered subset of group labels to display.
        palette: Colour specification for groups.
        show_box: Overlay box plot if True.
        show_points: Overlay individual points if True.
        title: Figure title.
        width, height: Figure dimensions in pixels.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_violin(sp, genes=["EPCAM", "CD3D"], group_by="leiden")
    """
    _require_plotly()

    if isinstance(genes, str):
        genes = [genes]

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    unique_groups = group_order or sorted(groups.unique().tolist())
    unique_groups = [str(g) for g in unique_groups]
    color_dict = _cat_color_map(unique_groups, palette)

    n_genes = len(genes)
    fig = make_subplots(rows=1, cols=n_genes, shared_yaxes=False, subplot_titles=genes)

    box_visible = "box" if show_box else False
    points_visible = "all" if show_points else False

    for gi, gene in enumerate(genes):
        try:
            expr = _get_expr_vector(spatioloji_obj, gene, layer)
        except ValueError:
            continue
        for g in unique_groups:
            mask = groups == g
            if mask.sum() == 0:
                continue
            fig.add_trace(
                go.Violin(
                    y=expr[mask.values],
                    name=g,
                    box_visible=box_visible,
                    points=points_visible,
                    fillcolor=color_dict.get(g, "#aaaaaa"),
                    line_color="black",
                    opacity=0.75,
                    showlegend=(gi == 0),
                    legendgroup=g,
                ),
                row=1,
                col=gi + 1,
            )
        fig.update_yaxes(title_text="Expression" if gi == 0 else "", row=1, col=gi + 1)

    fig.update_layout(
        title=title or f"Gene Expression by {group_by}",
        violinmode="group",
        width=width or max(600, n_genes * 200),
        height=height,
        template="simple_white",
        legend=dict(itemsizing="constant"),
    )
    return _ifinalize(fig, save_path, show)


def iplot_heatmap(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = "leiden",
    layer: str = "log_normalized",
    group_order: list[str] | None = None,
    scale: str = "row",
    cmap: str = "RdBu_r",
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive heatmap of mean gene expression per group.

    Args:
        spatioloji_obj: spatioloji object.
        genes: Gene name(s).
        group_by: Grouping column in ``cell_meta``.
        layer: Expression layer.
        group_order: Ordered group labels.
        scale: Z-score direction — ``'row'``, ``'column'``, or ``'none'``.
        cmap: Matplotlib colormap name.
        title: Figure title.
        width, height: Figure dimensions in pixels.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_heatmap(sp, genes=markers, group_by="leiden")
    """
    _require_plotly()

    if isinstance(genes, str):
        genes = [genes]

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    unique_groups = group_order or sorted(groups.unique().tolist())
    unique_groups = [str(g) for g in unique_groups]

    valid_genes = [g for g in genes if g in spatioloji_obj.gene_index]
    if not valid_genes:
        raise ValueError("None of the specified genes found in dataset.")

    n_genes, n_groups = len(valid_genes), len(unique_groups)
    expr = np.zeros((n_genes, n_groups))
    for i, gene in enumerate(valid_genes):
        vals = _get_expr_vector(spatioloji_obj, gene, layer)
        for j, grp in enumerate(unique_groups):
            mask = groups == grp
            if mask.sum():
                expr[i, j] = vals[mask.values].mean()

    if scale == "row":
        std = expr.std(axis=1, keepdims=True) + 1e-10
        expr = (expr - expr.mean(axis=1, keepdims=True)) / std
    elif scale == "column":
        std = expr.std(axis=0, keepdims=True) + 1e-10
        expr = (expr - expr.mean(axis=0, keepdims=True)) / std

    colorscale = _mpl_to_plotly_colorscale(cmap)
    zmid = 0 if scale in ("row", "column") else None

    fig = go.Figure(
        go.Heatmap(
            z=expr,
            x=unique_groups,
            y=valid_genes,
            colorscale=colorscale,
            zmid=zmid,
            colorbar=dict(title="Z-score" if scale != "none" else "Expression"),
            hovertemplate="Group: %{x}<br>Gene: %{y}<br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title or f"Gene Expression Heatmap — {group_by}",
        xaxis_title=group_by,
        yaxis_title="Genes",
        width=width or max(600, n_groups * 35 + 200),
        height=height or max(400, n_genes * 18 + 150),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_dotplot(
    spatioloji_obj,
    genes: str | list[str],
    group_by: str = "leiden",
    layer: str = "log_normalized",
    group_order: list[str] | None = None,
    size_range: tuple[float, float] = (4.0, 30.0),
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive dot plot: dot size = fraction expressing, colour = mean expression.

    Args:
        spatioloji_obj: spatioloji object.
        genes: Gene name(s).
        group_by: Grouping column in ``cell_meta``.
        layer: Expression layer.
        group_order: Ordered group labels.
        size_range: (min_px, max_px) marker size range.
        cmap: Matplotlib colormap name for mean expression.
        vmin, vmax: Colour-scale limits.
        title: Figure title.
        width, height: Figure dimensions in pixels.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_dotplot(sp, genes=markers, group_by="leiden")
    """
    _require_plotly()

    if isinstance(genes, str):
        genes = [genes]

    if group_by not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{group_by}' not found in cell_meta")
    groups = spatioloji_obj.cell_meta[group_by].astype(str)
    unique_groups = group_order or sorted(groups.unique().tolist())
    unique_groups = [str(g) for g in unique_groups]

    valid_genes = [g for g in genes if g in spatioloji_obj.gene_index]
    if not valid_genes:
        raise ValueError("None of the specified genes found in dataset.")

    n_genes, n_groups = len(valid_genes), len(unique_groups)
    mean_expr = np.zeros((n_genes, n_groups))
    frac_expr = np.zeros((n_genes, n_groups))

    for i, gene in enumerate(valid_genes):
        vals = _get_expr_vector(spatioloji_obj, gene, layer)
        for j, grp in enumerate(unique_groups):
            mask = groups == grp
            sub = vals[mask.values]
            if len(sub):
                pos = sub[sub > 0]
                if len(pos):
                    mean_expr[i, j] = pos.mean()
                frac_expr[i, j] = (sub > 0).sum() / len(sub)

    vmin = float(mean_expr.min()) if vmin is None else vmin
    vmax = float(mean_expr.max()) if vmax is None else vmax

    xs, ys, sizes, colors, hover = [], [], [], [], []
    for i, gene in enumerate(valid_genes):
        for j, grp in enumerate(unique_groups):
            xs.append(grp)
            ys.append(gene)
            s = size_range[0] + frac_expr[i, j] * (size_range[1] - size_range[0])
            sizes.append(s)
            colors.append(mean_expr[i, j])
            hover.append(
                f"Group: {grp}<br>Gene: {gene}<br>Mean expr: {mean_expr[i, j]:.3f}"
                f"<br>% expressing: {frac_expr[i, j] * 100:.1f}%"
            )

    colorscale = _mpl_to_plotly_colorscale(cmap)

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                colorbar=dict(title="Mean Expr", thickness=14),
                line=dict(width=0.5, color="black"),
                opacity=0.85,
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title or "Gene Expression Dot Plot",
        xaxis_title=group_by,
        yaxis_title="Genes",
        width=width or max(600, n_groups * 40 + 200),
        height=height or max(400, n_genes * 22 + 150),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


# =============================================================================
# SECTION 4 — SPATIAL DOT PLOTS
# =============================================================================


def iplot_global_dots(
    spatioloji_obj,
    feature: str,
    feature_df=None,
    feature_column: str | None = None,
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 4.0,
    alpha: float = 0.8,
    hover_cols: list[str] | None = None,
    title: str | None = None,
    width: int = 800,
    height: int = 750,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive global spatial scatter plot coloured by any feature.

    Args:
        spatioloji_obj: spatioloji object (must have ``CenterX_global_px`` in cell_meta).
        feature: Feature column in ``cell_meta``, or ``"metadata"`` (use ``feature_column``).
        feature_df: Optional external DataFrame with cell_id and feature columns.
        feature_column: Column to use when ``feature="metadata"``.
        palette: Colour spec for categorical data.
        cmap: Matplotlib colormap for continuous data.
        vmin, vmax: Colour-scale limits (continuous only).
        dot_size: Marker size in pixels.
        alpha: Marker opacity.
        hover_cols: Extra cell_meta columns in hover tooltip.
        title: Plot title.
        width, height: Figure dimensions in pixels.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_global_dots(sp, feature="cell_type")
        >>> sj.visualization.iplot_global_dots(sp, feature="total_counts", cmap="plasma")
    """
    _require_plotly()

    meta = spatioloji_obj.cell_meta.reset_index()
    for col in ("CenterX_global_px", "CenterY_global_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    feat_series, feat_col = _resolve_feature(spatioloji_obj, feature, feature_df, feature_column)
    cell_ids = spatioloji_obj.cell_index.astype(str)
    x = meta["CenterX_global_px"].values
    y = meta["CenterY_global_px"].values

    hover_parts = [f"Cell: {cid}<br>{feat_col}: {v}" for cid, v in zip(cell_ids, feat_series.values, strict=False)]
    for col in hover_cols or []:
        if col in meta.columns:
            for i, val in enumerate(meta[col].values):
                hover_parts[i] += f"<br>{col}: {val}"
    hover_text = np.array(hover_parts)

    fig = go.Figure()
    is_cat = not pd.api.types.is_numeric_dtype(feat_series)
    if is_cat:
        cats = sorted(feat_series.dropna().unique().tolist(), key=str)
        color_dict = _cat_color_map(cats, palette)
        for trace in _scatter_traces_categorical(x, y, feat_series, color_dict, hover_text, dot_size, alpha):
            fig.add_trace(trace)
    else:
        colorscale = _mpl_to_plotly_colorscale(cmap)
        fig.add_trace(
            _scatter_trace_continuous(
                x, y, feat_series.values, colorscale, vmin, vmax, feat_col, hover_text, dot_size, alpha
            )
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or feat_col,
        xaxis_title="X (global px)",
        yaxis_title="Y (global px)",
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_local_dots(
    spatioloji_obj,
    feature: str,
    fov_ids: list[str | int] | None = None,
    feature_df=None,
    feature_column: str | None = None,
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 4.0,
    alpha: float = 0.8,
    hover_cols: list[str] | None = None,
    title: str | None = None,
    width: int = 700,
    height: int = 650,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive per-FOV spatial scatter plot with FOV dropdown menu.

    Args:
        spatioloji_obj: spatioloji object (must have ``CenterX_local_px`` in cell_meta).
        feature: Feature column in ``cell_meta``, or ``"metadata"``.
        fov_ids: FOVs to include (all if None).
        (other args): see :func:`iplot_global_dots`.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_local_dots(sp, feature="cell_type", fov_ids=["1", "2", "3"])
    """
    _require_plotly()

    meta = spatioloji_obj.cell_meta.reset_index()
    fov_col = spatioloji_obj.config.fov_id_col
    for col in ("CenterX_local_px", "CenterY_local_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    feat_series, feat_col = _resolve_feature(spatioloji_obj, feature, feature_df, feature_column)
    all_fovs = sorted(meta[fov_col].astype(str).unique().tolist())
    if fov_ids is None:
        fov_ids = all_fovs
    fov_ids = [str(f) for f in fov_ids]

    is_cat = not pd.api.types.is_numeric_dtype(feat_series)
    cats = sorted(feat_series.dropna().unique().tolist(), key=str) if is_cat else None
    color_dict = _cat_color_map(cats, palette) if is_cat else None
    colorscale = None if is_cat else _mpl_to_plotly_colorscale(cmap)

    fig = go.Figure()
    fov_trace_counts: dict = {}
    first_fov = fov_ids[0]

    for fov_id in fov_ids:
        fov_mask = meta[fov_col].astype(str) == fov_id
        fov_meta = meta[fov_mask]
        fov_feat = feat_series.iloc[fov_mask.values]
        x = fov_meta["CenterX_local_px"].values
        y = fov_meta["CenterY_local_px"].values
        cell_ids = fov_meta[spatioloji_obj.config.cell_id_col].astype(str).values
        hover_parts = [f"Cell: {cid}<br>{feat_col}: {v}" for cid, v in zip(cell_ids, fov_feat.values, strict=False)]
        for col in hover_cols or []:
            if col in fov_meta.columns:
                for i, val in enumerate(fov_meta[col].values):
                    hover_parts[i] += f"<br>{col}: {val}"
        hover_text = np.array(hover_parts)

        n_before = len(fig.data)
        if is_cat:
            for trace in _scatter_traces_categorical(x, y, fov_feat, color_dict, hover_text, dot_size, alpha):
                trace.visible = fov_id == first_fov
                trace.showlegend = fov_id == first_fov
                trace.legendgroup = str(trace.name)
                fig.add_trace(trace)
        else:
            trace = _scatter_trace_continuous(
                x, y, fov_feat.values, colorscale, vmin, vmax, feat_col, hover_text, dot_size, alpha
            )
            trace.visible = fov_id == first_fov
            fig.add_trace(trace)
        fov_trace_counts[fov_id] = len(fig.data) - n_before

    # Dropdown
    total = len(fig.data)
    buttons = []
    offset = 0
    for fov_id in fov_ids:
        n = fov_trace_counts[fov_id]
        vis = [False] * total
        for i in range(offset, offset + n):
            vis[i] = True
        buttons.append(
            dict(
                label=f"FOV {fov_id}", method="update", args=[{"visible": vis}, {"title": f"{feat_col} — FOV {fov_id}"}]
            )
        )
        offset += n

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"{feat_col} — FOV {first_fov}",
        xaxis_title="X (local px)",
        yaxis_title="Y (local px)",
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0.01, xanchor="left", y=1.06, yanchor="top")
        ],
    )
    return _ifinalize(fig, save_path, show)


def iplot_global_dots_gene(
    spatioloji_obj,
    gene: str,
    layer: str | None = "log_normalized",
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 4.0,
    alpha: float = 0.8,
    title: str | None = None,
    width: int = 800,
    height: int = 750,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive global spatial scatter coloured by gene expression.

    Args:
        spatioloji_obj: spatioloji object.
        gene: Gene name.
        layer: Expression layer (None = raw counts).
        cmap: Matplotlib colormap name.
        vmin, vmax: Colour-scale limits.
        dot_size: Marker size.
        alpha: Marker opacity.
        title: Plot title (auto-generated if None).
        width, height: Figure dimensions.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_global_dots_gene(sp, gene="EPCAM")
    """
    _require_plotly()

    meta = spatioloji_obj.cell_meta.reset_index()
    for col in ("CenterX_global_px", "CenterY_global_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    values = _get_expr_vector(spatioloji_obj, gene, layer)
    cell_ids = spatioloji_obj.cell_index.astype(str)
    x = meta["CenterX_global_px"].values
    y = meta["CenterY_global_px"].values
    hover_text = np.array([f"Cell: {cid}<br>{gene}: {v:.3f}" for cid, v in zip(cell_ids, values, strict=False)])

    colorscale = _mpl_to_plotly_colorscale(cmap)
    fig = go.Figure()
    fig.add_trace(_scatter_trace_continuous(x, y, values, colorscale, vmin, vmax, gene, hover_text, dot_size, alpha))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    layer_label = layer or "raw"
    fig.update_layout(
        title=title or f"{gene} ({layer_label}) — global",
        xaxis_title="X (global px)",
        yaxis_title="Y (global px)",
        width=width,
        height=height,
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_local_dots_gene(
    spatioloji_obj,
    gene: str,
    layer: str | None = "log_normalized",
    fov_ids: list[str | int] | None = None,
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 4.0,
    alpha: float = 0.8,
    title: str | None = None,
    width: int = 700,
    height: int = 650,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive per-FOV gene expression scatter with FOV dropdown menu.

    Args:
        spatioloji_obj: spatioloji object.
        gene: Gene name.
        layer: Expression layer.
        fov_ids: FOVs to include (all if None).
        (other args): see :func:`iplot_global_dots_gene`.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_local_dots_gene(sp, gene="EPCAM", fov_ids=["1", "2"])
    """
    _require_plotly()

    meta = spatioloji_obj.cell_meta.reset_index()
    fov_col = spatioloji_obj.config.fov_id_col
    for col in ("CenterX_local_px", "CenterY_local_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    values = _get_expr_vector(spatioloji_obj, gene, layer)
    all_fovs = sorted(meta[fov_col].astype(str).unique().tolist())
    if fov_ids is None:
        fov_ids = all_fovs
    fov_ids = [str(f) for f in fov_ids]

    colorscale = _mpl_to_plotly_colorscale(cmap)
    vmin_eff = float(values.min()) if vmin is None else vmin
    vmax_eff = float(values.max()) if vmax is None else vmax
    cell_ids = spatioloji_obj.cell_index.astype(str)
    hover_all = np.array([f"Cell: {cid}<br>{gene}: {v:.3f}" for cid, v in zip(cell_ids, values, strict=False)])

    fig = go.Figure()
    first_fov = fov_ids[0]
    fov_trace_counts: dict = {}

    for fov_id in fov_ids:
        fov_mask = meta[fov_col].astype(str) == fov_id
        fov_meta = meta[fov_mask]
        x = fov_meta["CenterX_local_px"].values
        y = fov_meta["CenterY_local_px"].values
        fov_vals = values[fov_mask.values]
        fov_hover = hover_all[fov_mask.values]

        trace = _scatter_trace_continuous(
            x, y, fov_vals, colorscale, vmin_eff, vmax_eff, gene, fov_hover, dot_size, alpha
        )
        trace.visible = fov_id == first_fov
        trace.marker.colorbar.showticklabels = fov_id == first_fov
        fig.add_trace(trace)
        fov_trace_counts[fov_id] = 1

    total = len(fig.data)
    buttons = []
    for idx, fov_id in enumerate(fov_ids):
        vis = [False] * total
        vis[idx] = True
        buttons.append(
            dict(label=f"FOV {fov_id}", method="update", args=[{"visible": vis}, {"title": f"{gene} — FOV {fov_id}"}])
        )

    layer_label = layer or "raw"
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or f"{gene} ({layer_label}) — FOV {first_fov}",
        xaxis_title="X (local px)",
        yaxis_title="Y (local px)",
        width=width,
        height=height,
        template="simple_white",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0.01, xanchor="left", y=1.06, yanchor="top")
        ],
    )
    return _ifinalize(fig, save_path, show)


# =============================================================================
# SECTION 5 — SPATIAL POLYGON PLOTS
# =============================================================================


def iplot_global_polygon(
    spatioloji_obj,
    feature: str,
    feature_df=None,
    feature_column: str | None = None,
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    poly_alpha: float = 0.85,
    n_bins: int = 30,
    title: str | None = None,
    width: int = 850,
    height: int = 800,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive global polygon map coloured by any feature.

    Categorical: one trace per category with grouped polygon rings.
    Continuous: binned into ``n_bins`` colour steps for performance.

    Args:
        spatioloji_obj: spatioloji object with polygon data.
        feature: Feature column in ``cell_meta``, or ``"metadata"``.
        feature_df: Optional external feature DataFrame.
        feature_column: Column when ``feature="metadata"``.
        palette: Colour spec for categorical data.
        cmap: Matplotlib colormap for continuous data.
        vmin, vmax: Colour-scale limits (continuous only).
        poly_alpha: Polygon fill opacity.
        n_bins: Number of colour bins for continuous data.
        title: Plot title.
        width, height: Figure dimensions.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_global_polygon(sp, feature="cell_type")
        >>> sj.visualization.iplot_global_polygon(sp, feature="total_counts", cmap="plasma")
    """
    _require_plotly()

    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    gdf = spatioloji_obj.to_geopandas(coord_type="global", include_metadata=False)
    feat_series, feat_col = _resolve_feature(spatioloji_obj, feature, feature_df, feature_column)
    gdf[feat_col] = feat_series.values

    is_cat = not pd.api.types.is_numeric_dtype(feat_series)
    fig = go.Figure()

    if is_cat:
        cats = sorted(feat_series.dropna().unique().tolist(), key=str)
        color_dict = _cat_color_map(cats, palette)
        for trace in _polygon_traces_categorical(gdf, feat_col, color_dict, poly_alpha):
            fig.add_trace(trace)
    else:
        for trace in _polygon_traces_continuous(gdf, feat_col, cmap, vmin, vmax, poly_alpha, n_bins):
            fig.add_trace(trace)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or feat_col,
        xaxis_title="X (global px)",
        yaxis_title="Y (global px)",
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_local_polygon(
    spatioloji_obj,
    feature: str,
    fov_ids: list[str | int] | None = None,
    feature_df=None,
    feature_column: str | None = None,
    *,
    palette: str | list | dict | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    poly_alpha: float = 0.85,
    n_bins: int = 30,
    title: str | None = None,
    width: int = 750,
    height: int = 700,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive per-FOV polygon map with FOV dropdown menu.

    Args:
        spatioloji_obj: spatioloji object with polygon data.
        feature: Feature column in ``cell_meta``, or ``"metadata"``.
        fov_ids: FOVs to include (all if None).
        (other args): see :func:`iplot_global_polygon`.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_local_polygon(sp, feature="cell_type", fov_ids=["1","2"])
    """
    _require_plotly()

    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    gdf = spatioloji_obj.to_geopandas(coord_type="local", include_metadata=False)
    fov_col = spatioloji_obj.config.fov_id_col
    gdf[fov_col] = spatioloji_obj.cell_meta[fov_col].astype(str).values

    feat_series, feat_col = _resolve_feature(spatioloji_obj, feature, feature_df, feature_column)
    gdf[feat_col] = feat_series.values

    all_fovs = sorted(gdf[fov_col].unique().tolist())
    if fov_ids is None:
        fov_ids = all_fovs
    fov_ids = [str(f) for f in fov_ids]

    is_cat = not pd.api.types.is_numeric_dtype(feat_series)
    cats = sorted(feat_series.dropna().unique().tolist(), key=str) if is_cat else None
    color_dict = _cat_color_map(cats, palette) if is_cat else None

    fig = go.Figure()
    first_fov = fov_ids[0]
    fov_trace_counts: dict = {}

    for fov_id in fov_ids:
        fov_gdf = gdf[gdf[fov_col] == fov_id]
        if fov_gdf.empty:
            fov_trace_counts[fov_id] = 0
            continue
        n_before = len(fig.data)
        if is_cat:
            traces = _polygon_traces_categorical(fov_gdf, feat_col, color_dict, poly_alpha)
        else:
            traces = _polygon_traces_continuous(fov_gdf, feat_col, cmap, vmin, vmax, poly_alpha, n_bins)
        for trace in traces:
            trace.visible = fov_id == first_fov
            trace.showlegend = (fov_id == first_fov) and is_cat
            if is_cat:
                trace.legendgroup = str(trace.name)
            fig.add_trace(trace)
        fov_trace_counts[fov_id] = len(fig.data) - n_before

    total = len(fig.data)
    buttons = []
    offset = 0
    for fov_id in fov_ids:
        n = fov_trace_counts.get(fov_id, 0)
        vis = [False] * total
        for i in range(offset, offset + n):
            vis[i] = True
        buttons.append(
            dict(
                label=f"FOV {fov_id}", method="update", args=[{"visible": vis}, {"title": f"{feat_col} — FOV {fov_id}"}]
            )
        )
        offset += n

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or f"{feat_col} — FOV {first_fov}",
        xaxis_title="X (local px)",
        yaxis_title="Y (local px)",
        width=width,
        height=height,
        legend=dict(itemsizing="constant"),
        template="simple_white",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0.01, xanchor="left", y=1.06, yanchor="top")
        ],
    )
    return _ifinalize(fig, save_path, show)


def iplot_global_polygon_gene(
    spatioloji_obj,
    gene: str,
    layer: str | None = "log_normalized",
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    poly_alpha: float = 0.85,
    n_bins: int = 30,
    title: str | None = None,
    width: int = 850,
    height: int = 800,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive global polygon map coloured by gene expression.

    Args:
        spatioloji_obj: spatioloji object with polygon data.
        gene: Gene name.
        layer: Expression layer (None = raw counts).
        cmap: Matplotlib colormap name.
        vmin, vmax: Colour-scale limits.
        poly_alpha: Polygon fill opacity.
        n_bins: Number of colour bins.
        title: Plot title (auto-generated if None).
        width, height: Figure dimensions.
        save_path: Path to save.
        show: Display figure if True.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_global_polygon_gene(sp, gene="EPCAM")
    """
    _require_plotly()

    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    values = _get_expr_vector(spatioloji_obj, gene, layer)
    gdf = spatioloji_obj.to_geopandas(coord_type="global", include_metadata=False)
    gdf[gene] = values

    fig = go.Figure()
    for trace in _polygon_traces_continuous(gdf, gene, cmap, vmin, vmax, poly_alpha, n_bins):
        fig.add_trace(trace)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    layer_label = layer or "raw"
    fig.update_layout(
        title=title or f"{gene} ({layer_label}) — global",
        xaxis_title="X (global px)",
        yaxis_title="Y (global px)",
        width=width,
        height=height,
        template="simple_white",
    )
    return _ifinalize(fig, save_path, show)


def iplot_local_polygon_gene(
    spatioloji_obj,
    gene: str,
    layer: str | None = "log_normalized",
    fov_ids: list[str | int] | None = None,
    *,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    poly_alpha: float = 0.85,
    n_bins: int = 30,
    title: str | None = None,
    width: int = 750,
    height: int = 700,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Interactive per-FOV gene expression polygon map with FOV dropdown menu.

    Args:
        spatioloji_obj: spatioloji object with polygon data.
        gene: Gene name.
        layer: Expression layer.
        fov_ids: FOVs to include (all if None).
        (other args): see :func:`iplot_global_polygon_gene`.

    Returns:
        plotly.graph_objects.Figure

    Example:
        >>> sj.visualization.iplot_local_polygon_gene(sp, gene="EPCAM", fov_ids=["1","2"])
    """
    _require_plotly()

    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    values = _get_expr_vector(spatioloji_obj, gene, layer)
    gdf = spatioloji_obj.to_geopandas(coord_type="local", include_metadata=False)
    fov_col = spatioloji_obj.config.fov_id_col
    gdf[fov_col] = spatioloji_obj.cell_meta[fov_col].astype(str).values
    gdf[gene] = values

    all_fovs = sorted(gdf[fov_col].unique().tolist())
    if fov_ids is None:
        fov_ids = all_fovs
    fov_ids = [str(f) for f in fov_ids]

    vmin_eff = float(values.min()) if vmin is None else vmin
    vmax_eff = float(values.max()) if vmax is None else vmax

    fig = go.Figure()
    first_fov = fov_ids[0]
    fov_trace_counts: dict = {}

    for fov_id in fov_ids:
        fov_gdf = gdf[gdf[fov_col] == fov_id]
        if fov_gdf.empty:
            fov_trace_counts[fov_id] = 0
            continue
        n_before = len(fig.data)
        for trace in _polygon_traces_continuous(fov_gdf, gene, cmap, vmin_eff, vmax_eff, poly_alpha, n_bins):
            trace.visible = fov_id == first_fov
            fig.add_trace(trace)
        fov_trace_counts[fov_id] = len(fig.data) - n_before

    total = len(fig.data)
    buttons = []
    offset = 0
    for fov_id in fov_ids:
        n = fov_trace_counts.get(fov_id, 0)
        vis = [False] * total
        for i in range(offset, offset + n):
            vis[i] = True
        buttons.append(
            dict(label=f"FOV {fov_id}", method="update", args=[{"visible": vis}, {"title": f"{gene} — FOV {fov_id}"}])
        )
        offset += n

    layer_label = layer or "raw"
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title or f"{gene} ({layer_label}) — FOV {first_fov}",
        xaxis_title="X (local px)",
        yaxis_title="Y (local px)",
        width=width,
        height=height,
        template="simple_white",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=0.01, xanchor="left", y=1.06, yanchor="top")
        ],
    )
    return _ifinalize(fig, save_path, show)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Embedding plots
    "iplot_umap",
    "iplot_pca",
    "iplot_umap_grid",
    # Distribution plots
    "iplot_violin",
    "iplot_heatmap",
    "iplot_dotplot",
    # Spatial dot plots
    "iplot_global_dots",
    "iplot_local_dots",
    "iplot_global_dots_gene",
    "iplot_local_dots_gene",
    # Spatial polygon plots
    "iplot_global_polygon",
    "iplot_local_polygon",
    "iplot_global_polygon_gene",
    "iplot_local_polygon_gene",
]
