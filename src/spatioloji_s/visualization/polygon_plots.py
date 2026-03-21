"""
polygon_plots.py — Visualization for polygon-based spatial analysis.

Covers results produced by the ``spatial.polygon`` submodule:

    plot_polygon_graph                  — polygon fills + contact-edge overlay
    plot_contact_degree_distribution    — histogram / KDE of contact degree per type
    plot_polygon_neighborhood_enrichment — z-score heatmap (polygon contact enrichment)
    plot_polygon_neighborhood_composition — stacked-bar polygon neighbour composition
    plot_polygon_niches                 — polygon map coloured by niche label
    plot_boundary_cells_map             — polygon map highlighting boundary cells
    plot_free_boundary_map              — polygon map coloured by free-boundary fraction
    plot_contact_summary                — heatmap of contact frequency / length
    plot_density_map                    — polygon map coloured by local cell density
    plot_polygon_hotspot_map            — polygon Gi* hotspot / coldspot map
    plot_morphology_distribution        — violin / box of shape metrics per group
    plot_morphology_map                 — polygon spatial map coloured by one metric
    plot_morphology_class_map           — polygon map coloured by morphology class
    plot_morphology_correlation         — pairwise metric correlation heatmap
    plot_contact_permutation            — heatmap of contact z-scores from permutation test
    plot_boundary_enrichment            — horizontal bar chart of boundary-cell fractions
    plot_spatial_autocorrelation        — lollipop of autocorrelation per metric
    plot_morphology_association         — violin (group) or scatter (gene correlation)
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from ._spatial_helpers import (
    categorical_colors,
    clean_axes,
    continuous_colors,
    finalize_plot,
)

# ── Private polygon helper ─────────────────────────────────────────────────────


def _build_poly_collection(gdf, face_colors: list) -> PolyCollection:
    """Convert a GeoDataFrame to a matplotlib PolyCollection.

    Args:
        gdf: GeoDataFrame with a ``geometry`` column of Shapely Polygon objects.
        face_colors: List of RGBA colours, one per row of ``gdf``.

    Returns:
        A ``PolyCollection`` ready to be added to an axes.

    Raises:
        ImportError: If ``shapely`` is not installed.
    """
    try:
        from shapely.geometry import MultiPolygon
    except ImportError as exc:
        raise ImportError("shapely is required for polygon visualisation.") from exc

    verts = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            verts.append(np.zeros((3, 2)))
            continue
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        verts.append(np.array(geom.exterior.coords))

    return PolyCollection(verts, facecolors=face_colors, linewidths=0.2, edgecolors="none")


# ══════════════════════════════════════════════════════════════════════════════
# Polygon spatial graph
# ══════════════════════════════════════════════════════════════════════════════


def plot_polygon_graph(
    spatioloji_obj,
    graph,
    color_by: str | None = None,
    coord_type: str = "global",
    palette: str | list | dict = "tab20",
    edge_alpha: float = 0.4,
    edge_color: str = "#555555",
    edge_width: float = 0.6,
    poly_alpha: float = 0.7,
    max_edges: int = 30_000,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon fills with contact-graph edges overlaid.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        graph: ``PolygonSpatialGraph`` returned by any ``spatial.polygon.graph``
            function (``build_contact_graph``, ``build_buffer_graph``,
            ``build_knn_graph``).
        color_by: ``cell_meta`` column to colour polygon fills by. Categorical
            columns produce a legend; continuous columns produce a colorbar.
        coord_type: ``'global'`` (default) or ``'local'``.
        palette: Colour palette for categorical data, or colormap name for
            continuous data.
        edge_alpha: Opacity of contact edges.
        edge_color: Colour of contact edges.
        edge_width: Line width of contact edges.
        poly_alpha: Opacity of polygon fills.
        max_edges: Cap on drawn edges to avoid slow rendering of dense graphs.
        figsize: Figure size.
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> g = sj.spatial.polygon.graph.build_contact_graph(sp)
        >>> sj.visualization.plot_polygon_graph(sp, g, color_by="cell_type")
    """
    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=True)
    centroids = gdf.geometry.centroid
    cx_arr = centroids.x.values
    cy_arr = centroids.y.values
    cell_ids = gdf.index.tolist()
    id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    # ── edge segments from graph adjacency ────────────────────────────────────
    g_ids = list(getattr(graph, "cell_index", cell_ids))
    adj = graph.adjacency.tocoo()
    rows, cols_idx = adj.row, adj.col
    mask = rows < cols_idx
    rows, cols_idx = rows[mask], cols_idx[mask]

    if len(rows) > max_edges:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(rows), size=max_edges, replace=False)
        rows, cols_idx = rows[sel], cols_idx[sel]

    def _centroid(idx_in_graph: int):
        cid = g_ids[idx_in_graph]
        sp_idx = id_to_idx.get(cid)
        return (cx_arr[sp_idx], cy_arr[sp_idx]) if sp_idx is not None else None

    segs = [
        [a, b]
        for r, c in zip(rows, cols_idx, strict=False)
        if (a := _centroid(r)) is not None and (b := _centroid(c)) is not None
    ]

    # ── polygon fills ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if color_by is not None and color_by in gdf.columns:
        vals = gdf[color_by]
        if vals.dtype.kind in ("O", "U") or str(vals.dtype) == "category":
            colors, _, handles = categorical_colors(vals, palette)
            pc = _build_poly_collection(gdf, colors)
            ax.add_collection(pc)
            ax.legend(
                handles=handles,
                title=color_by,
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=7,
                title_fontsize=8,
                frameon=False,
            )
        else:
            cmap_name = palette if isinstance(palette, str) else "viridis"
            colors, norm, mappable = continuous_colors(vals, cmap_name, None, None, None)
            pc = _build_poly_collection(gdf, colors)
            ax.add_collection(pc)
            plt.colorbar(mappable, ax=ax, shrink=0.6, label=color_by)
    else:
        pc = _build_poly_collection(gdf, ["#cde3f5"] * len(gdf))
        ax.add_collection(pc)

    pc.set_alpha(poly_alpha)

    # ── edge overlay ───────────────────────────────────────────────────────────
    if segs:
        lc = LineCollection(segs, linewidths=edge_width, colors=edge_color, alpha=edge_alpha, zorder=2)
        ax.add_collection(lc)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title or f"Polygon graph ({graph.method}, {graph.n_edges:,} edges)")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Morphology distribution
# ══════════════════════════════════════════════════════════════════════════════


def plot_morphology_distribution(
    spatioloji_obj,
    metrics: str | list[str] | None = None,
    group_col: str | None = None,
    plot_type: Literal["violin", "box", "strip"] = "violin",
    palette: str | list | dict = "tab20",
    figsize: tuple[float, float] | None = None,
    title: str = "Cell morphology distribution",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Violin / box / strip plot of polygon morphology metrics per cell-type group.

    Morphology metrics must already be stored in ``cell_meta`` (i.e. run
    ``compute_morphology(store=True)`` first).

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        metrics: One or more metric names. Accepts either the full column name
            (``'morph_circularity'``) or the short name without prefix
            (``'circularity'``). Defaults to all ``morph_*`` columns.
        group_col: ``cell_meta`` column for grouping (e.g. ``'cell_type'``).
        plot_type: ``'violin'``, ``'box'``, or ``'strip'``.
        palette: Colour palette for groups.
        figsize: Figure size. Auto-computed from number of metrics if ``None``.
        title: Figure suptitle.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.spatial.polygon.morphology.compute_morphology(sp, store=True)
        >>> sj.visualization.plot_morphology_distribution(sp, group_col="cell_type")
    """
    meta = spatioloji_obj.cell_meta

    if metrics is None:
        metric_cols = [c for c in meta.columns if c.startswith("morph_")]
        if not metric_cols:
            raise ValueError("No 'morph_*' columns found. Run compute_morphology(store=True) first.")
    else:
        raw = [metrics] if isinstance(metrics, str) else list(metrics)
        metric_cols = []
        for m in raw:
            if m in meta.columns:
                metric_cols.append(m)
            elif f"morph_{m}" in meta.columns:
                metric_cols.append(f"morph_{m}")
            else:
                raise ValueError(
                    f"Metric '{m}' not found in cell_meta. "
                    f"Available morph columns: {[c for c in meta.columns if c.startswith('morph_')]}"
                )

    n_metrics = len(metric_cols)
    figsize = figsize or (max(5, n_metrics * 3), 5)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, col in zip(axes, metric_cols, strict=False):
        plot_data = meta[[col]].copy()
        if group_col is not None and group_col in meta.columns:
            plot_data[group_col] = meta[group_col]
            x_var, y_var = group_col, col
        else:
            plot_data["_all"] = "all cells"
            x_var, y_var = "_all", col

        if plot_type == "violin":
            sns.violinplot(data=plot_data, x=x_var, y=y_var, palette=palette, ax=ax, cut=0, linewidth=0.8)
        elif plot_type == "box":
            sns.boxplot(data=plot_data, x=x_var, y=y_var, palette=palette, ax=ax, linewidth=0.8)
        else:
            sns.stripplot(data=plot_data, x=x_var, y=y_var, palette=palette, ax=ax, size=2, jitter=True, alpha=0.4)

        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        clean_axes(ax)

    fig.suptitle(title, y=1.02, fontsize=12)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Morphology spatial map
# ══════════════════════════════════════════════════════════════════════════════


def plot_morphology_map(
    spatioloji_obj,
    metric: str,
    coord_type: str = "global",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map coloured by a single morphology metric.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        metric: Morphology column to colour by. Accepts the full column name
            (``'morph_circularity'``) or the short name (``'circularity'``).
        coord_type: ``'global'`` or ``'local'``.
        cmap: Continuous colormap name.
        vmin: Lower colour limit. Uses data min if ``None``.
        vmax: Upper colour limit. Uses data max if ``None``.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.spatial.polygon.morphology.compute_morphology(sp, store=True)
        >>> sj.visualization.plot_morphology_map(sp, "circularity")
    """
    meta = spatioloji_obj.cell_meta
    col = metric if metric in meta.columns else f"morph_{metric}"
    if col not in meta.columns:
        raise ValueError(
            f"Metric '{metric}' not found. "
            f"Available morph columns: {[c for c in meta.columns if c.startswith('morph_')]}"
        )

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=True)
    vals = gdf[col] if col in gdf.columns else meta.reindex(gdf.index)[col]

    vmin = float(vals.min()) if vmin is None else vmin
    vmax = float(vals.max()) if vmax is None else vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm_obj = plt.get_cmap(cmap)
    colors = [cm_obj(norm(v)) if pd.notna(v) else (0.85, 0.85, 0.85, 1.0) for v in vals]

    pc = _build_poly_collection(gdf, colors)
    pc.set_alpha(poly_alpha)

    mappable = ScalarMappable(norm=norm, cmap=cm_obj)
    mappable.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.colorbar(mappable, ax=ax, shrink=0.6, label=col)
    ax.set_title(title or f"Morphology map — {col}")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Contact permutation heatmap
# ══════════════════════════════════════════════════════════════════════════════


def plot_contact_permutation(
    contact_result: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    significance_threshold: float = 0.05,
    annotate: bool = True,
    title: str = "Contact permutation test (z-score)",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Symmetric heatmap of cell-type contact z-scores from a permutation test.

    Args:
        contact_result: DataFrame returned by
            ``spatial.polygon.statistics.contact_permutation_test()``.
            Must contain columns ``type_a``, ``type_b``, ``z_score``.
            Optionally ``p_adjusted`` or ``p_value`` for significance markers.
        figsize: Figure size. Auto-computed from number of cell types if ``None``.
        cmap: Diverging colormap.
        vmax: Symmetric colour range ``[-vmax, vmax]``. Auto if ``None``.
        significance_threshold: P-value threshold; pairs below get a ``*`` marker.
        annotate: If ``True``, overlay z-score values and significance markers.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.statistics.contact_permutation_test(sp, g, "cell_type")
        >>> sj.visualization.plot_contact_permutation(result)
    """
    all_types = sorted(set(contact_result["type_a"].tolist() + contact_result["type_b"].tolist()))
    n = len(all_types)
    idx = {t: i for i, t in enumerate(all_types)}
    p_col = "p_adjusted" if "p_adjusted" in contact_result.columns else "p_value"

    z_mat = np.zeros((n, n))
    sig_mat = np.zeros((n, n), dtype=bool)
    for _, row in contact_result.iterrows():
        i, j = idx[row["type_a"]], idx[row["type_b"]]
        z_mat[i, j] = z_mat[j, i] = row["z_score"]
        if p_col in contact_result.columns:
            sig_mat[i, j] = sig_mat[j, i] = row[p_col] < significance_threshold

    vmax = vmax or float(np.abs(z_mat).max())
    figsize = figsize or (max(5, n * 0.65 + 1.5), max(4, n * 0.6 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(z_mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Z-score")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_types, fontsize=8)
    ax.set_xlabel("Cell type B")
    ax.set_ylabel("Cell type A")
    ax.set_title(title)

    if annotate:
        for i in range(n):
            for j in range(n):
                z = z_mat[i, j]
                sig = "*" if sig_mat[i, j] else ""
                ax.text(
                    j,
                    i,
                    f"{z:.1f}{sig}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(z) > vmax * 0.6 else "black",
                )

    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary enrichment
# ══════════════════════════════════════════════════════════════════════════════


def plot_boundary_enrichment(
    boundary_result: pd.DataFrame,
    sort_by: Literal["fold_change", "z_score", "cell_type"] = "fold_change",
    palette: str = "coolwarm",
    figsize: tuple[float, float] | None = None,
    title: str = "Boundary-cell enrichment per type",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Horizontal bar chart of boundary-cell fold change per cell type.

    Bars coloured by direction: warm colours for enriched types
    (fold_change > 1), cool colours for depleted types (fold_change < 1).

    Args:
        boundary_result: DataFrame returned by
            ``spatial.polygon.statistics.boundary_enrichment_test()``.
            Must contain ``cell_type`` and ``fold_change``.
            Optionally ``significant`` (bool) and ``z_score``.
        sort_by: Bar sort order: ``'fold_change'``, ``'z_score'``, or
            ``'cell_type'`` (alphabetical).
        palette: Diverging colormap name.
        figsize: Figure size. Auto-computed from number of cell types if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.statistics.boundary_enrichment_test(sp, g, "cell_type")
        >>> sj.visualization.plot_boundary_enrichment(result)
    """
    df = boundary_result.copy()
    if sort_by == "fold_change":
        df = df.sort_values("fold_change", ascending=True)
    elif sort_by == "z_score" and "z_score" in df.columns:
        df = df.sort_values("z_score", ascending=True)
    else:
        df = df.sort_values("cell_type")

    n = len(df)
    figsize = figsize or (7, max(3, n * 0.4))
    fc_vals = df["fold_change"].values
    fc_min, fc_max = float(fc_vals.min()), float(fc_vals.max())
    # guard against degenerate case where all values equal 1
    if fc_min == fc_max:
        fc_min, fc_max = fc_min - 0.1, fc_max + 0.1
    norm = TwoSlopeNorm(vmin=fc_min, vcenter=1.0, vmax=fc_max)
    cm_obj = plt.get_cmap(palette)
    bar_colors = [cm_obj(norm(v)) for v in fc_vals]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(n), fc_vals - 1.0, left=1.0, color=bar_colors, height=0.7)
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--")

    if "significant" in df.columns:
        x_label = fc_vals.max() + 0.02
        for i, (_, row) in enumerate(df.iterrows()):
            if row.get("significant", False):
                ax.text(x_label, i, "*", va="center", fontsize=9)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df["cell_type"].tolist(), fontsize=8)
    ax.set_xlabel("Fold change (boundary fraction / expected)")
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Contact degree distribution
# ══════════════════════════════════════════════════════════════════════════════


def plot_contact_degree_distribution(
    graph,
    spatioloji_obj=None,
    group_col: str | None = None,
    palette: str | list | dict = "tab20",
    bins: int = 20,
    figsize: tuple[float, float] = (6, 4),
    title: str = "Polygon contact degree distribution",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Histogram or per-group KDE of polygon contact degrees.

    Args:
        graph: A ``PolygonSpatialGraph`` with a ``get_degree()`` method that
            returns a ``pd.Series`` of contact counts per cell.
        spatioloji_obj: Required when ``group_col`` is supplied.
        group_col: ``cell_meta`` column to split degrees by (e.g.
            ``'cell_type'``). When supplied, overlaid KDE curves are drawn.
        palette: Colour palette for groups.
        bins: Number of histogram bins (used only when ``group_col`` is
            ``None``).
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> g = sj.spatial.polygon.graph.build_contact_graph(sp)
        >>> sj.visualization.plot_contact_degree_distribution(g, sp, group_col="cell_type")
    """
    degrees: pd.Series = graph.get_degree()

    fig, ax = plt.subplots(figsize=figsize)

    if group_col is not None and spatioloji_obj is not None:
        groups = spatioloji_obj.cell_meta.reindex(degrees.index)[group_col]
        _, color_dict, _ = categorical_colors(groups.dropna(), palette)
        for grp, sub in degrees.groupby(groups):
            sns.kdeplot(sub.values, ax=ax, fill=True, alpha=0.4, color=color_dict.get(grp, "grey"), label=str(grp))
        ax.set_ylabel("Density")
        ax.legend(
            title=group_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8, frameon=False
        )
    else:
        ax.hist(degrees.values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Count")

    ax.axvline(
        graph.mean_degree, color="#d7191c", linestyle="--", linewidth=1.2, label=f"Mean = {graph.mean_degree:.1f}"
    )
    ax.set_xlabel("Contact degree")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Polygon neighbourhood enrichment
# ══════════════════════════════════════════════════════════════════════════════


def plot_polygon_neighborhood_enrichment(
    enrichment_result: dict,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    significance_threshold: float = 0.05,
    annotate: bool = True,
    title: str = "Polygon neighbourhood enrichment (z-score)",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of polygon contact enrichment z-scores (cell-type × cell-type).

    Args:
        enrichment_result: Dict returned by
            ``spatial.polygon.neighborhoods.neighborhood_enrichment()``.
            Must contain ``'z_scores'`` (DataFrame n_types × n_types).
            Optionally ``'p_values'`` for significance markers.
        figsize: Figure size.
        cmap: Diverging colormap.
        vmax: Symmetric colour range ``[-vmax, vmax]``. Auto if ``None``.
        significance_threshold: P-value threshold; pairs below get a ``*``.
        annotate: If ``True``, overlay z-score values and ``*`` markers.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.neighborhoods.neighborhood_enrichment(sp, g, "cell_type")
        >>> sj.visualization.plot_polygon_neighborhood_enrichment(result)
    """
    zscore: pd.DataFrame | None = enrichment_result.get("z_scores") or enrichment_result.get("zscore")
    if zscore is None:
        raise ValueError("enrichment_result must contain key 'z_scores'.")
    pvalue: pd.DataFrame | None = enrichment_result.get("p_values") or enrichment_result.get("pvalue")

    vmax = vmax or float(np.abs(zscore.values).max())
    n = zscore.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(zscore.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Z-score")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(zscore.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(zscore.index, fontsize=8)
    ax.set_xlabel("Receiver cell type")
    ax.set_ylabel("Sender cell type")
    ax.set_title(title)

    if annotate:
        for i in range(n):
            for j in range(n):
                z = zscore.iloc[i, j]
                sig = "*" if (pvalue is not None and pvalue.iloc[i, j] < significance_threshold) else ""
                ax.text(
                    j,
                    i,
                    f"{z:.1f}{sig}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(z) > vmax * 0.6 else "black",
                )

    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Polygon neighbourhood composition
# ══════════════════════════════════════════════════════════════════════════════


def plot_polygon_neighborhood_composition(
    composition_df: pd.DataFrame,
    spatioloji_obj=None,
    group_col: str | None = None,
    palette: str | list | dict = "tab20",
    top_n_types: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Polygon neighbourhood composition",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Stacked-bar chart of mean polygon neighbour composition per group.

    Args:
        composition_df: DataFrame ``(n_cells × n_types)`` returned by
            ``spatial.polygon.neighborhoods.neighborhood_composition()``.
        spatioloji_obj: Required when ``group_col`` is supplied.
        group_col: ``cell_meta`` column to group cells by.
        palette: Colour palette for cell-type bars.
        top_n_types: Keep only the N most abundant types; others merged as
            ``'Other'``.
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> comp = sj.spatial.polygon.neighborhoods.neighborhood_composition(sp, g, "cell_type")
        >>> sj.visualization.plot_polygon_neighborhood_composition(comp, sp, group_col="cell_type")
    """
    if group_col is not None and spatioloji_obj is not None:
        groups = spatioloji_obj.cell_meta.reindex(composition_df.index)[group_col]
        mean_comp = composition_df.groupby(groups).mean()
    else:
        mean_comp = composition_df.copy()

    if top_n_types is not None and mean_comp.shape[1] > top_n_types:
        top_cols = mean_comp.mean().nlargest(top_n_types).index.tolist()
        other = mean_comp.drop(columns=top_cols).sum(axis=1)
        mean_comp = mean_comp[top_cols].copy()
        mean_comp["Other"] = other

    n_groups = len(mean_comp)
    figsize = figsize or (max(6, n_groups * 0.5 + 2), 5)
    types = mean_comp.columns.tolist()
    _, color_dict, _ = categorical_colors(pd.Series(types), palette)

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(n_groups)
    for t in types:
        vals = mean_comp[t].values
        ax.bar(range(n_groups), vals, bottom=bottom, color=color_dict.get(t, "grey"), label=t, width=0.8)
        bottom += vals

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(mean_comp.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean neighbour fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(
        title="Cell type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8, frameon=False
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Polygon niches
# ══════════════════════════════════════════════════════════════════════════════


def plot_polygon_niches(
    spatioloji_obj,
    niche_labels: pd.Series,
    coord_type: str = "global",
    palette: str | list | dict = "tab10",
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str = "Polygon spatial niches",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map coloured by niche label.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        niche_labels: ``pd.Series`` indexed by cell IDs with niche label
            values, as returned by ``niche_identification()``.
        coord_type: ``'global'`` or ``'local'``.
        palette: Colour palette for niches.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> labels = sj.spatial.polygon.neighborhoods.niche_identification(sp, g, "cell_type")
        >>> sj.visualization.plot_polygon_niches(sp, labels)
    """
    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    labels = niche_labels.reindex(gdf.index)
    colors, _, handles = categorical_colors(labels, palette)
    pc = _build_poly_collection(gdf, colors)
    pc.set_alpha(poly_alpha)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(
        handles=handles,
        title="Niche",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary cells map
# ══════════════════════════════════════════════════════════════════════════════


def plot_boundary_cells_map(
    spatioloji_obj,
    coord_type: str = "global",
    boundary_col: str = "is_boundary",
    score_col: str = "boundary_score",
    mode: Literal["categorical", "score"] = "categorical",
    boundary_color: str = "#d7191c",
    interior_color: str = "#e0e0e0",
    cmap: str = "YlOrRd",
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str = "Boundary cells",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map highlighting boundary cells.

    Requires ``boundary_cells()`` to have been run with ``store=True``, which
    writes ``is_boundary`` and ``boundary_score`` into ``cell_meta``.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        coord_type: ``'global'`` or ``'local'``.
        boundary_col: ``cell_meta`` column with boolean boundary flag.
        score_col: ``cell_meta`` column with continuous boundary score (0–1).
        mode: ``'categorical'`` — red boundary / grey interior cells;
            ``'score'`` — continuous colormap of boundary score.
        boundary_color: Polygon colour for boundary cells (``mode='categorical'``).
        interior_color: Polygon colour for interior cells (``mode='categorical'``).
        cmap: Colormap for ``mode='score'``.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.spatial.polygon.neighborhoods.boundary_cells(sp, g, "cell_type", store=True)
        >>> sj.visualization.plot_boundary_cells_map(sp)
    """
    meta = spatioloji_obj.cell_meta
    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)

    if mode == "categorical":
        if boundary_col not in meta.columns:
            raise ValueError(f"'{boundary_col}' not found in cell_meta. Run boundary_cells(store=True) first.")
        is_boundary = meta.reindex(gdf.index)[boundary_col].fillna(False).astype(bool)
        colors = [boundary_color if b else interior_color for b in is_boundary]
        pc = _build_poly_collection(gdf, colors)
        pc.set_alpha(poly_alpha)
        import matplotlib.patches as mpatches

        handles = [
            mpatches.Patch(color=boundary_color, label="Boundary"),
            mpatches.Patch(color=interior_color, label="Interior"),
        ]
        fig, ax = plt.subplots(figsize=figsize)
        ax.add_collection(pc)
        ax.legend(handles=handles, fontsize=7, frameon=False)
    else:
        if score_col not in meta.columns:
            raise ValueError(f"'{score_col}' not found in cell_meta. Run boundary_cells(store=True) first.")
        vals = meta.reindex(gdf.index)[score_col]
        colors_list, norm, mappable = continuous_colors(vals, cmap, 0.0, 1.0, None)
        pc = _build_poly_collection(gdf, colors_list)
        pc.set_alpha(poly_alpha)
        fig, ax = plt.subplots(figsize=figsize)
        ax.add_collection(pc)
        plt.colorbar(mappable, ax=ax, shrink=0.6, label="Boundary score")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Free-boundary fraction map
# ══════════════════════════════════════════════════════════════════════════════


def plot_free_boundary_map(
    spatioloji_obj,
    free_boundary_series: pd.Series | None = None,
    coord_type: str = "global",
    cmap: str = "plasma",
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str = "Free-boundary fraction",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map coloured by free-boundary fraction (0 = fully contacted).

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        free_boundary_series: ``pd.Series`` indexed by cell IDs with values in
            [0, 1], returned by
            ``spatial.polygon.boundaries.free_boundary_fraction()``. If
            ``None``, looks for ``'free_boundary_fraction'`` in
            ``cell_meta``.
        coord_type: ``'global'`` or ``'local'``.
        cmap: Continuous colormap.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> fb = sj.spatial.polygon.boundaries.free_boundary_fraction(sp, g)
        >>> sj.visualization.plot_free_boundary_map(sp, fb)
    """
    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)

    if free_boundary_series is not None:
        vals = free_boundary_series.reindex(gdf.index)
    elif "free_boundary_fraction" in spatioloji_obj.cell_meta.columns:
        vals = spatioloji_obj.cell_meta.reindex(gdf.index)["free_boundary_fraction"]
    else:
        raise ValueError("free_boundary_series is None and 'free_boundary_fraction' not found in cell_meta.")

    colors, norm, mappable = continuous_colors(vals, cmap, 0.0, 1.0, None)
    pc = _build_poly_collection(gdf, colors)
    pc.set_alpha(poly_alpha)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.colorbar(mappable, ax=ax, shrink=0.6, label="Free-boundary fraction")
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Contact summary heatmap
# ══════════════════════════════════════════════════════════════════════════════


def plot_contact_summary(
    contact_summary_df: pd.DataFrame,
    value_col: str = "n_contacts",
    cmap: str = "YlOrRd",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of contact frequency or mean contact length between cell-type pairs.

    Args:
        contact_summary_df: DataFrame returned by
            ``spatial.polygon.boundaries.contact_summary()``. Must contain
            ``type_a``, ``type_b``, and a value column.
        value_col: Column to visualise. One of ``'n_contacts'``,
            ``'total_length'``, ``'mean_length'``, ``'median_length'``.
        cmap: Sequential colormap.
        annotate: If ``True``, overlay numeric values.
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title. Defaults to ``f"Contact summary — {value_col}"``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> cs = sj.spatial.polygon.boundaries.contact_summary(sp, g, "cell_type")
        >>> sj.visualization.plot_contact_summary(cs, value_col="mean_length")
    """
    if value_col not in contact_summary_df.columns:
        raise ValueError(f"value_col '{value_col}' not found. Available: {contact_summary_df.columns.tolist()}")

    all_types = sorted(set(contact_summary_df["type_a"].tolist() + contact_summary_df["type_b"].tolist()))
    n = len(all_types)
    idx = {t: i for i, t in enumerate(all_types)}
    mat = np.zeros((n, n))

    for _, row in contact_summary_df.iterrows():
        i, j = idx[row["type_a"]], idx[row["type_b"]]
        mat[i, j] = mat[j, i] = row[value_col]

    figsize = figsize or (max(5, n * 0.65 + 1.5), max(4, n * 0.6 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label=value_col)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_types, fontsize=8)
    ax.set_xlabel("Cell type B")
    ax.set_ylabel("Cell type A")
    ax.set_title(title or f"Contact summary — {value_col}")

    if annotate:
        fmt = "{:.0f}" if value_col == "n_contacts" else "{:.1f}"
        vmax = mat.max()
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    fmt.format(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] > vmax * 0.6 else "black",
                )

    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Cell density map
# ══════════════════════════════════════════════════════════════════════════════


def plot_density_map(
    spatioloji_obj,
    density_df: pd.DataFrame,
    metric: str = "local_density",
    coord_type: str = "global",
    cmap: str = "YlOrRd",
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map coloured by local cell density.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        density_df: DataFrame returned by
            ``spatial.polygon.patterns.cell_density_map()``. Columns:
            ``area``, ``cell_density``, ``local_density``.
        metric: Column to colour by. One of ``'area'``, ``'cell_density'``,
            ``'local_density'``.
        coord_type: ``'global'`` or ``'local'``.
        cmap: Continuous colormap.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title. Defaults to ``f"Cell density map — {metric}"``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> density = sj.spatial.polygon.patterns.cell_density_map(sp, g)
        >>> sj.visualization.plot_density_map(sp, density)
    """
    if metric not in density_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {density_df.columns.tolist()}")

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    vals = density_df[metric].reindex(gdf.index)
    colors, _, mappable = continuous_colors(vals, cmap, None, None, None)
    pc = _build_poly_collection(gdf, colors)
    pc.set_alpha(poly_alpha)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.colorbar(mappable, ax=ax, shrink=0.6, label=metric)
    ax.set_title(title or f"Cell density map — {metric}")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Polygon hotspot map (Gi*)
# ══════════════════════════════════════════════════════════════════════════════


def plot_polygon_hotspot_map(
    spatioloji_obj,
    hotspot_df: pd.DataFrame,
    coord_type: str = "global",
    mode: Literal["spot_type", "zscore"] = "spot_type",
    zscore_col: str = "z_score",
    cmap: str = "RdBu_r",
    palette: dict | None = None,
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str = "Polygon Gi* hotspot map",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map of Getis-Ord Gi* hotspot / coldspot results.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        hotspot_df: DataFrame returned by
            ``spatial.polygon.patterns.hotspot_detection()``. Columns:
            ``gi_star``, ``z_score``, ``p_value``.
        coord_type: ``'global'`` or ``'local'``.
        mode: ``'spot_type'`` — categorical colours derived from z-score
            threshold (|z| > 1.96); ``'zscore'`` — continuous diverging map.
        zscore_col: Column holding z-score values.
        cmap: Colormap for ``mode='zscore'``.
        palette: Dict of ``{'hotspot': color, 'coldspot': color, 'ns': color}``
            for ``mode='spot_type'``.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> hs = sj.spatial.polygon.patterns.hotspot_detection(sp, g, "total_counts")
        >>> sj.visualization.plot_polygon_hotspot_map(sp, hs)
    """
    _default = {"hotspot": "#d7191c", "coldspot": "#2c7bb6", "ns": "#e0e0e0"}
    palette = palette or _default

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    zvals = hotspot_df[zscore_col].reindex(gdf.index).fillna(0.0)

    if mode == "spot_type":
        labels = pd.Series("ns", index=gdf.index)
        labels[zvals > 1.96] = "hotspot"
        labels[zvals < -1.96] = "coldspot"
        colors = [palette.get(lbl, "grey") for lbl in labels]
        pc = _build_poly_collection(gdf, colors)
        pc.set_alpha(poly_alpha)
        import matplotlib.patches as mpatches

        handles = [mpatches.Patch(color=v, label=k) for k, v in palette.items()]
        fig, ax = plt.subplots(figsize=figsize)
        ax.add_collection(pc)
        ax.legend(
            handles=handles,
            title="Spot type",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )
    else:
        vmax = float(np.abs(zvals).max())
        colors_list, norm, mappable = continuous_colors(zvals, cmap, -vmax, vmax, 0.0)
        pc = _build_poly_collection(gdf, colors_list)
        pc.set_alpha(poly_alpha)
        fig, ax = plt.subplots(figsize=figsize)
        ax.add_collection(pc)
        plt.colorbar(mappable, ax=ax, shrink=0.6, label="Gi* z-score")

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Morphology class map
# ══════════════════════════════════════════════════════════════════════════════


def plot_morphology_class_map(
    spatioloji_obj,
    class_col: str = "morph_class",
    coord_type: str = "global",
    palette: str | list | dict = "Set2",
    poly_alpha: float = 0.85,
    figsize: tuple[float, float] = (9, 8),
    title: str = "Morphology class map",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon spatial map coloured by morphology class.

    Requires ``classify_morphology()`` to have been run, which writes a
    class label column (e.g. ``'morph_class'``) into ``cell_meta``.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        class_col: ``cell_meta`` column with morphology class labels (e.g.
            ``'round'``, ``'elongated'``, ``'irregular'`` from
            ``classify_morphology(method='threshold')``).
        coord_type: ``'global'`` or ``'local'``.
        palette: Colour palette for morphology classes.
        poly_alpha: Polygon fill opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.spatial.polygon.morphology.classify_morphology(sp, method="threshold")
        >>> sj.visualization.plot_morphology_class_map(sp)
    """
    if class_col not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"'{class_col}' not found in cell_meta. Run classify_morphology() first.")

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    labels = spatioloji_obj.cell_meta.reindex(gdf.index)[class_col]
    colors, _, handles = categorical_colors(labels, palette)
    pc = _build_poly_collection(gdf, colors)
    pc.set_alpha(poly_alpha)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(
        handles=handles,
        title="Morphology class",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Morphology correlation
# ══════════════════════════════════════════════════════════════════════════════


def plot_morphology_correlation(
    spatioloji_obj,
    metrics: list[str] | None = None,
    group_col: str | None = None,
    method: Literal["pearson", "spearman"] = "spearman",
    cmap: str = "RdBu_r",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "Morphology metric correlations",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of pairwise correlations between polygon morphology metrics.

    Args:
        spatioloji_obj: A ``spatioloji`` object with morphology metrics in
            ``cell_meta`` (run ``compute_morphology(store=True)`` first).
        metrics: List of metric column names (full or without ``'morph_'``
            prefix). Defaults to all ``morph_*`` columns.
        group_col: ``cell_meta`` column for stratification. When supplied,
            one correlation heatmap per group is drawn side by side.
        method: Correlation method: ``'pearson'`` or ``'spearman'``.
        cmap: Diverging colormap (range −1 to 1).
        annotate: If ``True``, overlay correlation values.
        figsize: Figure size. Auto-computed if ``None``.
        title: Figure suptitle.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.spatial.polygon.morphology.compute_morphology(sp, store=True)
        >>> sj.visualization.plot_morphology_correlation(sp, method="spearman")
    """
    meta = spatioloji_obj.cell_meta

    if metrics is None:
        metric_cols = [c for c in meta.columns if c.startswith("morph_")]
        if not metric_cols:
            raise ValueError("No 'morph_*' columns found. Run compute_morphology(store=True) first.")
    else:
        metric_cols = []
        for m in metrics:
            if m in meta.columns:
                metric_cols.append(m)
            elif f"morph_{m}" in meta.columns:
                metric_cols.append(f"morph_{m}")
            else:
                raise ValueError(f"Metric '{m}' not found in cell_meta.")

    def _corr_heatmap(ax, data: pd.DataFrame, subtitle: str | None = None) -> None:
        corr = data[metric_cols].corr(method=method)
        n = corr.shape[0]
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        labels = [c.replace("morph_", "") for c in corr.columns]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        if subtitle:
            ax.set_title(subtitle, fontsize=9)
        if annotate:
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if abs(corr.iloc[i, j]) > 0.6 else "black",
                    )
        return im

    if group_col is not None and group_col in meta.columns:
        groups = sorted(meta[group_col].dropna().unique())
        n_groups = len(groups)
        figsize = figsize or (max(5, len(metric_cols) * 0.7) * n_groups, max(4, len(metric_cols) * 0.65))
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        for ax, grp in zip(axes, groups, strict=False):
            sub = meta[meta[group_col] == grp]
            im = _corr_heatmap(ax, sub, subtitle=str(grp))
        plt.colorbar(im, ax=axes[-1], shrink=0.8, label=f"{method.capitalize()} r")
    else:
        n_m = len(metric_cols)
        figsize = figsize or (max(5, n_m * 0.7 + 1), max(4, n_m * 0.65))
        fig, ax = plt.subplots(figsize=figsize)
        im = _corr_heatmap(ax, meta)
        plt.colorbar(im, ax=ax, shrink=0.8, label=f"{method.capitalize()} r")

    fig.suptitle(title, y=1.02, fontsize=12)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Spatial autocorrelation lollipop
# ══════════════════════════════════════════════════════════════════════════════


def plot_spatial_autocorrelation(
    autocorr_df: pd.DataFrame,
    sort_by: Literal["statistic", "p_adjusted", "metric"] = "statistic",
    significance_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str = "Spatial autocorrelation (Moran's I)",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Lollipop chart of spatial autocorrelation statistics across metrics.

    Args:
        autocorr_df: DataFrame returned by
            ``spatial.polygon.statistics.spatial_autocorrelation_test()``.
            Columns: ``metric``, ``statistic``, ``expected``, ``z_score``,
            ``p_value``, ``p_adjusted``, ``significant``, ``method``.
        sort_by: Sort order for lollipops: ``'statistic'`` (descending),
            ``'p_adjusted'`` (ascending), or ``'metric'`` (alphabetical).
        significance_threshold: P-value threshold for colouring dots red.
        figsize: Figure size. Auto-computed from row count if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> ac = sj.spatial.polygon.statistics.spatial_autocorrelation_test(sp, g)
        >>> sj.visualization.plot_spatial_autocorrelation(ac)
    """
    df = autocorr_df.copy()
    p_col = "p_adjusted" if "p_adjusted" in df.columns else "p_value"

    if sort_by == "statistic":
        df = df.sort_values("statistic", ascending=True)
    elif sort_by == "p_adjusted":
        df = df.sort_values(p_col, ascending=False)
    else:
        df = df.sort_values("metric")

    n = len(df)
    figsize = figsize or (7, max(4, n * 0.3))
    expected_val = float(df["expected"].iloc[0]) if "expected" in df.columns else 0.0

    dot_colors = [
        "#d7191c" if (p_col in df.columns and row[p_col] < significance_threshold) else "#aaaaaa"
        for _, row in df.iterrows()
    ]

    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(n)
    ax.hlines(y_pos, expected_val, df["statistic"].values, colors="lightgrey", linewidth=1.2)
    ax.scatter(df["statistic"].values, list(y_pos), c=dot_colors, s=50, zorder=3)
    ax.axvline(expected_val, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["metric"].tolist(), fontsize=8)
    ax.set_xlabel("Moran's I statistic")
    ax.set_title(title)

    handles = [
        mpatches.Patch(color="#d7191c", label=f"p < {significance_threshold}"),
        mpatches.Patch(color="#aaaaaa", label="ns"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Morphology association
# ══════════════════════════════════════════════════════════════════════════════


def plot_morphology_association(
    assoc_result: dict,
    spatioloji_obj=None,
    palette: str | list | dict = "tab20",
    point_size: float = 6.0,
    alpha: float = 0.4,
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Violin (group mode) or scatter (gene-correlation mode) morphology association.

    Automatically detects the result mode from dict keys.

    Args:
        assoc_result: Dict returned by
            ``spatial.polygon.statistics.morphology_association_test()``.
            Group mode keys: ``metric``, ``group_col``, ``test``,
            ``statistic``, ``p_value``, ``significant``.
            Correlation mode keys: ``metric``, ``gene``, ``correlation``,
            ``p_value``.
        spatioloji_obj: Required to fetch cell-meta (group mode) or gene
            expression (correlation mode).
        palette: Colour palette for groups (group mode).
        point_size: Marker size for scatter (correlation mode).
        alpha: Point opacity (correlation mode).
        figsize: Figure size.
        title: Plot title. Defaults to ``f"Morphology association — {metric}"``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.statistics.morphology_association_test(
        ...     sp, metric="circularity", group_col="cell_type")
        >>> sj.visualization.plot_morphology_association(result, sp)
    """
    if spatioloji_obj is None:
        raise ValueError("spatioloji_obj is required for plot_morphology_association.")

    metric: str = assoc_result.get("metric", "morph_metric")
    col = metric if metric in spatioloji_obj.cell_meta.columns else f"morph_{metric}"

    fig, ax = plt.subplots(figsize=figsize)

    if "group_col" in assoc_result:
        # ── group mode: violin ────────────────────────────────────────────────
        group_col = assoc_result["group_col"]
        if col not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"'{col}' not found in cell_meta.")
        plot_data = spatioloji_obj.cell_meta[[col, group_col]].dropna()
        sns.violinplot(data=plot_data, x=group_col, y=col, palette=palette, ax=ax, cut=0, linewidth=0.8)
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)

        stat_str = (
            f"{assoc_result.get('test', '')}  "
            f"stat={assoc_result.get('statistic', float('nan')):.3f}  "
            f"p={assoc_result.get('p_value', float('nan')):.4f}"
        )
        ax.set_title(title or f"Morphology association — {col}")
        ax.text(
            0.98,
            0.97,
            stat_str,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
        )

    elif "gene" in assoc_result:
        # ── correlation mode: scatter ─────────────────────────────────────────
        gene = assoc_result["gene"]
        if col not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"'{col}' not found in cell_meta.")
        morph_vals = spatioloji_obj.cell_meta[col].dropna()
        try:
            expr = spatioloji_obj.get_expression(gene_names=[gene]).squeeze()
        except Exception as exc:
            raise ValueError(f"Could not retrieve expression for gene '{gene}'.") from exc
        common = morph_vals.index.intersection(expr.index)
        x, y = morph_vals.reindex(common).values, expr.reindex(common).values

        ax.scatter(x, y, s=point_size, c="steelblue", alpha=alpha, linewidths=0)
        # OLS line
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xr = np.array([x.min(), x.max()])
            ax.plot(xr, m * xr + b, color="#d7191c", linewidth=1.5)

        corr = assoc_result.get("correlation", float("nan"))
        pval = assoc_result.get("p_value", float("nan"))
        ax.set_xlabel(col)
        ax.set_ylabel(f"{gene} expression")
        ax.set_title(title or f"Morphology association — {col}")
        ax.text(
            0.98,
            0.97,
            f"r = {corr:.3f},  p = {pval:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
        )
    else:
        raise ValueError("assoc_result must contain 'group_col' (group mode) or 'gene' (correlation mode).")

    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ── Public API ─────────────────────────────────────────────────────────────────

__all__ = [
    "plot_polygon_graph",
    "plot_contact_degree_distribution",
    "plot_polygon_neighborhood_enrichment",
    "plot_polygon_neighborhood_composition",
    "plot_polygon_niches",
    "plot_boundary_cells_map",
    "plot_free_boundary_map",
    "plot_contact_summary",
    "plot_density_map",
    "plot_polygon_hotspot_map",
    "plot_morphology_distribution",
    "plot_morphology_map",
    "plot_morphology_class_map",
    "plot_morphology_correlation",
    "plot_contact_permutation",
    "plot_boundary_enrichment",
    "plot_spatial_autocorrelation",
    "plot_morphology_association",
    "plot_interface_polygon_map",
    "plot_interface_polygon_metrics",
    "plot_gradient_curve",
    "plot_spatial_distance",
    "plot_infiltration_summary",
    "plot_motif_map",
    "plot_motif_composition",
    "plot_assembly_map",
    "plot_structure_matches",
]


# ══════════════════════════════════════════════════════════════════════════════
# Interface map
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_polygon_map(
    spatioloji_obj,
    interface_result,
    coord_type: str = "global",
    colors: dict | None = None,
    contour_color: str = "black",
    contour_width: float = 2.0,
    poly_alpha: float = 0.7,
    show_interior: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Polygon map coloured by interface role with contour overlay.

    Args:
        spatioloji_obj: A ``spatioloji`` object with polygon data.
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        coord_type: ``'global'`` or ``'local'``.
        colors: Dict mapping label -> colour. Defaults provide red/blue.
        contour_color: Colour of the interface contour line.
        contour_width: Width of the contour line.
        poly_alpha: Polygon fill opacity.
        show_interior: If ``False``, interior and other cells shown in grey.
        ax: Optional matplotlib Axes to draw into.
        figsize: Figure size (ignored if ``ax`` provided).
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.polygon.interface.identify_interface(
        ...     sp, g, "cell_type", "Tumor", "Stromal")
        >>> sj.visualization.plot_interface_map(sp, result)
    """
    from matplotlib.patches import Patch

    default_colors = {
        "region_a_interface": "#e74c3c",
        "region_b_interface": "#3498db",
        "interior_a": "#fadbd8",
        "interior_b": "#d6eaf8",
        "other": "#e8e8e8",
    }
    grey = "#d5d5d5"
    cmap = {**(default_colors), **(colors or {})}

    gdf = spatioloji_obj.to_geopandas(coord_type=coord_type, include_metadata=False)
    labels = interface_result.cell_labels.reindex(gdf.index).fillna("other")

    if show_interior:
        face_colors = [cmap.get(lbl, grey) for lbl in labels]
    else:
        face_colors = [cmap.get(lbl, grey) if lbl.endswith("_interface") else grey for lbl in labels]

    pc = _build_poly_collection(gdf, face_colors)
    pc.set_alpha(poly_alpha)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.add_collection(pc)

    # Overlay contour with segment labels
    segs = interface_result.segments
    if segs is not None and len(segs) > 0:
        for _, seg_row in segs.iterrows():
            geom = seg_row.geometry
            seg_id = seg_row["segment_id"]
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    xs, ys = line.xy
                    ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
            elif geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
            mid = geom.interpolate(0.5, normalized=True)
            ax.annotate(
                f"Seg {seg_id}", (mid.x, mid.y),
                fontsize=7, fontweight="bold", color=contour_color,
                ha="center", va="bottom",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8, "ec": "none"},
                zorder=4,
            )
    elif interface_result.contour is not None:
        if interface_result.contour.geom_type == "MultiLineString":
            for line in interface_result.contour.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
        elif interface_result.contour.geom_type == "LineString":
            xs, ys = interface_result.contour.xy
            ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Legend
    legend_items = []
    for lbl in ["region_a_interface", "region_b_interface", "interior_a", "interior_b", "other"]:
        n = (labels == lbl).sum()
        if n > 0:
            display = lbl.replace("_", " ")
            legend_items.append(Patch(facecolor=cmap.get(lbl, grey), label=f"{display} ({n})"))
    ax.legend(handles=legend_items, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, frameon=False)

    ra = interface_result.region_a
    rb = interface_result.region_b
    ax.set_title(title or f"Interface: {ra} vs {rb} ({interface_result.method})")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Interface metrics bar chart
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_polygon_metrics(
    interface_result,
    metric: str = "length",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Horizontal bar chart of per-segment interface metrics.

    Args:
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        metric: Column to plot: ``'length'``, ``'tortuosity'``,
            ``'n_cells_a'``, or ``'n_cells_b'``.
        ax: Optional matplotlib Axes to draw into.
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> sj.visualization.plot_interface_metrics(result, metric="length")
    """
    segs = interface_result.segments

    if len(segs) == 0:
        figsize = figsize or (6, 3)
        own_fig_empty = ax is None
        if own_fig_empty:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No interface segments found", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(title or f"Interface segments — {metric}")
        clean_axes(ax)
        if not own_fig_empty:
            return fig
        return finalize_plot(fig, save_path, dpi, show)

    n = len(segs)
    figsize = figsize or (7, max(3, n * 0.5 + 1))
    vals = segs[metric].values.astype(float)

    # Replace inf and NaN with 0 for display; track originals for annotation
    vals_display = np.where(np.isfinite(vals), vals, 0.0)

    bar_labels = [f"Seg {segs['segment_id'].iloc[i]}" for i in range(n)]

    from matplotlib.colors import Normalize

    cm_obj = plt.get_cmap("YlOrRd")
    finite = vals[np.isfinite(vals)]
    v_min = float(finite.min()) if len(finite) > 0 else 0.0
    v_max = float(finite.max()) if len(finite) > 0 else 1.0
    if v_min == v_max:
        v_min, v_max = v_min - 0.1, v_max + 0.1
    norm = Normalize(vmin=v_min, vmax=v_max)
    bar_colors = [cm_obj(norm(v)) for v in vals_display]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.barh(range(n), vals_display, color=bar_colors, height=0.6)

    # Annotate bars with values
    x_offset = 0.02 * (v_max - v_min) if (v_max - v_min) > 0 else 0.1
    for bar, v in zip(bars, vals, strict=True):
        if np.isinf(v):
            ax.text(x_offset, bar.get_y() + bar.get_height() / 2,
                    "inf", va="center", fontsize=7, color="red")
        elif np.isnan(v):
            ax.text(x_offset, bar.get_y() + bar.get_height() / 2,
                    "NaN", va="center", fontsize=7, color="gray")
        else:
            ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(bar_labels, fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Interface segments — {metric}")
    clean_axes(ax)

    # When ax was provided externally, don't close the figure — caller owns it
    if not own_fig:
        return fig
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Gradient & infiltration plots
# ══════════════════════════════════════════════════════════════════════════════


def plot_gradient_curve(
    gradient_result,
    genes: list[str] | None = None,
    programs: list[str] | None = None,
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Expression vs distance-from-interface curve plot.

    Args:
        gradient_result: GradientResult from ``compute_gradient``.
        genes: Genes to plot (from gene_gradients). None = all.
        programs: Programs to plot (from program_gradients). None = skip.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size. Auto-calculated if None.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.
    """
    import numpy as np

    bins = gradient_result.bins
    items = []

    if genes is None and programs is None:
        genes = list(gradient_result.gene_gradients.index)
    if genes:
        for g in genes:
            if g in gradient_result.gene_gradients.index:
                items.append(("gene", g))
    if programs:
        for p in programs:
            if p in gradient_result.program_gradients.index:
                items.append(("program", p))

    if not items:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No items to plot", ha="center", va="center", transform=ax.transAxes)
        return finalize_plot(fig, save_path, dpi, show)

    n_items = len(items)
    n_rows = int(np.ceil(n_items / n_cols))
    if figsize is None:
        figsize = (4 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, (item_type, name) in enumerate(items):
        ax = axes[idx // n_cols, idx % n_cols]

        if item_type == "gene":
            subset = bins[bins["gene"] == name]
            grad_row = gradient_result.gene_gradients.loc[name]
        else:
            subset = bins[bins["gene"] == name] if name in bins["gene"].values else None
            grad_row = gradient_result.program_gradients.loc[name]

        if subset is not None and not subset.empty:
            x = subset["distance_bin"].values
            y = subset["mean_expr"].values
            std = subset["std_expr"].values
            ax.plot(x, y, "-o", markersize=3, color="steelblue")
            ax.fill_between(x, y - std, y + std, alpha=0.2, color="steelblue")

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        coef = grad_row.get("coef", np.nan)
        r2 = grad_row.get("r2", np.nan)
        pval = grad_row.get("pvalue", np.nan)
        ax.set_title(name, fontsize=10)
        ax.annotate(
            f"slope={coef:.3f}\nR²={r2:.3f}\np={pval:.2e}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
            family="monospace",
        )
        ax.set_xlabel("Distance from interface")
        ax.set_ylabel("Expression")
        clean_axes(ax)

    for idx in range(n_items, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout()
    return finalize_plot(fig, save_path, dpi, show)


def plot_spatial_distance(
    sp,
    distances,
    interface_result=None,
    coord_type: str = "global",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Spatial map colored by signed distance from interface.

    Args:
        sp: spatioloji object.
        distances: Series of signed distances indexed by cell ID.
        interface_result: Optional InterfaceResult to overlay contour.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        cmap: Diverging colormap name.
        figsize: Figure size.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.
    """
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    d_vals = distances.reindex(sp.cell_index).values

    vmax = max(abs(np.nanmin(d_vals)), abs(np.nanmax(d_vals)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sc = ax.scatter(x, y, c=d_vals, cmap=cmap, norm=norm, s=8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Signed distance from interface")

    if interface_result is not None and interface_result.contour is not None:
        for geom in interface_result.contour.geoms:
            coords = np.array(geom.coords)
            ax.plot(coords[:, 0], coords[:, 1], "k-", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Signed Distance from Interface")
    ax.set_aspect("equal")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_infiltration_summary(
    infiltration_result,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart summarizing infiltration metrics per immune type.

    Three side-by-side panels: penetration depth, density slope,
    infiltration fraction.

    Args:
        infiltration_result: InfiltrationResult from ``score_infiltration``.
        figsize: Figure size.
        show: Whether to display the figure.
        save_path: Save figure to path if provided.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure.
    """
    import numpy as np

    metrics = infiltration_result.per_type_metrics
    if figsize is None:
        figsize = (12, 4)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    types = metrics.index.tolist()
    y_pos = np.arange(len(types))

    ax = axes[0]
    ax.barh(y_pos, metrics["median_depth"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Median depth")
    ax.set_title("Penetration Depth")
    clean_axes(ax)

    ax = axes[1]
    vals = metrics["density_slope"].values
    colors = ["salmon" if v < 0 else "steelblue" for v in vals]
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Density slope")
    ax.set_title("Density Gradient")
    clean_axes(ax)

    ax = axes[2]
    ax.barh(y_pos, metrics["infiltration_fraction"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    ax.set_xlabel("Fraction")
    ax.set_title("Infiltration Fraction")
    ax.set_xlim(0, 1)
    clean_axes(ax)

    fig.suptitle(f"Immune Infiltration into {infiltration_result.target_region}", fontsize=12)
    fig.tight_layout()
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Motif plots
# ══════════════════════════════════════════════════════════════════════════════


def plot_motif_map(
    sp,
    motif_result,
    coord_type: str = "global",
    point_size: float = 8,
    palette: str = "tab20",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Spatial scatter map coloured by motif label.

    Args:
        sp: A ``spatioloji`` object.
        motif_result: ``MotifResult`` from ``run_motif_pipeline``.
        coord_type: ``'global'`` or ``'local'`` coordinate system.
        point_size: Marker size for scatter points.
        palette: Colour palette name for motif categories.
        figsize: Figure size ``(width, height)``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure.

    Example:
        >>> result = run_motif_pipeline(sp, graph, group_col="cell_type", n_motifs=5)
        >>> plot_motif_map(sp, result, show=False)
    """
    pos_df = sp.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]

    labels = motif_result.motif_catalog.labels.reindex(pos_df.index).astype(str)
    colors, _, handles = categorical_colors(labels, palette)

    if figsize is None:
        figsize = (9, 8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.asarray(pos_df[x_col]),
        np.asarray(pos_df[y_col]),
        s=point_size,
        c=colors,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_title("Spatial Motif Map")
    ax.legend(
        handles=handles,
        title="Motif",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_motif_composition(
    motif_result,
    figsize: tuple[float, float] | None = None,
    palette: str = "tab20",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Stacked horizontal bar chart showing cell-type fractions per motif.

    Args:
        motif_result: ``MotifResult`` from ``run_motif_pipeline``.
        figsize: Figure size ``(width, height)``.
        palette: Colour palette name for cell types.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure.

    Example:
        >>> result = run_motif_pipeline(sp, graph, group_col="cell_type", n_motifs=5)
        >>> plot_motif_composition(result, show=False)
    """
    sigs = motif_result.motif_catalog.signatures
    cell_types = sigs.columns.tolist()
    n_motifs = sigs.shape[0]

    if figsize is None:
        figsize = (8, max(3, n_motifs * 0.6))

    pal = sns.color_palette(palette, len(cell_types))
    color_dict = dict(zip(cell_types, pal, strict=False))

    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(n_motifs)
    motif_labels = [str(m) for m in sigs.index]

    for ct in cell_types:
        vals = sigs[ct].values
        ax.barh(motif_labels, vals, left=left, color=color_dict[ct], label=ct)
        left += vals

    ax.set_xlabel("Fraction")
    ax.set_ylabel("Motif")
    ax.set_title("Motif Composition")
    ax.legend(
        title="Cell type",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    clean_axes(ax)
    fig.tight_layout()
    return finalize_plot(fig, save_path, dpi, show)


def plot_assembly_map(
    sp,
    motif_result,
    coord_type: str = "global",
    point_size: float = 8,
    palette: str = "Set2",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Spatial map coloured by assembly label.

    Args:
        sp: A ``spatioloji`` object.
        motif_result: ``MotifResult`` from ``run_motif_pipeline``.
        coord_type: ``'global'`` or ``'local'`` coordinate system.
        point_size: Marker size for scatter points.
        palette: Colour palette name for assembly categories.
        figsize: Figure size ``(width, height)``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If ``motif_result.assembly_catalog`` is None.

    Example:
        >>> result = run_motif_pipeline(sp, graph, group_col="cell_type", n_motifs=5)
        >>> plot_assembly_map(sp, result, show=False)
    """
    if motif_result.assembly_catalog is None:
        raise ValueError("motif_result.assembly_catalog is None — run the pipeline with detect_assemblies_flag=True.")

    pos_df = sp.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]

    raw_labels = motif_result.assembly_catalog.labels.reindex(pos_df.index)
    str_labels = raw_labels.astype(str).replace("-1", "unassigned")
    colors, _, handles = categorical_colors(str_labels, palette)

    if figsize is None:
        figsize = (9, 8)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.asarray(pos_df[x_col]),
        np.asarray(pos_df[y_col]),
        s=point_size,
        c=colors,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_title("Spatial Assembly Map")
    ax.legend(
        handles=handles,
        title="Assembly",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


def plot_structure_matches(
    sp,
    motif_result,
    structure_name: str,
    coord_type: str = "global",
    point_size: float = 8,
    highlight_color: str = "#d62728",
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Highlight cells matching a known structure on the spatial map.

    Matched cells are shown in ``highlight_color``; all other cells
    are rendered in grey.

    Args:
        sp: A ``spatioloji`` object.
        motif_result: ``MotifResult`` from ``run_motif_pipeline``.
        structure_name: Name of the structure to highlight (must exist
            in ``motif_result.structure_matches``).
        coord_type: ``'global'`` or ``'local'`` coordinate system.
        point_size: Marker size for scatter points.
        highlight_color: Colour for matched cells.
        figsize: Figure size ``(width, height)``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If ``motif_result.structure_matches`` is None.

    Example:
        >>> result = run_motif_pipeline(sp, graph, group_col="cell_type",
        ...     n_motifs=5, match_signatures={"tumor_core": {"Tumor": 0.6}})
        >>> plot_structure_matches(sp, result, "tumor_core", show=False)
    """
    if motif_result.structure_matches is None:
        raise ValueError("motif_result.structure_matches is None — run the pipeline with match_signatures or builtin.")

    pos_df = sp.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]

    per_cell = motif_result.structure_matches.per_cell.reindex(pos_df.index).fillna("unmatched")
    mask = per_cell == structure_name

    if figsize is None:
        figsize = (9, 8)

    fig, ax = plt.subplots(figsize=figsize)
    # Draw non-matched first, then matched on top
    ax.scatter(
        np.asarray(pos_df[x_col])[~mask],
        np.asarray(pos_df[y_col])[~mask],
        s=point_size,
        c="lightgrey",
        linewidths=0,
        label="other",
        zorder=1,
    )
    ax.scatter(
        np.asarray(pos_df[x_col])[mask],
        np.asarray(pos_df[y_col])[mask],
        s=point_size,
        c=highlight_color,
        linewidths=0,
        label=structure_name,
        zorder=2,
    )
    ax.set_aspect("equal")
    ax.set_title(f"Structure Match: {structure_name}")
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        frameon=False,
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)
