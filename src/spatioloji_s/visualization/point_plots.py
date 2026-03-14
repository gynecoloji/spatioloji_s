"""
point_plots.py — Visualization for centroid-based (point) spatial analysis.

Covers results produced by the ``spatial.point`` submodule:

    plot_spatial_graph              — cell scatter + KNN / radius / Delaunay edges
    plot_degree_distribution        — histogram / KDE of node degrees per type
    plot_neighborhood_enrichment    — cell-type × cell-type z-score heatmap
    plot_neighborhood_composition   — stacked-bar composition per group
    plot_niches                     — spatial scatter coloured by niche label
    plot_niche_signatures           — heatmap of niche × cell-type mean composition
    plot_neighborhood_diversity     — spatial scatter coloured by diversity metric
    plot_morans_i_map               — LISA cluster map  (HH / HL / LH / LL / ns)
    plot_morans_scatter             — classic Moran scatter (value vs spatial lag)
    plot_spatially_variable_genes   — ranked lollipop of top SVGs
    plot_co_occurrence              — co-occurrence score vs. distance (multi-line)
    plot_ripley                     — Ripley K / L curve with CSR ref + envelope
    plot_ripley_multi               — multiple K / L curves overlaid
    plot_getis_ord_map              — Gi* hotspot / coldspot spatial scatter
    plot_nearest_neighbor_distances — histogram / violin / spatial map of NND
    plot_proximity_score            — heatmap of median cross-type distances
    plot_permutation_test           — null distribution histogram + observed line
"""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from ._spatial_helpers import (
    categorical_colors,
    clean_axes,
    continuous_colors,
    finalize_plot,
)

# ══════════════════════════════════════════════════════════════════════════════
# Spatial graph
# ══════════════════════════════════════════════════════════════════════════════


def plot_spatial_graph(
    spatioloji_obj,
    graph,
    color_by: str | None = None,
    coord_type: str = "global",
    palette: str | list | dict = "tab20",
    point_size: float = 4.0,
    edge_alpha: float = 0.15,
    edge_color: str = "grey",
    edge_width: float = 0.4,
    max_edges: int = 50_000,
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Plot a point spatial graph — cell scatter with adjacency edges overlaid.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        graph: A ``PointSpatialGraph`` returned by any ``spatial.point.graph``
            ``build_*_graph`` function.
        color_by: Column in ``cell_meta`` to colour cells by. Categorical columns
            produce a legend; continuous columns produce a colorbar.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        palette: Seaborn / matplotlib palette name, list of colours, or dict
            mapping category labels to colours (categorical only).
        point_size: Marker size (pt²).
        edge_alpha: Opacity of graph edges.
        edge_color: Colour of graph edges.
        edge_width: Line width of graph edges.
        max_edges: Cap on drawn edges to prevent slow rendering of dense graphs.
        figsize: Figure size ``(width, height)`` in inches.
        title: Plot title. Auto-generated if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure (e.g. ``"graph.pdf"``).
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> g = sj.spatial.point.graph.build_knn_graph(sp, k=10)
        >>> sj.visualization.plot_spatial_graph(sp, g, color_by="leiden")
    """
    pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]
    cell_index = pos_df.index

    # ── build edge segments ────────────────────────────────────────────────────
    adj = graph.adjacency.tocoo()
    rows, cols = adj.row, adj.col
    mask = rows < cols  # upper triangle only (undirected)
    rows, cols = rows[mask], cols[mask]

    g_ids = list(getattr(graph, "cell_ids", getattr(graph, "cell_index", cell_index)))
    g_pos = pos_df.reindex(g_ids)

    if len(rows) > max_edges:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(rows), size=max_edges, replace=False)
        rows, cols = rows[idx], cols[idx]

    segs = [
        [(g_pos.iloc[r][x_col], g_pos.iloc[r][y_col]), (g_pos.iloc[c][x_col], g_pos.iloc[c][y_col])]
        for r, c in zip(rows, cols, strict=False)
    ]

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    if segs:
        lc = LineCollection(segs, linewidths=edge_width, colors=edge_color, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)

    if color_by is None:
        ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c="steelblue", linewidths=0, zorder=2)
    else:
        vals = spatioloji_obj.cell_meta.reindex(cell_index)[color_by]
        if vals.dtype.kind in ("O", "U") or str(vals.dtype) == "category":
            colors, _, handles = categorical_colors(vals, palette)
            ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c=colors, linewidths=0, zorder=2)
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
            colors, _, mappable = continuous_colors(vals, cmap_name, None, None, None)
            ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c=colors, linewidths=0, zorder=2)
            plt.colorbar(mappable, ax=ax, shrink=0.6, label=color_by)

    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"Spatial graph ({graph.method}, {graph.n_edges:,} edges)")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood enrichment
# ══════════════════════════════════════════════════════════════════════════════


def plot_neighborhood_enrichment(
    enrichment_result: dict,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    significance_threshold: float = 0.05,
    annotate: bool = True,
    title: str = "Neighbourhood enrichment (z-score)",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of neighbourhood enrichment z-scores (cell-type × cell-type).

    Accepts results from both
    ``spatial.point.neighborhoods.neighborhood_enrichment`` (key ``'zscore'``) and
    ``spatial.polygon.neighborhoods.neighborhood_enrichment`` (key ``'z_scores'``).

    Args:
        enrichment_result: Dict returned by ``neighborhood_enrichment()``.
            Must contain ``zscore`` or ``z_scores`` (DataFrame n_types × n_types).
            Optionally ``pvalue`` or ``p_values`` for significance markers.
        figsize: Figure size ``(width, height)``.
        cmap: Diverging colormap for z-scores.
        vmax: Symmetric colour range ``[-vmax, vmax]``. Auto if ``None``.
        significance_threshold: P-value threshold; cells below get a ``*`` marker.
        annotate: If ``True``, overlay z-score values and significance markers.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.neighborhoods.neighborhood_enrichment(sp, g, "cell_type")
        >>> sj.visualization.plot_neighborhood_enrichment(result)
    """
    zscore: pd.DataFrame | None = enrichment_result.get("zscore") or enrichment_result.get("z_scores")
    if zscore is None:
        raise ValueError("enrichment_result must contain key 'zscore' or 'z_scores'.")
    pvalue: pd.DataFrame | None = enrichment_result.get("pvalue") or enrichment_result.get("p_values")

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
# Neighbourhood composition
# ══════════════════════════════════════════════════════════════════════════════


def plot_neighborhood_composition(
    composition_df: pd.DataFrame,
    spatioloji_obj=None,
    group_col: str | None = None,
    palette: str | list | dict = "tab20",
    top_n_types: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "Neighbourhood composition",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Stacked-bar chart of mean neighbourhood composition per cell-type group.

    Args:
        composition_df: DataFrame ``(n_cells × n_types)`` returned by
            ``neighborhood_composition()``. Values are fractions (0–1) or counts.
        spatioloji_obj: Required when ``group_col`` is supplied — provides the
            grouping column from ``cell_meta``.
        group_col: Column in ``cell_meta`` to group cells by (e.g. ``'cell_type'``).
            If ``None``, each row is treated as its own group.
        palette: Colour palette for cell-type bars.
        top_n_types: Keep only the top N most abundant types; others merged as
            ``'Other'``.
        figsize: Figure size. Auto-computed from number of groups if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> comp = sj.spatial.point.neighborhoods.neighborhood_composition(sp, g, "cell_type")
        >>> sj.visualization.plot_neighborhood_composition(comp, sp, group_col="cell_type")
    """
    if group_col is not None and spatioloji_obj is not None:
        groups = spatioloji_obj.cell_meta.reindex(composition_df.index)[group_col]
        mean_comp = composition_df.groupby(groups).mean()
    else:
        mean_comp = composition_df.copy()

    if top_n_types is not None and composition_df.shape[1] > top_n_types:
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
# Niches
# ══════════════════════════════════════════════════════════════════════════════


def plot_niches(
    spatioloji_obj,
    niche_labels: pd.Series,
    coord_type: str = "global",
    palette: str | list | dict = "tab10",
    point_size: float = 5.0,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Spatial niches",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Spatial scatter coloured by niche label.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        niche_labels: ``pd.Series`` indexed by cell IDs with niche label values,
            as returned by ``identify_niches()["labels"]``.
        coord_type: ``'global'`` or ``'local'`` coordinates.
        palette: Colour palette for niches.
        point_size: Marker size (pt²).
        figsize: Figure size ``(width, height)``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.neighborhoods.identify_niches(sp, g, "cell_type")
        >>> sj.visualization.plot_niches(sp, result["labels"])
    """
    pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]
    labels = niche_labels.reindex(pos_df.index)
    colors, _, handles = categorical_colors(labels, palette)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c=colors, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
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
# Local Moran's I — LISA cluster map
# ══════════════════════════════════════════════════════════════════════════════


def plot_morans_i_map(
    spatioloji_obj,
    local_result: pd.DataFrame,
    coord_type: str = "global",
    spot_type_col: str = "spot_type",
    palette: dict | None = None,
    point_size: float = 5.0,
    show_ns: bool = True,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Local Moran's I — LISA clusters",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """LISA cluster spatial map (HH / LL / HL / LH / ns).

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        local_result: DataFrame returned by ``morans_i(local=True)`` with a
            ``spot_type`` column holding LISA class labels
            (``'HH'``, ``'LL'``, ``'HL'``, ``'LH'``, ``'ns'``).
        coord_type: ``'global'`` or ``'local'``.
        spot_type_col: Column name for LISA cluster class.
        palette: Dict mapping spot-type labels to colours. Defaults to the
            standard LISA colour scheme (red HH, blue LL, orange HL, light-blue
            LH, grey ns).
        point_size: Marker size (pt²).
        show_ns: If ``False``, ``ns`` cells are not drawn.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> local_i = sj.spatial.point.patterns.morans_i(sp, g, "EPCAM", local=True)
        >>> sj.visualization.plot_morans_i_map(sp, local_i)
    """
    _default = {"HH": "#d7191c", "LL": "#2c7bb6", "HL": "#fdae61", "LH": "#abd9e9", "ns": "#e0e0e0"}
    palette = palette or _default

    pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]
    plot_data = pos_df.join(local_result[[spot_type_col]], how="left")
    plot_data[spot_type_col] = plot_data[spot_type_col].fillna("ns")

    if not show_ns:
        plot_data = plot_data[plot_data[spot_type_col] != "ns"]

    fig, ax = plt.subplots(figsize=figsize)
    for stype in [s for s in ["HH", "HL", "LH", "LL", "ns"] if s in plot_data[spot_type_col].unique()]:
        sub = plot_data[plot_data[spot_type_col] == stype]
        ax.scatter(sub[x_col], sub[y_col], s=point_size, c=palette.get(stype, "grey"), label=stype, linewidths=0)

    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(
        title="LISA cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8, frameon=False
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Spatially variable genes
# ══════════════════════════════════════════════════════════════════════════════


def plot_spatially_variable_genes(
    svg_result: pd.DataFrame,
    n_top: int = 30,
    fdr_col: str = "fdr",
    i_col: str = "I",
    fdr_threshold: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str = "Spatially variable genes",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Ranked lollipop chart of top spatially variable genes (SVGs).

    Args:
        svg_result: DataFrame returned by ``spatially_variable_genes()``.
            Index is gene names; must contain ``I`` (Moran's I) and ``fdr``
            columns.
        n_top: Number of top genes to display.
        fdr_col: Column name for adjusted p-values (FDR).
        i_col: Column name for Moran's I statistic.
        fdr_threshold: FDR threshold for colouring significant genes red.
        figsize: Figure size. Auto-computed from ``n_top`` if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> svg = sj.spatial.point.patterns.spatially_variable_genes(sp, g)
        >>> sj.visualization.plot_spatially_variable_genes(svg, n_top=20)
    """
    if i_col not in svg_result.columns:
        raise ValueError(f"Column '{i_col}' not found. Available: {svg_result.columns.tolist()}")

    top = svg_result.nlargest(n_top, i_col)
    figsize = figsize or (7, max(4, n_top * 0.28))

    colors = [
        "#d7191c" if (fdr_col in top.columns and row[fdr_col] < fdr_threshold) else "#aaaaaa"
        for _, row in top.iterrows()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(top))
    ax.hlines(y_pos, 0, top[i_col].values, colors="lightgrey", linewidth=1.2)
    ax.scatter(top[i_col].values, list(y_pos), c=colors, s=40, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top.index.tolist(), fontsize=8)
    ax.set_xlabel("Moran's I")
    ax.set_title(title)
    handles = [
        mpatches.Patch(color="#d7191c", label=f"FDR < {fdr_threshold}"),
        mpatches.Patch(color="#aaaaaa", label="ns"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Co-occurrence
# ══════════════════════════════════════════════════════════════════════════════


def plot_co_occurrence(
    co_occurrence_result: dict,
    pairs: list[tuple[str, str]] | None = None,
    palette: str = "tab20",
    log_scale: bool = False,
    figsize: tuple[float, float] = (8, 5),
    title: str = "Co-occurrence score",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Multi-line co-occurrence score vs. distance plot.

    Args:
        co_occurrence_result: Dict returned by ``co_occurrence()``.
            Must contain ``score`` (DataFrame n_pairs × n_bins) and
            ``midpoints`` (1-D array of bin mid-distances).
        pairs: List of ``(type_a, type_b)`` tuples to plot. Plots all pairs if
            ``None``.
        palette: Seaborn / matplotlib palette for lines.
        log_scale: If ``True``, use a log-scale y-axis.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> co = sj.spatial.point.patterns.co_occurrence(sp, "cell_type")
        >>> sj.visualization.plot_co_occurrence(co, pairs=[("T cell", "Epithelial")])
    """
    score: pd.DataFrame = co_occurrence_result["score"]
    midpoints: np.ndarray = np.asarray(co_occurrence_result["midpoints"])

    if pairs is not None:
        keep = []
        for p in pairs:
            key = str(p) if str(p) in score.index else f"{p[0]}__{p[1]}"
            if key in score.index:
                keep.append(key)
            else:
                warnings.warn(f"Pair {p} not found in co-occurrence result.", stacklevel=2)
        score = score.loc[keep]

    pal = sns.color_palette(palette, len(score))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="random (1.0)")
    for (label, row), color in zip(score.iterrows(), pal, strict=False):
        ax.plot(midpoints, row.values, color=color, linewidth=1.5, label=str(label))

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Distance (px)")
    ax.set_ylabel("Co-occurrence score")
    ax.set_title(title)
    ax.legend(
        title="Type pair", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8, frameon=False
    )
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Ripley K / L
# ══════════════════════════════════════════════════════════════════════════════


def plot_ripley(
    ripley_result,
    show_envelope: bool = True,
    show_csr: bool = True,
    color: str = "#2c7bb6",
    envelope_color: str = "#abdda4",
    csr_color: str = "#d7191c",
    figsize: tuple[float, float] = (6, 5),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Ripley's K or L curve with CSR reference and optional simulation envelope.

    Args:
        ripley_result: ``RipleyResult`` returned by ``ripleys_k()``,
            ``ripleys_l()``, ``cross_k()``, ``cross_l()``, or
            ``simulation_envelope()``.
        show_envelope: If ``True`` and envelope bands are present, shade them.
        show_csr: If ``True``, draw the complete spatial randomness (CSR) line.
        color: Line colour for the observed statistic.
        envelope_color: Fill colour for the simulation envelope.
        csr_color: Line colour for the CSR reference.
        figsize: Figure size.
        title: Plot title. Auto-generated from ``function_type`` if ``None``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.ripley.simulation_envelope(sp, function="L")
        >>> sj.visualization.plot_ripley(result)
    """
    r = ripley_result.r
    stat = ripley_result.statistic
    csr = ripley_result.csr_expected
    ftype = ripley_result.function_type
    label = getattr(ripley_result, "label", ftype)

    fig, ax = plt.subplots(figsize=figsize)

    if show_envelope and getattr(ripley_result, "envelope_lo", None) is not None:
        ax.fill_between(
            r,
            ripley_result.envelope_lo,
            ripley_result.envelope_hi,
            color=envelope_color,
            alpha=0.5,
            label="95% envelope",
        )
    if show_csr:
        ax.plot(r, csr, color=csr_color, linestyle="--", linewidth=1.2, label="CSR")
    ax.plot(r, stat, color=color, linewidth=2.0, label=f"Observed {label}")

    ax.set_xlabel("r (px)")
    ax.set_ylabel(f"{ftype}(r)")
    ax.set_title(title or f"Ripley's {ftype}")
    ax.legend(fontsize=8, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Getis-Ord Gi*
# ══════════════════════════════════════════════════════════════════════════════


def plot_getis_ord_map(
    spatioloji_obj,
    getis_result: pd.DataFrame,
    coord_type: str = "global",
    mode: Literal["spot_type", "zscore"] = "spot_type",
    spot_type_col: str = "spot_type",
    zscore_col: str = "zscore",
    palette: dict | None = None,
    cmap: str = "RdBu_r",
    point_size: float = 5.0,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Getis-Ord Gi* — spatial hotspots",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Spatial scatter of Getis-Ord Gi* hotspot / coldspot results.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        getis_result: DataFrame returned by ``getis_ord_gi()`` with columns
            ``zscore``, ``pvalue``, and ``spot_type``
            (``'hotspot'`` / ``'coldspot'`` / ``'ns'``).
        coord_type: ``'global'`` or ``'local'``.
        mode: ``'spot_type'`` — categorical colours (hotspot / coldspot / ns);
            ``'zscore'`` — continuous diverging colormap of z-score values.
        spot_type_col: Column holding categorical spot-type labels.
        zscore_col: Column holding continuous Gi* z-scores.
        palette: Dict of spot-type → colour for ``mode='spot_type'``.
        cmap: Colormap for ``mode='zscore'``.
        point_size: Marker size (pt²).
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> gi = sj.spatial.point.patterns.getis_ord_gi(sp, g, "total_counts")
        >>> sj.visualization.plot_getis_ord_map(sp, gi)
    """
    _default = {"hotspot": "#d7191c", "coldspot": "#2c7bb6", "ns": "#e0e0e0"}
    palette = palette or _default

    pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]
    merged = pos_df.join(getis_result, how="left")

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "spot_type":
        merged[spot_type_col] = merged[spot_type_col].fillna("ns")
        for stype, color in palette.items():
            sub = merged[merged[spot_type_col] == stype]
            ax.scatter(sub[x_col], sub[y_col], s=point_size, c=color, label=stype, linewidths=0)
        ax.legend(
            title="Spot type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title_fontsize=8, frameon=False
        )
    else:
        zvals = merged[zscore_col].fillna(0.0)
        vmax = float(np.abs(zvals).max())
        colors, _, mappable = continuous_colors(zvals, cmap, -vmax, vmax, 0.0)
        ax.scatter(merged[x_col], merged[y_col], s=point_size, c=colors, linewidths=0)
        plt.colorbar(mappable, ax=ax, shrink=0.6, label="Gi* z-score")

    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Degree distribution
# ══════════════════════════════════════════════════════════════════════════════


def plot_degree_distribution(
    graph,
    spatioloji_obj=None,
    group_col: str | None = None,
    palette: str | list | dict = "tab20",
    bins: int = 30,
    figsize: tuple[float, float] = (6, 4),
    title: str = "Node degree distribution",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Histogram or per-group KDE of spatial graph node degrees.

    Args:
        graph: A ``PointSpatialGraph`` with a ``degree_series()`` method.
        spatioloji_obj: Required when ``group_col`` is supplied.
        group_col: Column in ``cell_meta`` to split degrees by (e.g.
            ``'cell_type'``). When supplied, overlaid KDE curves are drawn
            instead of a histogram.
        palette: Colour palette for groups.
        bins: Number of histogram bins (used only when ``group_col`` is
            ``None``).
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> g = sj.spatial.point.graph.build_knn_graph(sp, k=10)
        >>> sj.visualization.plot_degree_distribution(g, sp, group_col="cell_type")
    """
    degrees: pd.Series = graph.degree_series()

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
    ax.set_xlabel("Degree")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Niche signatures
# ══════════════════════════════════════════════════════════════════════════════


def plot_niche_signatures(
    niche_result: dict,
    palette: str = "YlOrRd",
    figsize: tuple[float, float] | None = None,
    title: str = "Niche signatures",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of mean cell-type composition per niche (niche × cell-type).

    Args:
        niche_result: Dict returned by ``identify_niches()``. Must contain
            ``'signatures'`` (DataFrame n_niches × n_types of mean fractions).
        palette: Sequential colormap for heatmap values (0–1 fractions).
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution for raster formats.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.neighborhoods.identify_niches(sp, g, "cell_type")
        >>> sj.visualization.plot_niche_signatures(result)
    """
    sigs: pd.DataFrame | None = niche_result.get("signatures")
    if sigs is None:
        raise ValueError("niche_result must contain key 'signatures'.")

    n_niches, n_types = sigs.shape
    figsize = figsize or (max(6, n_types * 0.7), max(4, n_niches * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sigs.values, cmap=palette, vmin=0, vmax=sigs.values.max(), aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean fraction")

    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_niches))
    ax.set_xticklabels(sigs.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sigs.index.tolist(), fontsize=8)
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Niche")
    ax.set_title(title)

    for i in range(n_niches):
        for j in range(n_types):
            ax.text(
                j,
                i,
                f"{sigs.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if sigs.iloc[i, j] > sigs.values.max() * 0.6 else "black",
            )

    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Neighbourhood diversity
# ══════════════════════════════════════════════════════════════════════════════


def plot_neighborhood_diversity(
    spatioloji_obj,
    diversity_df: pd.DataFrame,
    metric: str = "shannon_entropy",
    coord_type: str = "global",
    cmap: str = "viridis",
    point_size: float = 5.0,
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Spatial scatter coloured by neighbourhood diversity metric.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        diversity_df: DataFrame returned by ``neighborhood_diversity()``.
            Columns include ``shannon_entropy``, ``simpson_index``,
            ``richness``.
        metric: Column to colour by. One of ``'shannon_entropy'``,
            ``'simpson_index'``, ``'richness'``.
        coord_type: ``'global'`` or ``'local'``.
        cmap: Continuous colormap name.
        point_size: Marker size (pt²).
        figsize: Figure size.
        title: Plot title. Defaults to ``f"Neighbourhood diversity — {metric}"``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> div = sj.spatial.point.neighborhoods.neighborhood_diversity(sp, g, "cell_type")
        >>> sj.visualization.plot_neighborhood_diversity(sp, div, metric="shannon_entropy")
    """
    if metric not in diversity_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {diversity_df.columns.tolist()}")

    pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
    x_col, y_col = pos_df.columns[0], pos_df.columns[1]
    vals = diversity_df[metric].reindex(pos_df.index)
    colors, _, mappable = continuous_colors(vals, cmap, None, None, None)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c=colors, linewidths=0)
    plt.colorbar(mappable, ax=ax, shrink=0.6, label=metric)
    ax.set_aspect("equal")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"Neighbourhood diversity — {metric}")
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Moran scatter plot
# ══════════════════════════════════════════════════════════════════════════════


def plot_morans_scatter(
    values: pd.Series,
    local_result: pd.DataFrame,
    spatioloji_obj=None,
    color_by: str | None = None,
    palette: str | list | dict = "tab20",
    point_size: float = 8.0,
    alpha: float = 0.5,
    figsize: tuple[float, float] = (5, 5),
    title: str = "Moran scatter plot",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Classic Moran scatter plot (standardised value vs. spatial lag).

    Cells are plotted with their z-scored expression on the x-axis and the
    local Moran's I value (a proxy for the spatial lag) on the y-axis.
    Quadrant labels (HH / HL / LH / LL) indicate the LISA classification.

    Args:
        values: Expression / metric values, indexed by cell ID.
        local_result: DataFrame returned by ``morans_i(local=True)`` with
            columns ``local_i`` and optionally ``spot_type``.
        spatioloji_obj: If provided alongside ``color_by``, cell-meta column
            used for colouring points.
        color_by: ``cell_meta`` column for categorical colouring. If ``None``,
            colours by LISA ``spot_type`` when available.
        palette: Colour palette (used when ``color_by`` is set).
        point_size: Marker size (pt²).
        alpha: Point opacity.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> local_i = sj.spatial.point.patterns.morans_i(sp, g, "EPCAM", local=True)
        >>> expr = sp.get_expression(gene_names=["EPCAM"]).squeeze()
        >>> sj.visualization.plot_morans_scatter(expr, local_i)
    """
    common = values.index.intersection(local_result.index)
    x = values.reindex(common)
    x_std = (x - x.mean()) / (x.std() + 1e-12)
    y = local_result.reindex(common)["local_i"]

    fig, ax = plt.subplots(figsize=figsize)

    # colour points
    if color_by is not None and spatioloji_obj is not None:
        cat_vals = spatioloji_obj.cell_meta.reindex(common)[color_by]
        colors, _, handles = categorical_colors(cat_vals, palette)
        ax.scatter(x_std, y, s=point_size, c=colors, alpha=alpha, linewidths=0)
        ax.legend(
            handles=handles,
            title=color_by,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )
    elif "spot_type" in local_result.columns:
        _lisa_pal = {"HH": "#d7191c", "LL": "#2c7bb6", "HL": "#fdae61", "LH": "#abd9e9", "ns": "#e0e0e0"}
        spot = local_result.reindex(common)["spot_type"].fillna("ns")
        colors = [_lisa_pal.get(s, "grey") for s in spot]
        for stype, color in _lisa_pal.items():
            mask = spot == stype
            ax.scatter(x_std[mask], y[mask], s=point_size, c=color, alpha=alpha, linewidths=0, label=stype)
        ax.legend(title="LISA", fontsize=7, title_fontsize=8, frameon=False)
    else:
        ax.scatter(x_std, y, s=point_size, c="steelblue", alpha=alpha, linewidths=0)

    # reference lines
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    # OLS regression line
    mask_valid = pd.notna(x_std) & pd.notna(y)
    if mask_valid.sum() > 2:
        m, b = np.polyfit(x_std[mask_valid], y[mask_valid], 1)
        xr = np.array([x_std[mask_valid].min(), x_std[mask_valid].max()])
        ax.plot(xr, m * xr + b, color="#d7191c", linewidth=1.5, zorder=5)

    # quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    offset_x, offset_y = (xlim[1] - xlim[0]) * 0.04, (ylim[1] - ylim[0]) * 0.04
    for txt, xp, yp in [
        ("HH", xlim[1] - offset_x, ylim[1] - offset_y),
        ("HL", xlim[1] - offset_x, ylim[0] + offset_y),
        ("LH", xlim[0] + offset_x, ylim[1] - offset_y),
        ("LL", xlim[0] + offset_x, ylim[0] + offset_y),
    ]:
        ax.text(xp, yp, txt, ha="center", va="center", fontsize=8, color="grey", fontstyle="italic")

    ax.set_xlabel("Standardised value (z-score)")
    ax.set_ylabel("Spatial lag (local Moran's I)")
    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Ripley multi-curve
# ══════════════════════════════════════════════════════════════════════════════


def plot_ripley_multi(
    ripley_results: list,
    labels: list[str] | None = None,
    show_csr: bool = True,
    palette: str = "tab10",
    figsize: tuple[float, float] = (7, 5),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Multiple Ripley K / L curves overlaid on a single axes.

    Useful for comparing spatial clustering of different cell types or
    conditions at the same scale.

    Args:
        ripley_results: List of ``RipleyResult`` objects returned by
            ``ripleys_k()``, ``ripleys_l()``, ``cross_k()``, or ``cross_l()``.
        labels: Legend labels, one per result. Defaults to each result's
            ``label`` attribute, falling back to ``"Result {i}"``.
        show_csr: If ``True``, draw the CSR reference from the first result as
            a dashed black line.
        palette: Seaborn / matplotlib palette for lines.
        figsize: Figure size.
        title: Plot title. Defaults to ``f"Ripley's {function_type} — comparison"``.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> r_t = sj.spatial.point.ripley.ripleys_l(sp, cell_ids=t_cells)
        >>> r_b = sj.spatial.point.ripley.ripleys_l(sp, cell_ids=b_cells)
        >>> sj.visualization.plot_ripley_multi([r_t, r_b], labels=["T cells", "B cells"])
    """
    if not ripley_results:
        raise ValueError("ripley_results must contain at least one RipleyResult.")

    ftypes = [r.function_type for r in ripley_results]
    if len(set(ftypes)) > 1:
        warnings.warn(f"Mixing function types: {set(ftypes)}. Plot may be misleading.", stacklevel=2)

    ftype = ftypes[0]
    pal = sns.color_palette(palette, len(ripley_results))
    _labels = labels or [getattr(r, "label", f"Result {i}") for i, r in enumerate(ripley_results)]

    fig, ax = plt.subplots(figsize=figsize)

    if show_csr:
        ax.plot(
            ripley_results[0].r,
            ripley_results[0].csr_expected,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="CSR",
        )

    for result, label, color in zip(ripley_results, _labels, pal, strict=False):
        ax.plot(result.r, result.statistic, color=color, linewidth=1.8, label=label)

    ax.set_xlabel("r (px)")
    ax.set_ylabel(f"{ftype}(r)")
    ax.set_title(title or f"Ripley's {ftype} — comparison")
    ax.legend(fontsize=8, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Nearest-neighbour distances
# ══════════════════════════════════════════════════════════════════════════════


def plot_nearest_neighbor_distances(
    nnd_df: pd.DataFrame,
    spatioloji_obj=None,
    group_col: str | None = None,
    palette: str | list | dict = "tab20",
    mode: Literal["histogram", "violin", "spatial"] = "histogram",
    coord_type: str = "global",
    bins: int = 40,
    point_size: float = 5.0,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7, 5),
    title: str = "Nearest-neighbour distances",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Nearest-neighbour distance distribution as histogram, violin, or spatial map.

    Args:
        nnd_df: DataFrame returned by ``nearest_neighbor_distances()``.
            Must contain an ``nnd`` column.
        spatioloji_obj: Required for ``mode='violin'`` (needs ``cell_meta``)
            and ``mode='spatial'`` (needs spatial coordinates).
        group_col: ``cell_meta`` column used to split distributions in
            ``mode='histogram'`` (overlaid KDE) or ``mode='violin'``.
        palette: Colour palette for groups.
        mode: ``'histogram'`` — distribution plot; ``'violin'`` — per-group
            violin; ``'spatial'`` — scatter coloured by NND.
        coord_type: Coordinate system for ``mode='spatial'``.
        bins: Histogram bins (``mode='histogram'`` only).
        point_size: Marker size for ``mode='spatial'``.
        cmap: Colormap for ``mode='spatial'``.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> nnd = sj.spatial.point.statistics.nearest_neighbor_distances(sp)
        >>> sj.visualization.plot_nearest_neighbor_distances(nnd, sp, mode="violin",
        ...     group_col="cell_type")
    """
    if "nnd" not in nnd_df.columns:
        raise ValueError(f"'nnd' column not found. Available: {nnd_df.columns.tolist()}")

    fig, ax = plt.subplots(figsize=figsize)

    if mode == "histogram":
        if group_col is not None and spatioloji_obj is not None:
            groups = spatioloji_obj.cell_meta.reindex(nnd_df.index)[group_col]
            _, color_dict, _ = categorical_colors(groups.dropna(), palette)
            for grp, sub_idx in groups.groupby(groups).groups.items():
                sns.kdeplot(
                    nnd_df.loc[sub_idx, "nnd"].values,
                    ax=ax,
                    fill=True,
                    alpha=0.4,
                    color=color_dict.get(grp, "grey"),
                    label=str(grp),
                )
            ax.set_ylabel("Density")
            ax.legend(title=group_col, fontsize=7, title_fontsize=8, frameon=False)
        else:
            ax.hist(nnd_df["nnd"].values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
            ax.set_ylabel("Count")
        ax.set_xlabel("Nearest-neighbour distance (px)")

    elif mode == "violin":
        if spatioloji_obj is None or group_col is None:
            raise ValueError("mode='violin' requires spatioloji_obj and group_col.")
        plot_data = nnd_df[["nnd"]].copy()
        plot_data[group_col] = spatioloji_obj.cell_meta.reindex(nnd_df.index)[group_col].values
        sns.violinplot(data=plot_data, x=group_col, y="nnd", palette=palette, ax=ax, cut=0, linewidth=0.8)
        ax.set_xlabel("")
        ax.set_ylabel("NND (px)")
        ax.tick_params(axis="x", rotation=45)

    else:  # spatial
        if spatioloji_obj is None:
            raise ValueError("mode='spatial' requires spatioloji_obj.")
        pos_df = spatioloji_obj.get_spatial_coords(coord_type=coord_type, as_dataframe=True)
        x_col, y_col = pos_df.columns[0], pos_df.columns[1]
        vals = nnd_df["nnd"].reindex(pos_df.index)
        colors, _, mappable = continuous_colors(vals, cmap, None, None, None)
        ax.scatter(pos_df[x_col], pos_df[y_col], s=point_size, c=colors, linewidths=0)
        plt.colorbar(mappable, ax=ax, shrink=0.6, label="NND (px)")
        ax.set_aspect("equal")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    ax.set_title(title)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Proximity score heatmap
# ══════════════════════════════════════════════════════════════════════════════


def plot_proximity_score(
    proximity_result: dict,
    use_normalized: bool = True,
    cmap: str = "viridis_r",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str = "Proximity score (median cross-type distance)",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of median cross-type distances (cell-type × cell-type).

    Lower values indicate closer proximity between types.

    Args:
        proximity_result: Dict returned by ``proximity_score()``. Keys:
            ``'median_distance'``, ``'mean_distance'``, ``'normalized'``
            (all DataFrames, n_types × n_types).
        use_normalized: If ``True`` and ``'normalized'`` is not ``None``, use
            that matrix; otherwise fall back to ``'median_distance'``.
        cmap: Reversed colormap (lower = closer = darker by default).
        annotate: If ``True``, overlay numeric values.
        figsize: Figure size. Auto-computed if ``None``.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> prox = sj.spatial.point.statistics.proximity_score(sp, "cell_type")
        >>> sj.visualization.plot_proximity_score(prox)
    """
    mat: pd.DataFrame = (
        proximity_result.get("normalized")
        if use_normalized and proximity_result.get("normalized") is not None
        else proximity_result.get("median_distance")
    )
    if mat is None:
        raise ValueError("proximity_result must contain 'median_distance' or 'normalized' key.")

    n = mat.shape[0]
    figsize = figsize or (max(5, n * 0.7 + 1), max(4, n * 0.6 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalised distance" if use_normalized else "Median distance (px)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index.tolist(), fontsize=8)
    ax.set_xlabel("Target cell type")
    ax.set_ylabel("Source cell type")
    ax.set_title(title)

    if annotate:
        vmax = float(np.nanmax(mat.values))
        for i in range(n):
            for j in range(n):
                v = mat.iloc[i, j]
                if pd.notna(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if v > vmax * 0.6 else "black",
                    )

    return finalize_plot(fig, save_path, dpi, show)


# ══════════════════════════════════════════════════════════════════════════════
# Permutation test null distribution
# ══════════════════════════════════════════════════════════════════════════════


def plot_permutation_test(
    perm_result: dict,
    n_bins: int = 40,
    color_null: str = "#aaaaaa",
    color_observed: str = "#d7191c",
    figsize: tuple[float, float] = (6, 4),
    title: str = "Permutation test",
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Null distribution histogram with observed and expected value markers.

    Args:
        perm_result: Dict returned by ``permutation_test()``. Expected keys:
            ``'observed'`` (float), ``'null_dist'`` (array), ``'expected'``
            (float), ``'zscore'`` (float), ``'pvalue'`` (float),
            ``'alternative'`` (str).
        n_bins: Number of histogram bins.
        color_null: Fill colour for the null distribution bars.
        color_observed: Colour for the observed-value line.
        figsize: Figure size.
        title: Plot title.
        show: If ``True``, call ``plt.show()``.
        save_path: File path to save the figure.
        dpi: Resolution.

    Returns:
        ``plt.Figure`` or ``None``.

    Example:
        >>> result = sj.spatial.point.statistics.permutation_test(sp, "cell_type", metric_fn)
        >>> sj.visualization.plot_permutation_test(result)
    """
    null = np.asarray(perm_result["null_dist"])
    observed = float(perm_result["observed"])
    expected = float(perm_result.get("expected", null.mean()))
    zscore = float(perm_result.get("zscore", 0.0))
    pvalue = float(perm_result.get("pvalue", 1.0))
    alternative = str(perm_result.get("alternative", "two-sided"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(null, bins=n_bins, color=color_null, edgecolor="white", linewidth=0.4, label="Null distribution")
    ax.axvline(observed, color=color_observed, linewidth=2.0, label=f"Observed = {observed:.3f}")
    ax.axvline(expected, color="black", linewidth=1.2, linestyle="--", label=f"Expected = {expected:.3f}")

    ax.text(
        0.98,
        0.96,
        f"z = {zscore:.2f}, p = {pvalue:.4f} ({alternative})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
    )

    ax.set_xlabel("Statistic")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    return finalize_plot(fig, save_path, dpi, show)


# ── Public API ─────────────────────────────────────────────────────────────────

__all__ = [
    "plot_spatial_graph",
    "plot_degree_distribution",
    "plot_neighborhood_enrichment",
    "plot_neighborhood_composition",
    "plot_niches",
    "plot_niche_signatures",
    "plot_neighborhood_diversity",
    "plot_morans_i_map",
    "plot_morans_scatter",
    "plot_spatially_variable_genes",
    "plot_co_occurrence",
    "plot_ripley",
    "plot_ripley_multi",
    "plot_getis_ord_map",
    "plot_nearest_neighbor_distances",
    "plot_proximity_score",
    "plot_permutation_test",
    "plot_interface_point_map",
    "plot_interface_point_metrics",
    "plot_gradient_curve",
    "plot_spatial_distance",
    "plot_infiltration_summary",
]


# ══════════════════════════════════════════════════════════════════════════════
# Gradient & infiltration plots (delegates to polygon_plots)
# ══════════════════════════════════════════════════════════════════════════════

from spatioloji_s.visualization.polygon_plots import (  # noqa: E402
    plot_gradient_curve,
    plot_infiltration_summary,
    plot_spatial_distance,
)

# ══════════════════════════════════════════════════════════════════════════════
# Interface map (point/scatter mode)
# ══════════════════════════════════════════════════════════════════════════════


def plot_interface_point_map(
    spatioloji_obj,
    interface_result,
    coord_type: str = "global",
    colors: dict | None = None,
    contour_color: str = "black",
    contour_width: float = 2.0,
    point_size: float = 5.0,
    show_interior: bool = True,
    ax=None,
    figsize: tuple[float, float] = (9, 8),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
):
    """Scatter plot coloured by interface role with contour overlay.

    Args:
        spatioloji_obj: A ``spatioloji`` object.
        interface_result: ``InterfaceResult`` from ``identify_interface()``.
        coord_type: ``'global'`` or ``'local'``.
        colors: Dict mapping label -> colour. Defaults provide red/blue.
        contour_color: Colour of the interface contour line.
        contour_width: Width of the contour line.
        point_size: Scatter dot size.
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
        >>> result = sj.spatial.point.interface.identify_interface(
        ...     sp, g, "cell_type", "Tumor", "Stromal")
        >>> sj.visualization.point_plots.plot_interface_map(sp, result)
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

    if coord_type == "global":
        x = spatioloji_obj.spatial.x_global
        y = spatioloji_obj.spatial.y_global
    else:
        x = spatioloji_obj.spatial.x_local
        y = spatioloji_obj.spatial.y_local

    labels = interface_result.cell_labels.reindex(spatioloji_obj.cell_index).fillna("other")

    if show_interior:
        c = [cmap.get(lbl, grey) for lbl in labels]
    else:
        c = [cmap.get(lbl, grey) if lbl.endswith("_interface") else grey for lbl in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(x, y, c=c, s=point_size, edgecolors="none", zorder=1)

    # Overlay contour
    if interface_result.contour is not None:
        if interface_result.contour.geom_type == "MultiLineString":
            for line in interface_result.contour.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)
        elif interface_result.contour.geom_type == "LineString":
            xs, ys = interface_result.contour.xy
            ax.plot(xs, ys, color=contour_color, linewidth=contour_width, zorder=3)

    ax.set_aspect("equal")

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


def plot_interface_point_metrics(
    interface_result,
    metric: str = "length",
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
):
    """Horizontal bar chart of per-segment interface metrics (point mode).

    Delegates to ``polygon_plots.plot_interface_metrics``.

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
    """
    from spatioloji_s.visualization.polygon_plots import (
        plot_interface_polygon_metrics as _poly_plot_metrics,
    )

    return _poly_plot_metrics(
        interface_result,
        metric=metric,
        ax=ax,
        figsize=figsize,
        title=title,
        show=show,
        save_path=save_path,
        dpi=dpi,
    )
