# src/spatioloji_s/visualization/ccc_plots.py
"""Visualization functions for Cell-Cell Communication results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from spatioloji_s.visualization._spatial_helpers import clean_axes, finalize_plot

# ── 1. Score heatmap ─────────────────────────────────────────────────────────


def plot_ccc_heatmap(
    ccc_result,
    lr_name: str | None = None,
    metric: str = "mean_score",
    top_n: int | None = None,
    alpha: float = 0.05,
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    annot: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Heatmap of communication scores (sender_type x receiver_type).

    If ``lr_name`` is specified, shows a single heatmap for that LR pair.
    Otherwise, aggregates across all significant LR pairs (FDR < alpha)
    or uses ``top_n`` LR pairs by mean score.

    Args:
        ccc_result: CCCResult from ``run_ccc``.
        lr_name: LR pair name to plot. ``None`` = aggregate across
            significant pairs.
        metric: Column to use for heatmap values. Default ``'mean_score'``.
        top_n: If set, use only the top N LR pairs by mean score.
        alpha: FDR threshold for filtering significant pairs when
            ``lr_name`` is None.
        figsize: Figure size.
        cmap: Colormap.
        annot: Annotate cells with values.
        title: Custom title.
        ax: Existing axes. If None, creates new figure.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    scores = ccc_result.scores.copy()

    if lr_name is not None:
        scores = scores[scores["lr_name"] == lr_name]
        if scores.empty:
            print(f"No scores found for lr_name='{lr_name}'")
            return None
        default_title = f"CCC: {lr_name}"
    else:
        if "fdr" in scores.columns:
            scores = scores[scores["fdr"] < alpha]
        if top_n is not None:
            top_pairs = scores.groupby("lr_name")[metric].mean().nlargest(top_n).index
            scores = scores[scores["lr_name"].isin(top_pairs)]
        scores = scores.groupby(["sender_type", "receiver_type"])[metric].mean().reset_index()
        default_title = f"CCC Heatmap ({metric}, FDR<{alpha})"

    pivot = scores.pivot_table(index="sender_type", columns="receiver_type", values=metric, aggfunc="mean")

    if figsize is None:
        n = max(len(pivot), len(pivot.columns))
        figsize = (max(6, n * 1.2), max(5, n * 1.0))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=annot,
        fmt=".3f" if annot else "",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": metric},
    )
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title(title or default_title)

    if own_fig:
        return finalize_plot(fig, save_path, dpi, show)
    return fig


# ── 2. Dot plot (LR pairs x type pairs) ─────────────────────────────────────


def plot_ccc_dotplot(
    ccc_result,
    top_n: int = 20,
    alpha: float = 0.05,
    size_col: str = "n_edges",
    color_col: str = "mean_score",
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Dot plot of top CCC interactions.

    Each dot represents an LR pair x sender_type x receiver_type
    interaction. Dot size = number of edges, color = mean score.

    Args:
        ccc_result: CCCResult from ``run_ccc``.
        top_n: Number of top interactions to show (by mean_score).
        alpha: FDR threshold for filtering.
        size_col: Column for dot size. Default ``'n_edges'``.
        color_col: Column for dot color. Default ``'mean_score'``.
        figsize: Figure size.
        cmap: Colormap.
        title: Custom title.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    scores = ccc_result.scores.copy()
    if "fdr" in scores.columns:
        scores = scores[scores["fdr"] < alpha]

    if scores.empty:
        print("No significant interactions to plot.")
        return None

    scores = scores.nlargest(top_n, "mean_score")
    scores["label"] = scores["lr_name"] + " | " + scores["sender_type"] + " → " + scores["receiver_type"]
    scores = scores.sort_values("mean_score", ascending=True)

    if figsize is None:
        figsize = (8, max(4, len(scores) * 0.35))

    fig, ax = plt.subplots(figsize=figsize)

    sizes = scores[size_col].values
    size_norm = sizes / max(sizes.max(), 1) * 200 + 20

    sc = ax.scatter(
        scores[color_col],
        scores["label"],
        s=size_norm,
        c=scores[color_col],
        cmap=cmap,
        edgecolors="k",
        linewidths=0.5,
        zorder=2,
    )
    ax.set_xlabel(color_col)
    ax.set_title(title or f"Top {len(scores)} CCC Interactions (FDR<{alpha})")
    clean_axes(ax)
    plt.colorbar(sc, ax=ax, label=color_col, shrink=0.7)

    return finalize_plot(fig, save_path, dpi, show)


# ── 3. Zone comparison bar chart ─────────────────────────────────────────────


def plot_ccc_zones(
    ccc_result,
    lr_name: str | None = None,
    top_n: int = 10,
    metric: str = "mean_score",
    figsize: tuple[float, float] | None = None,
    palette: dict | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Grouped bar chart comparing CCC scores across spatial zones.

    Shows how communication strength differs at the interface vs.
    interior regions.

    Args:
        ccc_result: CCCResult from ``run_ccc`` (must have zone_comparison).
        lr_name: LR pair to plot. ``None`` = top N by fold_change.
        top_n: Number of interactions if lr_name is None.
        metric: Score column. Default ``'mean_score'``.
        figsize: Figure size.
        palette: Zone color dict. Default uses built-in colors.
        title: Custom title.
        ax: Existing axes.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    if ccc_result.zone_comparison is None:
        print("No zone_comparison in CCCResult. Run with interface_result in config.")
        return None

    zones = ccc_result.zone_comparison.copy()
    zones = zones[zones["zone"] != "other"]

    if lr_name is not None:
        zones = zones[zones["lr_name"] == lr_name]
    else:
        # Pick top interactions by max fold_change at interface
        interface_fc = zones[zones["zone"] == "interface"].nlargest(top_n, "fold_change")
        keys = interface_fc[["lr_name", "sender_type", "receiver_type"]]
        zones = zones.merge(keys, on=["lr_name", "sender_type", "receiver_type"])

    if zones.empty:
        print("No zone data to plot.")
        return None

    zones["label"] = zones["lr_name"] + " | " + zones["sender_type"] + "→" + zones["receiver_type"]

    if palette is None:
        palette = {"interface": "#e74c3c", "interior_a": "#3498db", "interior_b": "#2ecc71"}

    if figsize is None:
        n_labels = zones["label"].nunique()
        figsize = (max(8, n_labels * 1.5), 5)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.barplot(
        data=zones,
        x="label",
        y=metric,
        hue="zone",
        palette=palette,
        ax=ax,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or "CCC by Spatial Zone")
    ax.tick_params(axis="x", rotation=45)
    clean_axes(ax)
    ax.legend(title="Zone", fontsize=8)

    if own_fig:
        return finalize_plot(fig, save_path, dpi, show)
    return fig


# ── 4. Communication gradient plot ───────────────────────────────────────────


def plot_ccc_gradient(
    ccc_result,
    lr_name: str | None = None,
    top_n: int = 5,
    figsize: tuple[float, float] | None = None,
    palette: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Bar chart of communication gradient slopes across the interface.

    Shows which interactions have the strongest spatial gradient
    (slope of score vs. distance to interface).

    Args:
        ccc_result: CCCResult from ``run_ccc`` (must have zone_gradient).
        lr_name: Specific LR pair. ``None`` = top N by |slope|.
        top_n: Number of interactions if lr_name is None.
        figsize: Figure size.
        palette: Color palette name.
        title: Custom title.
        ax: Existing axes.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    if ccc_result.zone_gradient is None:
        print("No zone_gradient in CCCResult. Run with interface_result in config.")
        return None

    grad = ccc_result.zone_gradient.copy()

    if lr_name is not None:
        grad = grad[grad["lr_name"] == lr_name]
    else:
        grad = grad.iloc[grad["slope"].abs().nlargest(top_n).index]

    if grad.empty:
        print("No gradient data to plot.")
        return None

    grad["label"] = grad["lr_name"] + " | " + grad["sender_type"] + "→" + grad["receiver_type"]
    grad = grad.sort_values("slope")
    colors = ["#e74c3c" if s > 0 else "#3498db" for s in grad["slope"]]

    # Mark significant with filled bars, non-significant with hatching
    sig_mask = grad["pvalue"] < 0.05

    if figsize is None:
        figsize = (8, max(4, len(grad) * 0.4))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.barh(grad["label"], grad["slope"], color=colors, edgecolor="k", linewidth=0.5)
    # Add hatching for non-significant
    for bar, is_sig in zip(bars, sig_mask.values, strict=True):
        if not is_sig:
            bar.set_hatch("//")
            bar.set_alpha(0.5)

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Gradient slope (score ~ distance)")
    ax.set_title(title or "Communication Gradient Across Interface")
    clean_axes(ax)

    # Legend for sig/non-sig
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", edgecolor="k", label="p < 0.05"),
        Patch(facecolor="gray", edgecolor="k", hatch="//", alpha=0.5, label="p >= 0.05"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    if own_fig:
        return finalize_plot(fig, save_path, dpi, show)
    return fig


# ── 5. Morphology comparison ────────────────────────────────────────────────


def plot_ccc_morphology(
    ccc_result,
    lr_name: str | None = None,
    top_n: int = 10,
    metric: str = "mean_score",
    figsize: tuple[float, float] | None = None,
    palette: str | dict | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Grouped bar chart comparing CCC scores across morphology groups.

    Args:
        ccc_result: CCCResult from ``run_ccc`` (must have morphology_comparison).
        lr_name: LR pair to plot. ``None`` = top N by fold_change range.
        top_n: Number of interactions if lr_name is None.
        metric: Score column. Default ``'mean_score'``.
        figsize: Figure size.
        palette: Morphology group colors.
        title: Custom title.
        ax: Existing axes.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    if ccc_result.morphology_comparison is None:
        print("No morphology_comparison in CCCResult. Run with morphology_col in config.")
        return None

    morph = ccc_result.morphology_comparison.copy()

    if lr_name is not None:
        morph = morph[morph["lr_name"] == lr_name]
    else:
        # Pick interactions with largest fold_change spread across groups
        fc_range = morph.groupby(["lr_name", "sender_type", "receiver_type"])["fold_change"].agg(
            lambda x: x.max() - x.min()
        )
        top_keys = fc_range.nlargest(top_n).reset_index()[["lr_name", "sender_type", "receiver_type"]]
        morph = morph.merge(top_keys, on=["lr_name", "sender_type", "receiver_type"])

    if morph.empty:
        print("No morphology data to plot.")
        return None

    morph["label"] = morph["lr_name"] + " | " + morph["sender_type"] + "→" + morph["receiver_type"]

    if figsize is None:
        n_labels = morph["label"].nunique()
        figsize = (max(8, n_labels * 1.5), 5)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.barplot(
        data=morph,
        x="label",
        y=metric,
        hue="morphology_group",
        palette=palette,
        ax=ax,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or "CCC by Sender Morphology")
    ax.tick_params(axis="x", rotation=45)
    clean_axes(ax)
    ax.legend(title="Morphology", fontsize=8)

    if own_fig:
        return finalize_plot(fig, save_path, dpi, show)
    return fig


# ── 6. Cell score spatial map ────────────────────────────────────────────────


def plot_ccc_spatial(
    sp,
    ccc_result,
    lr_name: str,
    role: str = "sender",
    coord_type: str = "global",
    cmap: str = "Reds",
    point_size: float = 1.0,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Spatial scatter of per-cell CCC scores for one LR pair.

    Args:
        sp: spatioloji object.
        ccc_result: CCCResult from ``run_ccc``.
        lr_name: LR pair name to plot.
        role: ``'sender'`` or ``'receiver'``.
        coord_type: ``'global'`` or ``'local'``.
        cmap: Colormap. Default ``'Reds'`` for sender, consider
            ``'Blues'`` for receiver.
        point_size: Scatter dot size.
        figsize: Figure size.
        title: Custom title.
        ax: Existing axes.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    col = f"{lr_name}_{role}"
    if col not in ccc_result.cell_scores.columns:
        print(f"Column '{col}' not found in cell_scores. Available: {list(ccc_result.cell_scores.columns)}")
        return None

    scores = ccc_result.cell_scores[col].reindex(sp.cell_index).fillna(0).values

    if coord_type == "global":
        x = np.asarray(sp.spatial.x_global)
        y = np.asarray(sp.spatial.y_global)
    else:
        x = np.asarray(sp.spatial.x_local)
        y = np.asarray(sp.spatial.y_local)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sc = ax.scatter(x, y, c=scores, s=point_size, cmap=cmap, edgecolors="none")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title or f"{lr_name} — {role} score")
    plt.colorbar(sc, ax=ax, label=f"{role} score", shrink=0.7)

    if own_fig:
        return finalize_plot(fig, save_path, dpi, show)
    return fig


# ── 7. Network chord / circos-style summary ──────────────────────────────────


def plot_ccc_network(
    ccc_result,
    alpha: float = 0.05,
    top_n: int | None = None,
    metric: str = "mean_score",
    figsize: tuple[float, float] = (8, 8),
    node_palette: str | dict | None = None,
    edge_cmap: str = "YlOrRd",
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Network plot of cell type communication.

    Nodes = cell types, edges = aggregated communication score
    (summed across significant LR pairs). Edge width and color
    proportional to total score.

    Args:
        ccc_result: CCCResult from ``run_ccc``.
        alpha: FDR threshold.
        top_n: Keep only top N edges by score.
        metric: Score metric for edge weight.
        figsize: Figure size.
        node_palette: Color palette for cell type nodes.
        edge_cmap: Colormap for edges.
        title: Custom title.
        show: Whether to display.
        save_path: Save path.
        dpi: DPI for saved figure.

    Returns:
        matplotlib Figure or None.
    """
    scores = ccc_result.scores.copy()
    if "fdr" in scores.columns:
        scores = scores[scores["fdr"] < alpha]

    if scores.empty:
        print("No significant interactions for network plot.")
        return None

    # Aggregate across LR pairs to get total score per type pair
    net = scores.groupby(["sender_type", "receiver_type"])[metric].sum().reset_index()
    net.columns = ["source", "target", "weight"]

    if top_n is not None:
        net = net.nlargest(top_n, "weight")

    if net.empty:
        return None

    # Build node positions in a circle
    all_types = sorted(set(net["source"]) | set(net["target"]))
    n_types = len(all_types)
    angles = np.linspace(0, 2 * np.pi, n_types, endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(all_types, angles, strict=True)}

    # Node colors
    if isinstance(node_palette, dict):
        node_colors = [node_palette.get(t, "lightblue") for t in all_types]
    else:
        pal = sns.color_palette(node_palette or "Set2", n_types)
        node_colors = list(pal)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    w_max = net["weight"].max()
    cm = plt.get_cmap(edge_cmap)
    for _, row in net.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        width = max(0.5, (w / w_max) * 8)
        color = cm(w / w_max)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=width,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Draw nodes
    for i, t in enumerate(all_types):
        x, y = pos[t]
        ax.scatter(x, y, s=800, c=[node_colors[i]], edgecolors="k", linewidths=1.5, zorder=5)
        ax.text(x, y, t, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"CCC Network (FDR<{alpha})")

    return finalize_plot(fig, save_path, dpi, show)
