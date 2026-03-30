# plots.py
"""
Spatial transcriptomics plotting functions.

Public API:
    stitch_fov_images          – stitch FOV images into one canvas
    plot_global_polygon        – polygon plots in global coords
    plot_local_polygon         – polygon plots per FOV (local coords)
    plot_global_dots           – dot plots in global coords
    plot_local_dots            – dot plots per FOV (local coords)
"""

import os
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

# =============================================================================
# SECTION 1 — PRIVATE HELPERS
# =============================================================================


def _get_feature_data(sj_obj, feature: str, feature_df=None, feature_column=None) -> pd.DataFrame:
    """
    Return a two-column DataFrame: [cell_id_col, feature].

    Three sources (in priority order):
        1. External `feature_df`  — must contain cell_id and `feature` columns
        2. feature="metadata"     — pulls `feature_column` from cell_meta
        3. Default                — pulls `feature` directly from cell_meta
    """
    cell_id = sj_obj.config.cell_id_col

    if feature_df is not None:
        if cell_id not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain '{cell_id}' and '{feature}' columns")
        return feature_df[[cell_id, feature]].copy()

    meta = sj_obj.cell_meta.reset_index()

    if feature == "metadata":
        if feature_column is None or feature_column not in meta.columns:
            raise ValueError(f"feature_column='{feature_column}' not found in cell_meta")
        return meta[[cell_id, feature_column]].rename(columns={feature_column: feature_column})

    if feature not in meta.columns:
        raise ValueError(f"Feature '{feature}' not found in cell_meta")
    return meta[[cell_id, feature]].copy()


def _build_color_mapping(
    series: pd.Series,
    kind: Literal["categorical", "continuous"],
    colormap: str = "viridis",
    color_map: dict | None = None,
    vmin=None,
    vmax=None,
):
    """
    Build everything needed to color-code a feature series.

    Returns
    -------
    For categorical:
        color_lookup : Dict[category → RGB tuple]
        legend_handles : List[Patch]
        norm=None, cmap=None

    For continuous:
        color_lookup : None   (colors applied via norm+cmap directly)
        legend_handles : None
        norm : Normalize
        cmap : colormap object
    """
    if kind == "categorical":
        # Auto-convert if not already categorical
        if not pd.api.types.is_categorical_dtype(series):
            n = series.nunique()
            if n > 20:
                raise ValueError(f"Too many categories ({n}) for categorical plot. Use continuous instead.")
            series = series.astype("category")

        cats = series.cat.categories
        palette = plt.cm.tab10.colors

        # Use provided color_map or auto-assign
        if color_map is not None:
            lookup = {str(k): v for k, v in color_map.items()}
        else:
            lookup = {str(cat): palette[i % len(palette)] for i, cat in enumerate(cats)}

        handles = [Patch(facecolor=v, label=k) for k, v in lookup.items()]
        return lookup, handles, None, None

    else:  # continuous
        vmin = vmin if vmin is not None else series.min()
        vmax = vmax if vmax is not None else series.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(colormap)
        return None, None, norm, cmap


def _setup_grid(n_plots: int, grid_layout: tuple[int, int] | None, fig_w: int, fig_h: int):
    """
    Create a figure + flattened axes array for multi-FOV grid plots.

    Returns
    -------
    fig, axs (flattened np.ndarray)
    """
    if grid_layout is not None:
        rows, cols = grid_layout
    else:
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(fig_w * cols, fig_h * rows))
    axs = np.array(axs).flatten()
    return fig, axs


def _finalize(fig, save_dir: str, filename: str, dpi: int, show: bool):
    """Save → show/close. Always called at the end of public plot functions."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")
    if not show:
        plt.close(fig)
    return fig


# =============================================================================
# SECTION 2 — FOV IMAGE STITCHING
# =============================================================================


def stitch_fov_images(
    sj_obj,
    fov_ids: list[str | int] | None = None,
    flip_vertical: bool = True,
    save_path: str | None = None,
    show_plot: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 12),
    dpi: int = 300,
    verbose: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Stitch FOV images into one canvas using global FOV positions.

    Returns
    -------
    (stitched_image, min_x, min_y)
        The numpy image array and the coordinate offsets used —
        pass min_x / min_y to the plot_global_* functions so cell
        coordinates align correctly with the canvas.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if sj_obj.fov_positions is None:
        raise ValueError("fov_positions not found in spatioloji object")
    if sj_obj.images.n_total == 0:
        raise ValueError("No images found in spatioloji object")

    fov_ids = [str(f) for f in (fov_ids or sj_obj.images.fov_ids)]
    if not fov_ids:
        raise ValueError("No FOV IDs provided or available")

    # ── FOV position table ────────────────────────────────────────────────────
    positions = sj_obj.fov_positions.copy()
    positions.index = positions.index.astype(str)

    for col in ("x_global_px", "y_global_px"):
        if col not in positions.columns:
            raise ValueError(f"Missing required column in fov_positions: '{col}'")

    positions = positions.loc[positions.index.isin(fov_ids)]
    if positions.empty:
        raise ValueError(f"None of the selected FOVs {fov_ids} have position data")

    if verbose:
        print(f"Stitching {len(positions)} FOVs...")

    # ── Detect single-FOV image size ──────────────────────────────────────────
    first_img = sj_obj.get_image(str(positions.index[0]))
    if first_img is None:
        raise ValueError(f"Could not load image for FOV {positions.index[0]}")
    img_h, img_w = first_img.shape[:2]
    if verbose:
        print(f"FOV image size: {img_w} × {img_h} px")

    # ── Build canvas ──────────────────────────────────────────────────────────
    min_x = positions["x_global_px"].min()
    min_y = positions["y_global_px"].min()
    positions["x_off"] = (positions["x_global_px"] - min_x).astype(int)
    positions["y_off"] = (positions["y_global_px"] - min_y).astype(int)

    canvas_w = int(positions["x_off"].max() + img_w)
    canvas_h = int(positions["y_off"].max() + img_h)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if verbose:
        print(f"Canvas size: {canvas_w} × {canvas_h} px")

    # ── Place each FOV ────────────────────────────────────────────────────────
    placed = []
    for fov, row in positions.iterrows():
        img = sj_obj.get_image(str(fov))
        if img is None:
            if verbose:
                print(f"  Skipping FOV {fov}: image not found")
            continue

        if flip_vertical:
            img = cv2.flip(img, 0)

        x0, y0 = int(row["x_off"]), int(row["y_off"])
        canvas[y0 : y0 + img_h, x0 : x0 + img_w] = img
        placed.append(fov)
        if verbose:
            print(f"  Placed FOV {fov} at ({x0}, {y0})")

    if not placed:
        print("Warning: no FOVs were placed on canvas.")
        return None, None, None

    if verbose:
        print(f"Done — {len(placed)} FOVs placed.")

    # ── Display canvas ────────────────────────────────────────────────────────
    display_img = cv2.flip(canvas, 0) if flip_vertical else canvas
    rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    plot_title = title or f"Stitched FOV Image ({len(placed)} FOVs)"

    if save_path:
        # Auto-add extension if missing
        if not save_path.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf")):
            save_path += ".png"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(plot_title)
        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        if verbose:
            print(f"Saved → {save_path}")

    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    return canvas, min_x, min_y


# =============================================================================
# SECTION 3 — POLYGON RENDERING ENGINE
# =============================================================================


def _render_polygons_on_ax(ax, gdf, feature, kind, color_lookup, norm, cmap, edge_color, edge_width, alpha):
    """
    Draw all cell polygons onto a single Axes using PolyCollection (fast).

    Parameters
    ----------
    ax          : matplotlib Axes
    df          : DataFrame with columns [cell_id_col, x_col, y_col, feature]
    kind        : 'categorical' or 'continuous'
    color_lookup: Dict[category → color]  — used when kind='categorical'
    norm, cmap  : used when kind='continuous'
    """
    polys, face_colors = [], []

    for _cell_id, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Extract exterior coords — vertex order guaranteed by Shapely
        coords = np.array(geom.exterior.coords)
        polys.append(coords)

        if kind == "categorical":
            face_colors.append(color_lookup.get(str(row[feature]), (0.5, 0.5, 0.5, 1.0)))
        else:
            face_colors.append(cmap(norm(row[feature])))

    collection = PolyCollection(
        polys,
        facecolors=face_colors,
        edgecolors=edge_color,
        linewidths=edge_width,
        alpha=alpha,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


# =============================================================================
# SECTION 3 — PUBLIC POLYGON FUNCTIONS
# =============================================================================


def plot_global_polygon(
    sj_obj,
    feature: str,
    feature_df=None,
    feature_column: str | None = None,
    background_img: np.ndarray | None = None,
    min_x: float | None = None,
    min_y: float | None = None,
    # continuous params
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    # categorical params
    color_map: dict | None = None,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.01, 0.5),
    # shared style
    edge_color: str = "none",
    edge_width: float = 0.01,
    alpha: float = 1.0,
    figsize: tuple[int, int] = (12, 12),
    fig_title: str | None = None,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell polygons in global coordinates, colored by any feature.
    Auto-detects categorical vs. continuous based on data type.

    Usage
    -----
    # continuous gene expression
    sj.plotting.plot_global_polygon(sp, feature='PanCK')

    # categorical cell type
    sj.plotting.plot_global_polygon(sp, feature='cell_type')

    # from cell_meta column
    sj.plotting.plot_global_polygon(sp, feature='metadata', feature_column='cluster')
    """
    if sj_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    cell_id_col = sj_obj.config.cell_id_col
    x_col, y_col = sj_obj.config.get_coordinate_columns("global")

    # ── Resolve feature name & data ──────────────────────────────────────────
    feat_df = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    # if feature="metadata", _get_feature_data returns feature_column as the col name
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature
    feat_name = fig_title or feat_col

    # ── Auto-detect kind ──────────────────────────────────────────────────────
    kind = "categorical" if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else "continuous"

    # ── Build color mapping ───────────────────────────────────────────────────
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col],
        kind=kind,
        colormap=colormap,
        color_map=color_map,
        vmin=vmin,
        vmax=vmax,
    )

    # ── Apply global offsets ──────────────────────────────────────────────────
    gdf = sj_obj.to_geopandas(coord_type="global", include_metadata=False)
    # Apply global offset to geometry
    if min_x is None:
        min_x = sj_obj.polygons[x_col].min()
    if min_y is None:
        min_y = sj_obj.polygons[y_col].min()

    from shapely.affinity import translate

    gdf["geometry"] = gdf["geometry"].apply(lambda geom: translate(geom, xoff=-min_x, yoff=-min_y))

    # Merge feature onto gdf (gdf is indexed by cell_id)
    gdf = gdf.join(feat_df.set_index(cell_id_col)[[feat_col]], how="inner")
    if gdf.empty:
        raise ValueError("No overlapping cells between polygons and feature data")

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    if background_img is not None:
        h, w = background_img.shape[:2]
        ax.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), origin="upper")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    _render_polygons_on_ax(
        ax,
        gdf,
        feat_col,
        kind,
        color_lookup,
        norm,
        cmap,
        edge_color,
        edge_width,
        alpha,
    )

    if background_img is not None:
        ax.invert_yaxis()
    ax.set_title(feat_name, fontsize=14)

    # ── Legend / colorbar ─────────────────────────────────────────────────────
    if kind == "categorical":
        ax.legend(handles=legend_handles, loc=legend_loc, bbox_to_anchor=legend_bbox, title=feat_name)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.8, label=feat_name)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(" ", "_").lower()
        bg = "_with_bg" if background_img is not None else ""
        filename = f"polygon_{safe}_global_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)


def plot_local_polygon(
    sj_obj,
    feature: str,
    fov_ids: list[str | int] | None = None,
    feature_df=None,
    feature_column: str | None = None,
    background_img: bool = True,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = "right",
    color_map: dict | None = None,
    edge_color: str = "black",
    edge_width: float = 0.5,
    alpha: float = 0.8,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: tuple[int, int] | None = None,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    if sj_obj.polygons is None:
        raise ValueError("No polygon data in spatioloji object")

    cell_id_col = sj_obj.config.cell_id_col
    fov_id_col = sj_obj.config.fov_id_col

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = "categorical" if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else "continuous"
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col],
        kind=kind,
        colormap=colormap,
        color_map=color_map,
        vmin=vmin,
        vmax=vmax,
    )

    # ── Build GDF (local coords) — vertex order guaranteed by Shapely ─────────
    gdf = sj_obj.to_geopandas(coord_type="local", include_metadata=False)
    gdf[fov_id_col] = sj_obj.cell_meta[fov_id_col].astype(str)
    gdf = gdf.join(feat_df.set_index(cell_id_col)[[feat_col]], how="inner")
    if gdf.empty:
        raise ValueError("No overlapping cells between polygons and feature data")

    # ── FOV list ──────────────────────────────────────────────────────────────
    if fov_ids is None:
        fov_ids = sorted(gdf[fov_id_col].unique().tolist())
    fov_ids = [str(f) for f in fov_ids]

    # ── Grid ──────────────────────────────────────────────────────────────────
    fig, axs = _setup_grid(len(fov_ids), grid_layout, figure_width, figure_height)

    for idx, fov_id in enumerate(fov_ids):
        ax = axs[idx]
        fov_gdf = gdf[gdf[fov_id_col] == fov_id]

        # ── Shared image dimensions (consistent across all subplots) ─────────────
        sample_img = sj_obj.get_image(fov_id)
        if sample_img is None:
            raise ValueError(f"Could not load image for FOV '{fov_ids[0]}'")
        shared_h, shared_w = sample_img.shape[:2]

        if fov_gdf.empty:
            ax.set_title(f"FOV {fov_id} (no data)", fontsize=title_fontsize)
            ax.axis("off")
            continue

        if background_img:
            img = sj_obj.get_image(fov_id)
            if img is not None:
                ax.imshow(np.flipud(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), origin="lower")

        _render_polygons_on_ax(
            ax,
            fov_gdf,
            feat_col,
            kind,
            color_lookup,
            norm,
            cmap,
            edge_color,
            edge_width,
            alpha,
        )

        ax.set_xlim(0, shared_w)
        ax.set_ylim(0, shared_h)
        ax.set_title(f"FOV {fov_id}", fontsize=title_fontsize)

    for j in range(len(fov_ids), len(axs)):
        axs[j].axis("off")

    # ── Shared legend / colorbar ──────────────────────────────────────────────
    feat_name = feat_col
    if kind == "categorical":
        fig.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(1.02, 0.5), title=feat_name, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if colorbar_position == "right":
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(sm, cax=cbar_ax, label=feat_name)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        else:
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
            fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=feat_name)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.suptitle(feat_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if filename is None:
        safe = feat_col.replace(" ", "_").lower()
        bg = "_with_bg" if background_img else ""
        filename = f"polygon_{safe}_local_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)


# =============================================================================
# SECTION 4 — DOT RENDERING ENGINE
# =============================================================================


def _render_dots_on_ax(
    ax, df, x_col, y_col, feature, kind, color_lookup, norm, cmap, dot_size, edge_color, edge_width, alpha
):
    """
    Draw all cell dots onto a single Axes using ax.scatter.

    Parameters
    ----------
    ax          : matplotlib Axes
    df          : DataFrame with columns [x_col, y_col, feature]
    kind        : 'categorical' or 'continuous'
    color_lookup: Dict[category → color]  — used when kind='categorical'
    norm, cmap  : used when kind='continuous'

    Returns
    -------
    scatter object (needed for colorbar when kind='continuous')
    """
    if kind == "categorical":
        # Draw each category separately so legend labels work naturally
        scatter = None
        for cat, grp in df.groupby(feature):
            ax.scatter(
                grp[x_col],
                grp[y_col],
                s=dot_size,
                c=[color_lookup.get(str(cat), (0.5, 0.5, 0.5))],
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=alpha,
                label=str(cat),
            )
    else:
        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[feature],
            cmap=cmap,
            norm=norm,
            s=dot_size,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    return scatter


# =============================================================================
# SECTION 4 — PUBLIC DOT FUNCTIONS
# =============================================================================


def plot_global_dots(
    sj_obj,
    feature: str,
    feature_df=None,
    feature_column: str | None = None,
    background_img: np.ndarray | None = None,
    min_x: float | None = None,
    min_y: float | None = None,
    # continuous params
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    # categorical params
    color_map: dict | None = None,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.01, 0.5),
    # shared style
    dot_size: int = 20,
    edge_color: str | None = None,
    edge_width: float = 0.0,
    alpha: float = 1.0,
    figsize: tuple[int, int] = (12, 12),
    fig_title: str | None = None,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell centroids in global coordinates, colored by any feature.
    Auto-detects categorical vs. continuous based on data type.

    Uses CenterX_global_px / CenterY_global_px from cell_meta.

    Usage
    -----
    sj.plotting.plot_global_dots(sp, feature='PanCK')
    sj.plotting.plot_global_dots(sp, feature='cell_type')
    sj.plotting.plot_global_dots(sp, feature='metadata', feature_column='cluster')
    """
    cell_id_col = sj_obj.config.cell_id_col
    meta = sj_obj.cell_meta.reset_index()

    # ── Check required coordinate columns ─────────────────────────────────────
    for col in ("CenterX_global_px", "CenterY_global_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature
    feat_name = fig_title or feat_col

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = "categorical" if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else "continuous"
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col],
        kind=kind,
        colormap=colormap,
        color_map=color_map,
        vmin=vmin,
        vmax=vmax,
    )

    # ── Apply global offsets ──────────────────────────────────────────────────
    coords = meta[[cell_id_col, "CenterX_global_px", "CenterY_global_px"]].copy()
    if min_x is None:
        min_x = coords["CenterX_global_px"].min()
    if min_y is None:
        min_y = coords["CenterY_global_px"].min()
    coords["x_plot"] = coords["CenterX_global_px"] - min_x
    coords["y_plot"] = coords["CenterY_global_px"] - min_y

    merged = coords.merge(feat_df, on=cell_id_col)
    if merged.empty:
        raise ValueError("No overlapping cells between cell_meta and feature data")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if background_img is not None:
        h, w = background_img.shape[:2]
        ax.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), origin="upper")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    scatter = _render_dots_on_ax(
        ax,
        merged,
        "x_plot",
        "y_plot",
        feat_col,
        kind,
        color_lookup,
        norm,
        cmap,
        dot_size,
        edge_color,
        edge_width,
        alpha,
    )
    if background_img is not None:
        ax.invert_yaxis()
    ax.set_title(feat_name, fontsize=14)

    # ── Legend / colorbar ─────────────────────────────────────────────────────
    if kind == "categorical":
        ax.legend(handles=legend_handles, loc=legend_loc, bbox_to_anchor=legend_bbox, title=feat_name)
    else:
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=feat_name)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(" ", "_").lower()
        bg = "_with_bg" if background_img is not None else ""
        filename = f"dots_{safe}_global_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)


def plot_local_dots(
    sj_obj,
    feature: str,
    fov_ids: list[str | int] | None = None,
    feature_df=None,
    feature_column: str | None = None,
    background_img: bool = True,
    # continuous params
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = "right",
    # categorical params
    color_map: dict | None = None,
    # shared style
    dot_size: float = 20,
    edge_color: str = "none",
    edge_width: float = 0.0,
    alpha: float = 0.8,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: tuple[int, int] | None = None,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell centroids per FOV (local coordinates), colored by any feature.
    Auto-detects categorical vs. continuous.

    Uses CenterX_local_px / CenterY_local_px from cell_meta.

    Usage
    -----
    sj.plotting.plot_local_dots(sp, feature='PanCK', fov_ids=['1','2','3'])
    sj.plotting.plot_local_dots(sp, feature='cell_type', fov_ids=['1','2'])
    """
    cell_id_col = sj_obj.config.cell_id_col
    fov_id_col = sj_obj.config.fov_id_col
    meta = sj_obj.cell_meta.reset_index()

    # ── Check required coordinate columns ─────────────────────────────────────
    for col in ("CenterX_local_px", "CenterY_local_px"):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    # ── FOV list ──────────────────────────────────────────────────────────────
    if fov_ids is None:
        fov_ids = sorted(meta[fov_id_col].unique().astype(str).tolist())
    fov_ids = [str(f) for f in fov_ids]

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = "categorical" if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else "continuous"
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col],
        kind=kind,
        colormap=colormap,
        color_map=color_map,
        vmin=vmin,
        vmax=vmax,
    )

    coords = meta[[cell_id_col, fov_id_col, "CenterX_local_px", "CenterY_local_px"]].copy()
    merged = coords.merge(feat_df, on=cell_id_col)
    if merged.empty:
        raise ValueError("No overlapping cells between cell_meta and feature data")
    merged[fov_id_col] = merged[fov_id_col].astype(str)

    # ── Grid ──────────────────────────────────────────────────────────────────
    fig, axs = _setup_grid(len(fov_ids), grid_layout, figure_width, figure_height)

    for idx, fov_id in enumerate(fov_ids):
        ax = axs[idx]
        fov_data = merged[merged[fov_id_col] == fov_id]

        if fov_data.empty:
            ax.set_title(f"FOV {fov_id} (no data)", fontsize=title_fontsize)
            ax.axis("off")
            continue
        # ── Shared axis limits from image size ───────
        sample_img = sj_obj.get_image(fov_id)
        if sample_img is None:
            raise ValueError(f"Could not load image for FOV '{fov_ids[0]}' to determine shared dimensions")
        shared_h, shared_w = sample_img.shape[:2]

        ax.set_xlim(0, shared_w)
        ax.set_ylim(0, shared_h)

        # Optional background image
        if background_img:
            img = sj_obj.get_image(fov_id)
            if img is not None:
                h, w = img.shape[:2]
                ax.imshow(np.flipud(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), origin="lower")

        _render_dots_on_ax(
            ax,
            fov_data,
            "CenterX_local_px",
            "CenterY_local_px",
            feat_col,
            kind,
            color_lookup,
            norm,
            cmap,
            dot_size,
            edge_color,
            edge_width,
            alpha,
        )

        ax.set_title(f"FOV {fov_id}", fontsize=title_fontsize)

    # Hide unused axes
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis("off")

    # ── Shared legend / colorbar ──────────────────────────────────────────────
    feat_name = feat_col
    if kind == "categorical":
        fig.legend(handles=legend_handles, loc="center right", bbox_to_anchor=(1.02, 0.5), title=feat_name, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if colorbar_position == "right":
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(sm, cax=cbar_ax, label=feat_name)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        else:
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
            fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=feat_name)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.suptitle(feat_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(" ", "_").lower()
        bg = "_with_bg" if background_img else ""
        filename = f"dots_{safe}_local_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)


# =============================================================================
# SECTION A — GENE EXPRESSION HELPER
# =============================================================================


def _get_gene_expression(sj_obj, gene: str, layer: str | None = "log_normalized") -> pd.DataFrame:
    """
    Return a two-column DataFrame: [cell_id_col, gene].
    Same shape as _get_feature_data output — plugs into existing rendering pipeline.

    Parameters
    ----------
    sj_obj : spatioloji
    gene   : str — must exist in sj_obj.gene_index
    layer  : str or None
        None       → raw expression (sj_obj.expression)
        'layername' → sj_obj.layers['layername']
    """
    cell_id = sj_obj.config.cell_id_col

    # Validate gene exists
    if gene not in sj_obj.gene_index:
        raise ValueError(f"Gene '{gene}' not found. Check sj_obj.gene_index for available genes.")

    # Extract expression vector (n_cells,)
    if layer is None:
        expr = sj_obj.get_expression(gene_names=gene).flatten()
    else:
        if layer not in sj_obj.layers:
            raise ValueError(f"Layer '{layer}' not found. Available layers: {list(sj_obj.layers.keys())}")
        layer_data = sj_obj.get_layer(layer)
        gene_idx = sj_obj._get_gene_indices([gene])[0]

        from scipy import sparse

        if sparse.issparse(layer_data):
            expr = layer_data[:, gene_idx].toarray().flatten()
        else:
            expr = layer_data[:, gene_idx].flatten()

    # Build DataFrame aligned to master cell index
    return pd.DataFrame(
        {
            cell_id: sj_obj.cell_index.astype(str),
            gene: expr,
        }
    )


# =============================================================================
# SECTION B — GENE POLYGON FUNCTIONS
# =============================================================================


def plot_global_polygon_gene(
    sj_obj,
    gene: str,
    layer: str | None = "log_normalized",
    background_img: np.ndarray | None = None,
    min_x: float | None = None,
    min_y: float | None = None,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    edge_color: str = "none",
    edge_width: float = 0.01,
    alpha: float = 1.0,
    figsize: tuple[int, int] = (12, 12),
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell polygons in global coordinates colored by gene expression.

    Usage
    -----
    sj.visualization.plot_global_polygon_gene(sp, gene='PanCK')
    sj.visualization.plot_global_polygon_gene(sp, gene='CD3D', layer='normalized')
    sj.visualization.plot_global_polygon_gene(sp, gene='EPCAM', layer=None)  # raw
    """
    feat_df = _get_gene_expression(sj_obj, gene, layer)
    layer_label = layer or "raw"
    fig_title = f"{gene} ({layer_label})"

    return plot_global_polygon(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        background_img=background_img,
        min_x=min_x,
        min_y=min_y,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        edge_color=edge_color,
        edge_width=edge_width,
        alpha=alpha,
        figsize=figsize,
        fig_title=fig_title,
        save_dir=save_dir,
        filename=filename or f"polygon_{gene}_{layer_label}_global{'_bg' if background_img is not None else ''}.png",
        dpi=dpi,
        show_plot=show_plot,
    )


def plot_local_polygon_gene(
    sj_obj,
    gene: str,
    layer: str | None = "log_normalized",
    fov_ids: list[str | int] | None = None,
    background_img: bool = True,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = "right",
    edge_color: str = "none",
    edge_width: float = 0.01,
    alpha: float = 1.0,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: tuple[int, int] | None = None,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell polygons per FOV colored by gene expression.

    Usage
    -----
    sj.visualization.plot_local_polygon_gene(sp, gene='PanCK', fov_ids=['001','002'])
    sj.visualization.plot_local_polygon_gene(sp, gene='CD3D', layer='normalized')
    """
    feat_df = _get_gene_expression(sj_obj, gene, layer)
    layer_label = layer or "raw"

    return plot_local_polygon(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        fov_ids=fov_ids,
        background_img=background_img,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        colorbar_position=colorbar_position,
        edge_color=edge_color,
        edge_width=edge_width,
        alpha=alpha,
        figure_width=figure_width,
        figure_height=figure_height,
        grid_layout=grid_layout,
        title_fontsize=title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        save_dir=save_dir,
        filename=filename or f"polygon_{gene}_{layer_label}_local{'_bg' if background_img is True else ''}.png",
        dpi=dpi,
        show_plot=show_plot,
    )


# =============================================================================
# SECTION C — GENE DOT FUNCTIONS
# =============================================================================


def plot_global_dots_gene(
    sj_obj,
    gene: str,
    layer: str | None = "log_normalized",
    background_img: np.ndarray | None = None,
    min_x: float | None = None,
    min_y: float | None = None,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: int = 20,
    edge_color: str | None = None,
    edge_width: float = 0.0,
    alpha: float = 1.0,
    figsize: tuple[int, int] = (12, 12),
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell centroids in global coordinates colored by gene expression.

    Usage
    -----
    sj.visualization.plot_global_dots_gene(sp, gene='PanCK')
    sj.visualization.plot_global_dots_gene(sp, gene='CD3D', layer='normalized')
    sj.visualization.plot_global_dots_gene(sp, gene='EPCAM', layer=None)  # raw
    """
    feat_df = _get_gene_expression(sj_obj, gene, layer)
    layer_label = layer or "raw"
    fig_title = f"{gene} ({layer_label})"

    return plot_global_dots(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        background_img=background_img,
        min_x=min_x,
        min_y=min_y,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        dot_size=dot_size,
        edge_color=edge_color,
        edge_width=edge_width,
        alpha=alpha,
        figsize=figsize,
        fig_title=fig_title,
        save_dir=save_dir,
        filename=filename or f"dots_{gene}_{layer_label}_global{'_bg' if background_img is not None else ''}.png",
        dpi=dpi,
        show_plot=show_plot,
    )


def plot_local_dots_gene(
    sj_obj,
    gene: str,
    layer: str | None = "log_normalized",
    fov_ids: list[str | int] | None = None,
    background_img: bool = True,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = "right",
    dot_size: float = 20,
    edge_color: str = "none",
    edge_width: float = 0.0,
    alpha: float = 0.8,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: tuple[int, int] | None = None,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    save_dir: str = "./",
    filename: str | None = None,
    dpi: int = 300,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Plot cell centroids per FOV colored by gene expression.

    Usage
    -----
    sj.visualization.plot_local_dots_gene(sp, gene='PanCK', fov_ids=['001','002'])
    sj.visualization.plot_local_dots_gene(sp, gene='CD3D', layer='normalized')
    """
    feat_df = _get_gene_expression(sj_obj, gene, layer)
    layer_label = layer or "raw"

    return plot_local_dots(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        fov_ids=fov_ids,
        background_img=background_img,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        colorbar_position=colorbar_position,
        dot_size=dot_size,
        edge_color=edge_color,
        edge_width=edge_width,
        alpha=alpha,
        figure_width=figure_width,
        figure_height=figure_height,
        grid_layout=grid_layout,
        title_fontsize=title_fontsize,
        suptitle_fontsize=suptitle_fontsize,
        save_dir=save_dir,
        filename=filename or f"dots_{gene}_{layer_label}_local{'_bg' if background_img is True else ''}.png",
        dpi=dpi,
        show_plot=show_plot,
    )


# =============================================================================
# SECTION D — SIMPLE SPATIAL PLOTS (spatioloji class, global coords)
# =============================================================================


_CATEGORICAL_THRESHOLD = 30  # numeric columns with <= this many unique values → categorical


def _infer_kind(arr: np.ndarray) -> str:
    """Decide 'categorical' or 'continuous' from an array.

    Rules:
    - String / object dtype → categorical
    - Float with non-integer values (e.g. 0.333, 0.667) → continuous
      (even if few unique values — common for scores like AUCell)
    - Integer-like with <= _CATEGORICAL_THRESHOLD unique values → categorical
    - Otherwise → continuous
    """
    if arr.dtype.kind in ("U", "S", "O"):
        return "categorical"
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0:
            return "continuous"
        # If any value is non-integer, treat as continuous
        # (scores like AUCell produce 0.333, 0.667 etc.)
        if not np.all(finite == np.floor(finite)):
            return "continuous"
        # Integer-valued floats (e.g. leiden stored as float)
        n_unique = len(np.unique(finite))
        return "categorical" if n_unique <= _CATEGORICAL_THRESHOLD else "continuous"
    # Integer dtype
    n_unique = len(np.unique(arr))
    return "categorical" if n_unique <= _CATEGORICAL_THRESHOLD else "continuous"


def _xenium_resolve_values(sp, values, layer=None):
    """Resolve values to (array, label, kind).

    Parameters
    ----------
    sp : spatioloji
    values : str or np.ndarray
        Cell_meta column, gene name, or pre-computed array.
    layer : str or None
        Expression layer for gene lookup.

    Returns
    -------
    arr : np.ndarray (n_cells,)
    label : str
    kind : 'categorical' or 'continuous'
    """
    from scipy import sparse

    if isinstance(values, np.ndarray):
        arr = values.flatten()
        kind = _infer_kind(arr)
        if kind == "categorical":
            # Preserve NaN as None in object array
            out = np.where(pd.isna(arr), None, arr.astype(str))
            return out, "custom", "categorical"
        return arr.astype(float), "custom", "continuous"

    # String: try layer gene → cell_meta → raw expression
    if layer is not None:
        layer_data = sp.get_layer(layer)
        gene_indices = sp._get_gene_indices([values])
        if len(gene_indices) > 0:
            if sparse.issparse(layer_data):
                arr = np.array(layer_data[:, gene_indices[0]].todense()).flatten().astype(float)
            else:
                arr = layer_data[:, gene_indices[0]].flatten().astype(float)
            return arr, values, "continuous"

    if values in sp.cell_meta.columns:
        series = sp.cell_meta[values]
        if not pd.api.types.is_numeric_dtype(series):
            # Preserve NaN as None in object array
            out = np.where(series.isna(), None, series.astype(str).values)
            return out, values, "categorical"
        arr = series.values.astype(float)
        kind = _infer_kind(arr)
        if kind == "categorical":
            out = np.where(np.isnan(arr), None, arr.astype(int).astype(str))
            return out, values, "categorical"
        return arr, values, "continuous"

    # Try raw expression
    gene_indices = sp._get_gene_indices([values])
    if len(gene_indices) > 0:
        arr = sp.expression.get_dense()[:, gene_indices[0]].flatten().astype(float)
        return arr, values, "continuous"

    raise ValueError(f"'{values}' not found as gene name or cell_meta column")


def xenium_plot_spatial(
    sp,
    values: str | np.ndarray,
    layer: str | None = None,
    kind: str | None = None,
    mode: str = "dot",
    color_dict: dict | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 1,
    alpha: float = 0.7,
    edge_color: str = "none",
    edge_width: float = 0.0,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    legend_ncol: int = 1,
    legend_fontsize: int = 7,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.02, 0.5),
    ax: plt.Axes | None = None,
    save_dir: str | None = None,
    filename: str | None = None,
    dpi: int = 300,
    show: bool = True,
) -> plt.Figure:
    """
    Plot cells in global coordinates colored by any feature.

    Supports both centroid dot plots and filled polygon plots.
    Accepts a cell_meta column name, gene name, or a pre-computed
    array.  Auto-detects categorical vs. continuous:

    - **String / object dtype** → categorical
    - **Numeric with <= 30 unique values** → categorical (e.g. cluster IDs)
    - **Numeric with > 30 unique values** → continuous

    Default colors are chosen automatically:

    - Categorical: ``tab10``/``tab20``/``gist_ncar`` by category count
    - Continuous: ``viridis``

    Parameters
    ----------
    sp : spatioloji
        spatioloji object.
    values : str or np.ndarray
        What to color by.  Can be:

        - A cell_meta column name (e.g. ``'leiden'``, ``'morph_area'``)
        - A gene name (e.g. ``'EPCAM'``) — uses ``layer`` if given
        - A numpy array of length ``n_cells``
    layer : str or None
        Expression layer for gene lookup (e.g. ``'log_normalized'``).
        Ignored when ``values`` is a cell_meta column or array.
    kind : {'categorical', 'continuous'} or None
        Force the data type for coloring.  If None (default), auto-detects
        based on dtype and number of unique values.  Use ``kind='continuous'``
        when scores (e.g. AUCell) have few unique values but should still
        be shown on a continuous color scale.
    mode : {'dot', 'polygon'}
        ``'dot'`` — scatter plot of cell centroids (fast, works always).
        ``'polygon'`` — filled cell boundary polygons (requires polygon
        data in ``sp.polygons``).
    color_dict : dict or None
        ``{category: color}`` for categorical features.  If None,
        auto-assigns colors.
    cmap : str or None
        Matplotlib colormap name.  If None, auto-selects based on
        data type (``tab20`` for categorical, ``viridis`` for
        continuous).
    vmin, vmax : float or None
        Value range for continuous color scale.
    dot_size : float
        Scatter point size (dot mode only), by default 1.
    alpha : float
        Transparency, by default 0.7.
    edge_color : str
        Polygon edge color (polygon mode only), by default ``'none'``.
    edge_width : float
        Polygon edge width (polygon mode only), by default 0.0.
    figsize : tuple
        Figure size, by default (10, 8).
    title : str or None
        Plot title.  If None, uses the feature name.
    legend_ncol : int
        Number of columns in categorical legend, by default 1.
    legend_fontsize : int
        Font size for legend labels, by default 7.
    legend_loc : str
        Legend location anchor, by default ``'center left'``.
    legend_bbox : tuple
        ``(x, y)`` position for legend anchor, by default ``(1.02, 0.5)``
        (outside the right edge of the plot).
    ax : matplotlib Axes or None
        If provided, plot onto this axes instead of creating a new figure.
    save_dir : str or None
        Directory to save the figure.  If None, does not save.
    filename : str or None
        Filename for saving.  If None, auto-generates from feature name.
    dpi : int
        Resolution for saved figure, by default 300.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> # Dot plot — categorical
    >>> sj.visualization.xenium_plot_spatial(sp, 'leiden')
    >>>
    >>> # Polygon plot — categorical with custom colors
    >>> sj.visualization.xenium_plot_spatial(sp, 'leiden', mode='polygon',
    ...     color_dict={'0': 'red', '1': 'blue'})
    >>>
    >>> # Continuous — gene expression, saved to file
    >>> sj.visualization.xenium_plot_spatial(sp, 'EPCAM', layer='log_normalized',
    ...     save_dir='./figures', filename='EPCAM_spatial.png')
    """
    from matplotlib.lines import Line2D

    if mode not in ("dot", "polygon"):
        raise ValueError(f"mode must be 'dot' or 'polygon', got '{mode}'")

    if mode == "polygon" and sp.polygons is None:
        raise ValueError("mode='polygon' requires polygon data in sp.polygons")

    arr, label, auto_kind = _xenium_resolve_values(sp, values, layer=layer)
    # User-specified kind overrides auto-detection
    if kind is not None:
        if kind not in ("categorical", "continuous"):
            raise ValueError(f"kind must be 'categorical', 'continuous', or None, got '{kind}'")
        # If forcing continuous on a string array, keep the raw float values
        if kind == "continuous" and auto_kind == "categorical":
            # Re-resolve without categorical conversion
            if isinstance(values, str) and values in sp.cell_meta.columns:
                arr = sp.cell_meta[values].values.astype(float)
    else:
        kind = auto_kind
    coords = sp.get_spatial_coords(coord_type="global")
    x, y = coords[:, 0], coords[:, 1]

    if title is None:
        layer_str = f" ({layer})" if layer and kind == "continuous" else ""
        title = f"{label}{layer_str}"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    na_color = (0.75, 0.75, 0.75, 1.0)

    # --- Build color mapping ---
    if kind == "categorical":
        nan_mask = pd.isna(pd.array(arr, dtype=object))
        categories = sorted(set(str(v) for v in arr[~nan_mask]))
        n_cats = len(categories)

        if color_dict is not None:
            c_dict = {str(k): v for k, v in color_dict.items()}
        else:
            if cmap is not None:
                palette = plt.get_cmap(cmap)
            elif n_cats <= 10:
                palette = plt.get_cmap("tab10")
            elif n_cats <= 20:
                palette = plt.get_cmap("tab20")
            else:
                palette = plt.get_cmap("gist_ncar")
            c_dict = {cat: palette(i / max(n_cats - 1, 1)) for i, cat in enumerate(categories)}

        if mode == "dot":
            # NaN cells first (gray background)
            if nan_mask.any():
                ax.scatter(x[nan_mask], y[nan_mask], c=[na_color], s=dot_size,
                           alpha=alpha, linewidths=0)
            # Valid cells on top
            valid_mask = ~nan_mask
            if valid_mask.any():
                valid_colors = [c_dict.get(str(v), na_color) for v in arr[valid_mask]]
                ax.scatter(x[valid_mask], y[valid_mask], c=valid_colors, s=dot_size,
                           alpha=alpha, linewidths=0)
        else:  # polygon
            gdf = sp.to_geopandas(coord_type="global", include_metadata=False)
            polys, face_colors = [], []
            for i, (_, row) in enumerate(gdf.iterrows()):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                poly_coords = np.array(geom.exterior.coords)
                polys.append(poly_coords)
                val = arr[i]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    rgba = (*na_color[:3], alpha)
                else:
                    c = c_dict.get(str(val), na_color)
                    rgba = (*c[:3], alpha) if len(c) >= 3 else (*c, alpha)
                face_colors.append(rgba)

            collection = PolyCollection(
                polys, facecolors=face_colors,
                edgecolors=edge_color, linewidths=edge_width,
            )
            ax.add_collection(collection)
            ax.autoscale_view()

        # Legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=cat,
                   markerfacecolor=c_dict.get(cat, na_color), markersize=6)
            for cat in categories
        ]
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=f"NaN ({n_nan:,})",
                       markerfacecolor=na_color, markeredgecolor="gray",
                       markeredgewidth=0.5, markersize=6)
            )
        ax.legend(
            handles=legend_elements, title=label,
            loc=legend_loc, bbox_to_anchor=legend_bbox,
            ncol=legend_ncol, fontsize=legend_fontsize,
            framealpha=0.9,
        )

    else:  # continuous
        resolved_cmap = cmap if cmap is not None else "viridis"
        v_min = vmin if vmin is not None else np.nanmin(arr)
        v_max = vmax if vmax is not None else np.nanmax(arr)

        if mode == "dot":
            sc = ax.scatter(x, y, c=arr, cmap=resolved_cmap, vmin=v_min, vmax=v_max,
                            s=dot_size, alpha=alpha, linewidths=0)
        else:  # polygon
            cmap_obj = plt.get_cmap(resolved_cmap)
            norm = colors.Normalize(vmin=v_min, vmax=v_max)
            gdf = sp.to_geopandas(coord_type="global", include_metadata=False)
            polys, face_colors = [], []
            for i, (_, row) in enumerate(gdf.iterrows()):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                poly_coords = np.array(geom.exterior.coords)
                polys.append(poly_coords)
                val = arr[i]
                if np.isnan(val):
                    rgba = (*na_color[:3], alpha)
                else:
                    c = cmap_obj(norm(val))
                    rgba = (*c[:3], alpha)
                face_colors.append(rgba)

            collection = PolyCollection(
                polys, facecolors=face_colors,
                edgecolors=edge_color, linewidths=edge_width,
            )
            ax.add_collection(collection)
            ax.autoscale_view()
            # Create a ScalarMappable for colorbar
            sc = cm.ScalarMappable(norm=norm, cmap=cmap_obj)

        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if own_fig:
        plt.tight_layout()

    # Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            safe = label.replace(" ", "_").lower()
            filename = f"xenium_{safe}_{mode}_{kind}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    if own_fig and not show:
        plt.close(fig)
    return fig


# Xenium standard stain channels and their default colormaps
_XENIUM_STAIN_DEFAULTS = {
    "dapi": {"index": 0, "cmap": "Blues", "label": "DAPI"},
    "boundary": {"index": 1, "cmap": "Reds", "label": "Boundary (membrane)"},
    "interior": {"index": 2, "cmap": "Purples", "label": "Interior (18S RNA)"},
    "protein": {"index": 3, "cmap": "Greens", "label": "Protein (IF)"},
}


def _resolve_channel_indices(channels: list, stain_keys: list[str]) -> list[int]:
    """Convert channel names or ints to integer indices."""
    indices = []
    for ch in channels:
        if isinstance(ch, int):
            indices.append(ch)
        elif isinstance(ch, str):
            ch_lower = ch.lower()
            if ch_lower in stain_keys:
                indices.append(stain_keys.index(ch_lower))
            else:
                raise ValueError(f"Unknown channel '{ch}'. Use one of: {stain_keys} or integer index.")
        else:
            indices.append(int(ch))
    return indices


def xenium_show_staining(
    tif_paths: str | list[str],
    channels: str | list[str] | None = None,
    cmaps: str | list[str] | None = None,
    level: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    vmax_percentile: float = 99.5,
    figsize_per_panel: tuple[float, float] = (8, 8),
    titles: list[str] | None = None,
    save_dir: str | None = None,
    filename: str | None = None,
    dpi: int = 150,
    show: bool = True,
) -> plt.Figure:
    """
    Display Xenium morphology staining images.

    Supports two input modes:

    1. **Single multi-channel OME-TIFF** (e.g. ``morphology.ome.tif``) —
       reads selected channels from the pyramid.
    2. **List of per-channel TIFFs** (e.g. ``morphology_focus_000X.ome.tif``)
       — one file per stain.

    Channels are auto-detected from Xenium standard naming:
    ``0=DAPI``, ``1=boundary``, ``2=interior (18S RNA)``,
    ``3=protein (IF)``.

    Parameters
    ----------
    tif_paths : str or list of str
        Path(s) to morphology TIFF file(s).

        - Single string → multi-channel OME-TIFF (reads channels by index)
        - List of strings → one file per channel (order = channel order)
    channels : str, list of str, or None
        Which channels to show.  Can be:

        - ``None`` → all available channels
        - ``'dapi'`` → single channel
        - ``['dapi', 'protein']`` → subset
        - ``[0, 2]`` → by index

        Standard names: ``'dapi'``, ``'boundary'``, ``'interior'``,
        ``'protein'``.
    cmaps : str, list of str, or None
        Colormap(s) for each channel.  If None, uses Xenium defaults
        (Blues, Reds, Purples, Greens).  If a single string, applied
        to all channels.
    level : int
        Pyramid level to read (0 = full resolution, higher = more
        downsampled).  Default 3 (fast preview).
    vmin : float or None
        Minimum intensity.  If None, uses array minimum.
    vmax : float or None
        Maximum intensity.  If None, uses ``vmax_percentile`` of
        the image data.
    vmax_percentile : float
        Percentile for auto-vmax, by default 99.5.  Clips hot pixels.
    figsize_per_panel : tuple
        ``(width, height)`` per channel panel.
    titles : list of str or None
        Custom titles per channel.  If None, uses standard stain names.
    save_dir : str or None
        Directory to save the figure.
    filename : str or None
        Filename for saving.
    dpi : int
        Save resolution, by default 150.
    show : bool
        If True, display the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> # Per-channel TIFFs — show all 4
    >>> xenium_show_staining([
    ...     'morphology_focus_0000.ome.tif',  # DAPI
    ...     'morphology_focus_0001.ome.tif',  # boundary
    ...     'morphology_focus_0002.ome.tif',  # interior
    ...     'morphology_focus_0003.ome.tif',  # protein
    ... ])
    >>>
    >>> # Single multi-channel TIFF — show all channels
    >>> xenium_show_staining('morphology.ome.tif')
    >>>
    >>> # Show only DAPI and protein
    >>> xenium_show_staining('morphology.ome.tif', channels=['dapi', 'protein'])
    >>>
    >>> # Custom colormap, full resolution
    >>> xenium_show_staining('morphology.ome.tif', channels='dapi', cmaps='gray', level=0)
    """
    try:
        import tifffile
    except ImportError as err:
        raise ImportError("tifffile is required: pip install tifffile") from err

    # --- Resolve input mode ---
    is_multi = isinstance(tif_paths, str)
    stain_keys = list(_XENIUM_STAIN_DEFAULTS.keys())

    if is_multi:
        # Single multi-channel OME-TIFF — detect channel count from shape
        tif = tifffile.TiffFile(tif_paths)
        try:
            full_shape = tif.series[0].levels[level].shape
            n_channels_available = full_shape[0] if len(full_shape) == 3 else 1
        except (IndexError, AttributeError):
            n_channels_available = 1
    else:
        n_channels_available = len(tif_paths)

    # --- Resolve which channels to show ---
    if channels is None:
        channel_indices = list(range(min(n_channels_available, 4)))
    else:
        if isinstance(channels, (str, int)):
            channels = [channels]
        channel_indices = _resolve_channel_indices(channels, stain_keys)

    n_show = len(channel_indices)

    # --- Resolve cmaps ---
    if cmaps is None:
        cmap_list = [_XENIUM_STAIN_DEFAULTS[stain_keys[min(i, 3)]]["cmap"] for i in channel_indices]
    elif isinstance(cmaps, str):
        cmap_list = [cmaps] * n_show
    else:
        cmap_list = list(cmaps)
        if len(cmap_list) < n_show:
            cmap_list.extend(["gray"] * (n_show - len(cmap_list)))

    # --- Resolve titles ---
    if titles is None:
        title_list = [_XENIUM_STAIN_DEFAULTS[stain_keys[min(i, 3)]]["label"] for i in channel_indices]
    else:
        title_list = list(titles)
        if len(title_list) < n_show:
            title_list.extend([f"Channel {i}" for i in channel_indices[len(title_list):]])

    # --- Load images ---
    images = []
    for ch_idx in channel_indices:
        if is_multi:
            img = tifffile.imread(tif_paths, series=0, level=level)
            if img.ndim == 3:
                img = img[ch_idx]
        else:
            if ch_idx >= len(tif_paths):
                raise ValueError(f"Channel index {ch_idx} but only {len(tif_paths)} files provided")
            img = tifffile.imread(tif_paths[ch_idx], is_ome=False, level=level)
        images.append(img)

    # --- Plot ---
    fig_w = figsize_per_panel[0] * n_show
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(1, n_show, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()

    for i, (img, cmap_name, ch_title) in enumerate(zip(images, cmap_list, title_list, strict=True)):
        ax = axes[i]
        v_lo = vmin if vmin is not None else float(img.min())
        v_hi = vmax if vmax is not None else float(np.percentile(img, vmax_percentile))
        ax.imshow(img, cmap=cmap_name, vmin=v_lo, vmax=v_hi)
        ax.set_title(f"{ch_title}: {img.shape}", fontsize=12)
        ax.axis("off")

    plt.tight_layout()

    # Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            ch_str = "_".join(title_list).replace(" ", "_").replace("(", "").replace(")", "").lower()
            filename = f"xenium_staining_{ch_str}_level{level}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    if not show:
        plt.close(fig)
    return fig


def xenium_plot_spatial_bg(
    sp,
    bg_tif: str,
    values: str | np.ndarray,
    layer: str | None = None,
    kind: str | None = None,
    mode: str = "dot",
    pixel_size: float = 0.2125,
    bg_level: int = 3,
    bg_cmap: str = "gray",
    bg_vmin_pct: float = 1.0,
    bg_vmax_pct: float = 99.0,
    color_dict: dict | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    dot_size: float = 1,
    alpha: float = 0.5,
    edge_color: str = "none",
    edge_width: float = 0.0,
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
    legend_ncol: int = 1,
    legend_fontsize: int = 7,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (1.02, 0.5),
    ax: plt.Axes | None = None,
    save_dir: str | None = None,
    filename: str | None = None,
    dpi: int = 200,
    show: bool = True,
) -> plt.Figure:
    """
    Overlay cells on a Xenium morphology image background.

    Loads a single-channel morphology TIFF (e.g. DAPI from
    ``morphology_focus/morphology_focus_0000.ome.tif``), converts
    cell coordinates from microns to pixels at the chosen pyramid
    level, and overlays colored dots or polygons.

    Parameters
    ----------
    sp : spatioloji
        spatioloji object.
    bg_tif : str
        Path to the background TIFF file (single-channel, e.g.
        ``morphology_focus_0000.ome.tif`` for DAPI).
    values : str or np.ndarray
        What to color cells by (same as :func:`xenium_plot_spatial`).
    layer : str or None
        Expression layer for gene lookup.
    mode : {'dot', 'polygon'}
        ``'dot'`` for centroid scatter, ``'polygon'`` for filled polygons.
    pixel_size : float
        Xenium base pixel size in microns at level 0, by default 0.2125.
    bg_level : int
        Pyramid level to read for the background image, by default 3.
    bg_cmap : str
        Colormap for the background image, by default ``'gray'``.
    bg_vmin_pct, bg_vmax_pct : float
        Percentile range for background contrast, by default 1-99.
    color_dict : dict or None
        Custom ``{category: color}`` mapping.
    cmap : str or None
        Matplotlib colormap.  Auto-selects if None.
    vmin, vmax : float or None
        Value range for continuous color scale.
    dot_size : float
        Scatter point size, by default 1.
    alpha : float
        Cell overlay transparency, by default 0.5.
    edge_color : str
        Polygon edge color, by default ``'none'``.
    edge_width : float
        Polygon edge width, by default 0.0.
    figsize : tuple
        Figure size, by default (12, 10).
    title : str or None
        Plot title.
    legend_ncol : int
        Legend columns.
    legend_fontsize : int
        Legend font size.
    legend_loc : str
        Legend anchor location.
    legend_bbox : tuple
        Legend anchor position.
    ax : matplotlib Axes or None
        Existing axes to plot onto.
    save_dir : str or None
        Directory to save the figure.
    filename : str or None
        Filename for saving.
    dpi : int
        Save resolution, by default 200.
    show : bool
        If True, display the figure.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> # Leiden clusters on DAPI
    >>> xenium_plot_spatial_bg(
    ...     sp, 'morphology_focus/morphology_focus_0000.ome.tif',
    ...     values='leiden')
    >>>
    >>> # Gene expression on DAPI, polygon mode
    >>> xenium_plot_spatial_bg(
    ...     sp, 'morphology_focus/morphology_focus_0000.ome.tif',
    ...     values='EPCAM', layer='log_normalized', mode='polygon',
    ...     cmap='Reds', alpha=0.6)
    """
    try:
        import tifffile
    except ImportError as err:
        raise ImportError("tifffile is required: pip install tifffile") from err

    from matplotlib.lines import Line2D

    if mode not in ("dot", "polygon"):
        raise ValueError(f"mode must be 'dot' or 'polygon', got '{mode}'")
    if mode == "polygon" and sp.polygons is None:
        raise ValueError("mode='polygon' requires polygon data in sp.polygons")

    # --- Load background image ---
    bg_img = tifffile.imread(bg_tif, is_ome=False, level=bg_level)
    px_size = pixel_size * (2**bg_level)  # µm per pixel at this level

    # --- Resolve values ---
    arr, label, auto_kind = _xenium_resolve_values(sp, values, layer=layer)
    if kind is not None:
        if kind not in ("categorical", "continuous"):
            raise ValueError(f"kind must be 'categorical', 'continuous', or None, got '{kind}'")
        if kind == "continuous" and auto_kind == "categorical":
            if isinstance(values, str) and values in sp.cell_meta.columns:
                arr = sp.cell_meta[values].values.astype(float)
    else:
        kind = auto_kind

    # --- Convert cell coords from µm to pixels ---
    coords_um = sp.get_spatial_coords(coord_type="global")
    x_px = coords_um[:, 0] / px_size
    y_px = coords_um[:, 1] / px_size

    if title is None:
        layer_str = f" ({layer})" if layer and kind == "continuous" else ""
        title = f"{label}{layer_str} on {os.path.basename(bg_tif)}"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Draw background ---
    bg_vmin = np.percentile(bg_img, bg_vmin_pct)
    bg_vmax = np.percentile(bg_img, bg_vmax_pct)
    ax.imshow(bg_img, cmap=bg_cmap, vmin=bg_vmin, vmax=bg_vmax, origin="upper")

    na_color = (0.75, 0.75, 0.75, 1.0)

    # --- Overlay cells ---
    if kind == "categorical":
        nan_mask = pd.isna(pd.array(arr, dtype=object))
        categories = sorted(set(str(v) for v in arr[~nan_mask]))
        n_cats = len(categories)

        if color_dict is not None:
            c_dict = {str(k): v for k, v in color_dict.items()}
        else:
            if cmap is not None:
                palette = plt.get_cmap(cmap)
            elif n_cats <= 10:
                palette = plt.get_cmap("tab10")
            elif n_cats <= 20:
                palette = plt.get_cmap("tab20")
            else:
                palette = plt.get_cmap("gist_ncar")
            c_dict = {cat: palette(i / max(n_cats - 1, 1)) for i, cat in enumerate(categories)}

        if mode == "dot":
            if nan_mask.any():
                ax.scatter(x_px[nan_mask], y_px[nan_mask], c=[na_color], s=dot_size,
                           alpha=alpha * 0.3, linewidths=0)
            valid_mask = ~nan_mask
            if valid_mask.any():
                valid_colors = [c_dict.get(str(v), na_color) for v in arr[valid_mask]]
                ax.scatter(x_px[valid_mask], y_px[valid_mask], c=valid_colors,
                           s=dot_size, alpha=alpha, linewidths=0)
        else:  # polygon
            gdf = sp.to_geopandas(coord_type="global", include_metadata=False)
            polys, face_colors = [], []
            for i, (_, row) in enumerate(gdf.iterrows()):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                poly_coords = np.array(geom.exterior.coords) / px_size
                polys.append(poly_coords)
                val = arr[i]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    face_colors.append((*na_color[:3], alpha * 0.3))
                else:
                    c = c_dict.get(str(val), na_color)
                    face_colors.append((*c[:3], alpha))

            collection = PolyCollection(
                polys, facecolors=face_colors,
                edgecolors=edge_color, linewidths=edge_width,
            )
            ax.add_collection(collection)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=cat,
                   markerfacecolor=c_dict.get(cat, na_color), markersize=6)
            for cat in categories
        ]
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=f"NaN ({n_nan:,})",
                       markerfacecolor=na_color, markeredgecolor="gray",
                       markeredgewidth=0.5, markersize=6)
            )
        ax.legend(
            handles=legend_elements, title=label,
            loc=legend_loc, bbox_to_anchor=legend_bbox,
            ncol=legend_ncol, fontsize=legend_fontsize, framealpha=0.9,
        )

    else:  # continuous
        resolved_cmap = cmap if cmap is not None else "hot"
        v_min = vmin if vmin is not None else np.nanmin(arr)
        v_max = vmax if vmax is not None else np.nanmax(arr)

        if mode == "dot":
            sc = ax.scatter(x_px, y_px, c=arr, cmap=resolved_cmap, vmin=v_min, vmax=v_max,
                            s=dot_size, alpha=alpha, linewidths=0)
        else:  # polygon
            cmap_obj = plt.get_cmap(resolved_cmap)
            norm = colors.Normalize(vmin=v_min, vmax=v_max)
            gdf = sp.to_geopandas(coord_type="global", include_metadata=False)
            polys, face_colors = [], []
            for i, (_, row) in enumerate(gdf.iterrows()):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                poly_coords = np.array(geom.exterior.coords) / px_size
                polys.append(poly_coords)
                val = arr[i]
                if np.isnan(val):
                    face_colors.append((*na_color[:3], alpha))
                else:
                    c = cmap_obj(norm(val))
                    face_colors.append((*c[:3], alpha))

            collection = PolyCollection(
                polys, facecolors=face_colors,
                edgecolors=edge_color, linewidths=edge_width,
            )
            ax.add_collection(collection)
            sc = cm.ScalarMappable(norm=norm, cmap=cmap_obj)

        fig.colorbar(sc, ax=ax, shrink=0.8, label=label)

    ax.set_title(title)
    ax.axis("off")

    if own_fig:
        plt.tight_layout()

    # Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            safe = label.replace(" ", "_").lower()
            bg_name = os.path.splitext(os.path.basename(bg_tif))[0]
            filename = f"xenium_{safe}_on_{bg_name}_{mode}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    if own_fig and not show:
        plt.close(fig)
    return fig
