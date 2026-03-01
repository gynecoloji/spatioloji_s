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

def _get_feature_data(sj_obj, feature: str,
                      feature_df=None, feature_column=None) -> pd.DataFrame:
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


def _build_color_mapping(series: pd.Series,
                         kind: Literal['categorical', 'continuous'],
                         colormap: str = 'viridis',
                         color_map: dict | None = None,
                         vmin=None, vmax=None):
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
    if kind == 'categorical':
        # Auto-convert if not already categorical
        if not pd.api.types.is_categorical_dtype(series):
            n = series.nunique()
            if n > 20:
                raise ValueError(f"Too many categories ({n}) for categorical plot. Use continuous instead.")
            series = series.astype('category')

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


def _setup_grid(n_plots: int,
                grid_layout: tuple[int, int] | None,
                fig_w: int, fig_h: int):
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
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved → {path}")
    if not show:
        plt.close(fig)
        return None
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

    for col in ('x_global_px', 'y_global_px'):
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
    min_x = positions['x_global_px'].min()
    min_y = positions['y_global_px'].min()
    positions['x_off'] = (positions['x_global_px'] - min_x).astype(int)
    positions['y_off'] = (positions['y_global_px'] - min_y).astype(int)

    canvas_w = int(positions['x_off'].max() + img_w)
    canvas_h = int(positions['y_off'].max() + img_h)
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

        x0, y0 = int(row['x_off']), int(row['y_off'])
        canvas[y0:y0 + img_h, x0:x0 + img_w] = img
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
        if not save_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf')):
            save_path += '.png'
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(plot_title)
        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f"Saved → {save_path}")

    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    return canvas, min_x, min_y

# =============================================================================
# SECTION 3 — POLYGON RENDERING ENGINE
# =============================================================================

def _render_polygons_on_ax(ax, gdf, feature,
                            kind, color_lookup, norm, cmap,
                            edge_color, edge_width, alpha):
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

        if kind == 'categorical':
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
    ax.set_aspect('equal')
    ax.axis('off')


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
    legend_loc: str = 'center left',
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
    x_col, y_col = sj_obj.config.get_coordinate_columns('global')

    # ── Resolve feature name & data ──────────────────────────────────────────
    feat_df = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    # if feature="metadata", _get_feature_data returns feature_column as the col name
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature
    feat_name = fig_title or feat_col

    # ── Auto-detect kind ──────────────────────────────────────────────────────
    kind = 'categorical' if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else 'continuous'

    # ── Build color mapping ───────────────────────────────────────────────────
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col], kind=kind,
        colormap=colormap, color_map=color_map,
        vmin=vmin, vmax=vmax,
    )

    # ── Apply global offsets ──────────────────────────────────────────────────
    gdf = sj_obj.to_geopandas(coord_type='global', include_metadata=False)
    # Apply global offset to geometry
    if min_x is None:
        min_x = sj_obj.polygons[x_col].min()
    if min_y is None:
        min_y = sj_obj.polygons[y_col].min()

    from shapely.affinity import translate
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: translate(geom, xoff=-min_x, yoff=-min_y)
    )

    # Merge feature onto gdf (gdf is indexed by cell_id)
    gdf = gdf.join(feat_df.set_index(cell_id_col)[[feat_col]], how='inner')
    if gdf.empty:
        raise ValueError("No overlapping cells between polygons and feature data")

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    if background_img is not None:
        h, w = background_img.shape[:2]
        ax.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), origin='upper')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    _render_polygons_on_ax(
        ax, gdf, feat_col,
        kind, color_lookup, norm, cmap, edge_color, edge_width, alpha,
    )

    if background_img is not None:
        ax.invert_yaxis()
    ax.set_title(feat_name, fontsize=14)

    # ── Legend / colorbar ─────────────────────────────────────────────────────
    if kind == 'categorical':
        ax.legend(handles=legend_handles, loc=legend_loc,
                  bbox_to_anchor=legend_bbox, title=feat_name)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.8, label=feat_name)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(' ', '_').lower()
        bg = '_with_bg' if background_img is not None else ''
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
    colorbar_position: str = 'right',
    color_map: dict | None = None,
    edge_color: str = 'black',
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
    fov_id_col  = sj_obj.config.fov_id_col

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df  = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = 'categorical' if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else 'continuous'
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col], kind=kind,
        colormap=colormap, color_map=color_map,
        vmin=vmin, vmax=vmax,
    )

    # ── Build GDF (local coords) — vertex order guaranteed by Shapely ─────────
    gdf = sj_obj.to_geopandas(coord_type='local', include_metadata=False)
    gdf[fov_id_col] = sj_obj.cell_meta[fov_id_col].astype(str)
    gdf = gdf.join(feat_df.set_index(cell_id_col)[[feat_col]], how='inner')
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
            ax.set_title(f'FOV {fov_id} (no data)', fontsize=title_fontsize)
            ax.axis('off')
            continue

        if background_img:
            img = sj_obj.get_image(fov_id)
            if img is not None:
                ax.imshow(np.flipud(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), origin='lower')

        _render_polygons_on_ax(
            ax, fov_gdf, feat_col,
            kind, color_lookup, norm, cmap, edge_color, edge_width, alpha,
        )

        ax.set_xlim(0, shared_w)
        ax.set_ylim(0, shared_h)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)

    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')

    # ── Shared legend / colorbar ──────────────────────────────────────────────
    feat_name = feat_col
    if kind == 'categorical':
        fig.legend(handles=legend_handles, loc='center right',
                   bbox_to_anchor=(1.02, 0.5), title=feat_name, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if colorbar_position == 'right':
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(sm, cax=cbar_ax, label=feat_name)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        else:
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label=feat_name)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.suptitle(feat_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if filename is None:
        safe = feat_col.replace(' ', '_').lower()
        bg = '_with_bg' if background_img else ''
        filename = f"polygon_{safe}_local_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)

# =============================================================================
# SECTION 4 — DOT RENDERING ENGINE
# =============================================================================

def _render_dots_on_ax(ax, df, x_col, y_col, feature,
                        kind, color_lookup, norm, cmap,
                        dot_size, edge_color, edge_width, alpha):
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
    if kind == 'categorical':
        # Draw each category separately so legend labels work naturally
        scatter = None
        for cat, grp in df.groupby(feature):
            ax.scatter(
                grp[x_col], grp[y_col],
                s=dot_size,
                c=[color_lookup.get(str(cat), (0.5, 0.5, 0.5))],
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=alpha,
                label=str(cat),
            )
    else:
        scatter = ax.scatter(
            df[x_col], df[y_col],
            c=df[feature],
            cmap=cmap,
            norm=norm,
            s=dot_size,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha,
        )

    ax.set_aspect('equal')
    ax.axis('off')
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
    legend_loc: str = 'center left',
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
    for col in ('CenterX_global_px', 'CenterY_global_px'):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df  = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature
    feat_name = fig_title or feat_col

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = 'categorical' if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else 'continuous'
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col], kind=kind,
        colormap=colormap, color_map=color_map,
        vmin=vmin, vmax=vmax,
    )

    # ── Apply global offsets ──────────────────────────────────────────────────
    coords = meta[[cell_id_col, 'CenterX_global_px', 'CenterY_global_px']].copy()
    if min_x is None:
        min_x = coords['CenterX_global_px'].min()
    if min_y is None:
        min_y = coords['CenterY_global_px'].min()
    coords['x_plot'] = coords['CenterX_global_px'] - min_x
    coords['y_plot'] = coords['CenterY_global_px'] - min_y

    merged = coords.merge(feat_df, on=cell_id_col)
    if merged.empty:
        raise ValueError("No overlapping cells between cell_meta and feature data")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if background_img is not None:
        h, w = background_img.shape[:2]
        ax.imshow(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), origin='upper')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    scatter = _render_dots_on_ax(
        ax, merged, 'x_plot', 'y_plot', feat_col,
        kind, color_lookup, norm, cmap,
        dot_size, edge_color, edge_width, alpha,
    )
    if background_img is not None:
        ax.invert_yaxis()
    ax.set_title(feat_name, fontsize=14)

    # ── Legend / colorbar ─────────────────────────────────────────────────────
    if kind == 'categorical':
        ax.legend(handles=legend_handles, loc=legend_loc,
                  bbox_to_anchor=legend_bbox, title=feat_name)
    else:
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=feat_name)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(' ', '_').lower()
        bg = '_with_bg' if background_img is not None else ''
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
    colorbar_position: str = 'right',
    # categorical params
    color_map: dict | None = None,
    # shared style
    dot_size: float = 20,
    edge_color: str = 'none',
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
    fov_id_col  = sj_obj.config.fov_id_col
    meta = sj_obj.cell_meta.reset_index()

    # ── Check required coordinate columns ─────────────────────────────────────
    for col in ('CenterX_local_px', 'CenterY_local_px'):
        if col not in meta.columns:
            raise ValueError(f"Missing required column in cell_meta: '{col}'")

    # ── FOV list ──────────────────────────────────────────────────────────────
    if fov_ids is None:
        fov_ids = sorted(meta[fov_id_col].unique().astype(str).tolist())
    fov_ids = [str(f) for f in fov_ids]

    # ── Feature data ──────────────────────────────────────────────────────────
    feat_df  = _get_feature_data(sj_obj, feature, feature_df, feature_column)
    feat_col = feature_column if (feature == "metadata" and feature_column) else feature

    # ── Auto-detect kind & build color mapping ────────────────────────────────
    kind = 'categorical' if not pd.api.types.is_numeric_dtype(feat_df[feat_col]) else 'continuous'
    color_lookup, legend_handles, norm, cmap = _build_color_mapping(
        feat_df[feat_col], kind=kind,
        colormap=colormap, color_map=color_map,
        vmin=vmin, vmax=vmax,
    )

    coords = meta[[cell_id_col, fov_id_col,
                   'CenterX_local_px', 'CenterY_local_px']].copy()
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
            ax.set_title(f'FOV {fov_id} (no data)', fontsize=title_fontsize)
            ax.axis('off')
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
                ax.imshow(np.flipud(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), origin='lower')

        _render_dots_on_ax(
            ax, fov_data, 'CenterX_local_px', 'CenterY_local_px', feat_col,
            kind, color_lookup, norm, cmap,
            dot_size, edge_color, edge_width, alpha,
        )


        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)

    # Hide unused axes
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')

    # ── Shared legend / colorbar ──────────────────────────────────────────────
    feat_name = feat_col
    if kind == 'categorical':
        fig.legend(handles=legend_handles, loc='center right',
                   bbox_to_anchor=(1.02, 0.5), title=feat_name, fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        if colorbar_position == 'right':
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(sm, cax=cbar_ax, label=feat_name)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        else:
            cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
            fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label=feat_name)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    plt.suptitle(feat_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # ── Save / show ───────────────────────────────────────────────────────────
    if filename is None:
        safe = feat_col.replace(' ', '_').lower()
        bg = '_with_bg' if background_img else ''
        filename = f"dots_{safe}_local_{kind}{bg}.png"

    return _finalize(fig, save_dir, filename, dpi, show_plot)

# =============================================================================
# SECTION A — GENE EXPRESSION HELPER
# =============================================================================

def _get_gene_expression(sj_obj, gene: str,
                         layer: str | None = 'log_normalized') -> pd.DataFrame:
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
        raise ValueError(
            f"Gene '{gene}' not found. "
            f"Check sj_obj.gene_index for available genes."
        )

    # Extract expression vector (n_cells,)
    if layer is None:
        expr = sj_obj.get_expression(gene_names=gene).flatten()
    else:
        if layer not in sj_obj.layers:
            raise ValueError(
                f"Layer '{layer}' not found. "
                f"Available layers: {list(sj_obj.layers.keys())}"
            )
        layer_data = sj_obj.get_layer(layer)
        gene_idx   = sj_obj._get_gene_indices([gene])[0]

        from scipy import sparse
        if sparse.issparse(layer_data):
            expr = layer_data[:, gene_idx].toarray().flatten()
        else:
            expr = layer_data[:, gene_idx].flatten()

    # Build DataFrame aligned to master cell index
    return pd.DataFrame({
        cell_id: sj_obj.cell_index.astype(str),
        gene:    expr,
    })

# =============================================================================
# SECTION B — GENE POLYGON FUNCTIONS
# =============================================================================

def plot_global_polygon_gene(
    sj_obj,
    gene: str,
    layer: str | None = 'log_normalized',
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
    layer_label = layer or 'raw'
    fig_title = f"{gene} ({layer_label})"

    return plot_global_polygon(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        background_img=background_img,
        min_x=min_x, min_y=min_y,
        colormap=colormap,
        vmin=vmin, vmax=vmax,
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
    layer: str | None = 'log_normalized',
    fov_ids: list[str | int] | None = None,
    background_img: bool = True,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = 'right',
    edge_color: str = 'none',
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
    layer_label = layer or 'raw'

    return plot_local_polygon(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        fov_ids=fov_ids,
        background_img=background_img,
        colormap=colormap,
        vmin=vmin, vmax=vmax,
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
    layer: str | None = 'log_normalized',
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
    layer_label = layer or 'raw'
    fig_title = f"{gene} ({layer_label})"

    return plot_global_dots(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        background_img=background_img,
        min_x=min_x, min_y=min_y,
        colormap=colormap,
        vmin=vmin, vmax=vmax,
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
    layer: str | None = 'log_normalized',
    fov_ids: list[str | int] | None = None,
    background_img: bool = True,
    colormap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = 'right',
    dot_size: float = 20,
    edge_color: str = 'none',
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
    layer_label = layer or 'raw'

    return plot_local_dots(
        sj_obj,
        feature=gene,
        feature_df=feat_df,
        fov_ids=fov_ids,
        background_img=background_img,
        colormap=colormap,
        vmin=vmin, vmax=vmax,
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
