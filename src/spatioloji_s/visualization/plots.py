"""
Spatial transcriptomics plotting functions for spatioloji objects.

This module provides visualization functions for spatioloji data including:
- FOV image stitching
- Polygon-based plots (global and local)
- Dot-based plots (global and local)
- Support for both categorical and continuous features
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Union
import matplotlib as mpl
from matplotlib import colors, cm
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _calculate_global_offsets(spatioloji_obj):
    """
    Calculate global coordinate offsets from polygon data.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with polygon data
    
    Returns
    -------
    tuple
        (min_x, min_y) offset values
    """
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available for offset calculation")
    
    # Get global coordinate column names from config
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns('global')
    
    min_x = spatioloji_obj.polygons[x_col].min()
    min_y = spatioloji_obj.polygons[y_col].min()
    
    return min_x, min_y


def _get_feature_data(spatioloji_obj, feature, feature_df=None, feature_column=None):
    """
    Extract feature data from various sources.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object
    feature : str
        Feature name or "metadata"
    feature_df : pd.DataFrame, optional
        External feature dataframe
    feature_column : str, optional
        Column name when feature="metadata"
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'cell' and feature columns
    """
    cell_id_col = spatioloji_obj.config.cell_id_col
    
    if feature_df is not None:
        # Using external feature dataframe
        if cell_id_col not in feature_df.columns or feature not in feature_df.columns:
            raise ValueError(f"feature_df must contain '{cell_id_col}' and '{feature}' columns")
        return feature_df[[cell_id_col, feature]].copy()
    
    elif feature == "metadata" and feature_column is not None:
        # Using a column from cell_meta
        if feature_column not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{feature_column}' not found in cell_meta")
        
        feature_subset = spatioloji_obj.cell_meta.reset_index()[[cell_id_col, feature_column]].copy()
        feature_subset.rename(columns={feature_column: 'feature_value'}, inplace=True)
        return feature_subset
    
    else:
        # Using cell_meta directly
        if feature not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Feature '{feature}' not found in cell_meta")
        
        feature_subset = spatioloji_obj.cell_meta.reset_index()[[cell_id_col, feature]].copy()
        return feature_subset

# ============================================================================
# FOV IMAGE STITCHING
# ============================================================================

def stitch_fov_images(
    spatioloji_obj,
    fov_ids: Optional[List[Union[str, int]]] = None,
    flip_vertical: bool = True,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 300,
    verbose: bool = True
) -> Tuple[np.ndarray, float, float]:
    """
    Stitch multiple FOV images together based on their global positions.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing FOV images and position data
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to include. If None, all available FOVs are used
    flip_vertical : bool, optional
        Whether to flip images vertically before stitching, by default True
    save_path : str, optional
        Path to save the stitched image. If None, image is not saved
    show_plot : bool, optional
        Whether to display the stitched image, by default True
    title : str, optional
        Title for the plot. If None, a default title is used
    figsize : Tuple[int, int], optional
        Figure size for plotting, by default (12, 12)
    dpi : int, optional
        Resolution for saving the image, by default 300
    verbose : bool, optional
        Whether to print progress information, by default True
    
    Returns
    -------
    Tuple[np.ndarray, float, float]
        (stitched_image, min_x, min_y) - the stitched image and coordinate offsets
    
    Raises
    ------
    ValueError
        If required data is missing or no valid FOVs are found
    """
    # Check if spatioloji has required data
    if spatioloji_obj.fov_positions is None:
        raise ValueError("FOV positions data not found in spatioloji object")
    
    if spatioloji_obj.images.n_total == 0:
        raise ValueError("No images found in spatioloji object")
    
    # If no FOV IDs provided, use all available FOVs with images
    if fov_ids is None:
        fov_ids = spatioloji_obj.images.fov_ids
    
    if not fov_ids:
        raise ValueError("No FOV IDs provided")
    
    # Get FOV positions
    fov_positions = spatioloji_obj.fov_positions.copy()
    
    # Check required columns
    required_cols = ['x_global_px', 'y_global_px']
    missing_cols = [col for col in required_cols if col not in fov_positions.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in fov_positions: {missing_cols}")
    
    # Convert FOV IDs to strings for consistency
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Filter FOV positions to only include selected FOVs
    fov_positions = fov_positions[fov_positions.index.astype(str).isin(fov_ids)]
    
    if len(fov_positions) == 0:
        raise ValueError(f"None of the selected FOVs {fov_ids} have position information")
    
    if verbose:
        print(f"Stitching {len(fov_positions)} FOVs: {sorted(fov_positions.index.astype(str).tolist())}")
    
    # Get image dimensions from the first available image
    first_fov = str(fov_positions.index[0])
    first_image = spatioloji_obj.get_image(first_fov)
    
    if first_image is None:
        raise ValueError(f"Could not find image for FOV {first_fov}")
    
    image_height, image_width = first_image.shape[:2]
    if verbose:
        print(f"Detected FOV image dimensions: {image_width} x {image_height}")
    
    # Calculate offsets
    min_x = fov_positions["x_global_px"].min()
    min_y = fov_positions["y_global_px"].min()
    
    fov_positions["x_offset"] = fov_positions["x_global_px"] - min_x
    fov_positions["y_offset"] = fov_positions["y_global_px"] - min_y
    
    # Calculate stitched image dimensions
    max_x = int(np.ceil((fov_positions["x_offset"].max() + image_width)))
    max_y = int(np.ceil((fov_positions["y_offset"].max() + image_height)))
    
    if verbose:
        print(f"Creating stitched image with dimensions: {max_x} x {max_y}")
    
    # Create empty canvas for the stitched image
    stitched = np.zeros((max_y, max_x, 3), dtype=np.uint8)
    
    # Keep track of successfully added FOVs
    added_fovs = []
    
    # Stitch images
    for fov in fov_positions.index:
        try:
            # Get image from spatioloji object
            img = spatioloji_obj.get_image(str(fov))
            
            if img is None:
                if verbose:
                    print(f"Could not find image for FOV {fov}")
                continue
            
            # Flip image if required
            img = cv2.flip(img, 0)
            
            # Calculate position in the stitched image
            row = fov_positions.loc[fov]
            x_offset = int(row["x_offset"])
            y_offset = int(row["y_offset"])
            
            # Place image in the stitched canvas
            h, w = img.shape[:2]
            stitched[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            added_fovs.append(fov)
            if verbose:
                print(f"Added FOV {fov} at position ({x_offset}, {y_offset})")
            
        except Exception as e:
            if verbose:
                print(f"Error processing FOV {fov}: {e}")
            continue
    
    if not added_fovs:
        print("Warning: No FOVs were successfully added to the stitched image.")
        return None, None, None
    
    # Save stitched image if path is provided
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            plt.figure(figsize=figsize)
            if flip_vertical:
                display_img = cv2.flip(stitched, 0)
            else:
                display_img = stitched
            plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            if title:
                plt.title(title)
            else:
                plt.title(f"Stitched FOV Image ({len(added_fovs)} FOVs)")
                
            plt.tight_layout()
            
            if not save_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf')):
                save_path = save_path + 'stitched_fov.png'
                
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"Saved stitched image to {save_path}")
                
        except Exception as e:
            print(f"Error saving stitched image: {e}")
    
    # Display the stitched image
    if show_plot:
        plt.figure(figsize=figsize)
        if flip_vertical:
            display_img = cv2.flip(stitched, 0)
        else:
            display_img = stitched
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Stitched FOV Image ({len(added_fovs)} FOVs)")
            
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print(f"Stitched image created successfully with {len(added_fovs)} FOVs.")
    
    # Return stitched image and offsets
    return stitched, min_x, min_y

# ============================================================================
# GLOBAL POLYGON PLOTTING
# ============================================================================
def plot_global_polygon_by_features(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    save_dir: str = "./",
    colormap: str = "viridis",
    edge_color: str = "none",
    edge_width: float = 0.01,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0
) -> plt.Figure:
    """
    Plot cell polygons colored by a continuous feature value
    in **cv2-aligned global pixel coordinates**.
    
    Optimized version using PolyCollection for fast rendering.
    """

    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")

    cell_id_col = spatioloji_obj.config.cell_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns("global")

    polygon_df = spatioloji_obj.polygons.copy()

    for col in [cell_id_col, x_col, y_col]:
        if col not in polygon_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ─────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(
            spatioloji_obj, feature, feature_df, feature_column
        )
        feature_subset.rename(columns={"feature_value": feature_column}, inplace=True)
        feature = feature_column
        feature_name = feature_column
    else:
        feature_subset = _get_feature_data(
            spatioloji_obj, feature, feature_df, feature_column
        )
        feature_name = feature

    # ─────────────────────────────────────────────
    # Global offsets
    # ─────────────────────────────────────────────
    if min_x is None or min_y is None:
        min_x, min_y = _calculate_global_offsets(spatioloji_obj)

    polygon_df["x_plot"] = polygon_df[x_col] - min_x
    polygon_df["y_plot"] = polygon_df[y_col] - min_y

    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)

    if merged_df.empty:
        raise ValueError("No matching cells found between polygons and features")

    if not pd.api.types.is_numeric_dtype(merged_df[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric")

    # ─────────────────────────────────────────────
    # Color normalization
    # ─────────────────────────────────────────────
    vmin = merged_df[feature].min() if vmin is None else vmin
    vmax = merged_df[feature].max() if vmax is None else vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)

    # ─────────────────────────────────────────────
    # Pre-compute all polygons and colors (FAST!)
    # ─────────────────────────────────────────────
    print(f"Preparing {merged_df[cell_id_col].nunique()} cell polygons...")
    
    polygons = []
    polygon_colors = []
    
    for cell_id, group in merged_df.groupby(cell_id_col):
        # Get coordinates for this cell
        coords = group[["x_plot", "y_plot"]].values
        polygons.append(coords)
        
        # Get color for this cell
        value = group[feature].iloc[0]
        color = cmap(norm(value))
        polygon_colors.append(color)
    
    print(f"Creating PolyCollection...")

    # ─────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if background_img is not None:
        img_h, img_w = background_img.shape[:2]

        ax.imshow(
            cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB),
            origin="upper"   # 🔑 cv2-aligned
        )

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # 🔑 y increases downward
    else:
        ax.set_xlim(0, merged_df["x_plot"].max() * 1.05)
        ax.set_ylim(merged_df["y_plot"].max() * 1.05, 0)

    # ─────────────────────────────────────────────
    # Draw ALL polygons at once using PolyCollection (FAST!)
    # ─────────────────────────────────────────────
    poly_collection = PolyCollection(
        polygons,
        facecolors=polygon_colors,
        edgecolors=edge_color if edge_color != "none" else "none",
        linewidths=edge_width,
        alpha=alpha
    )
    ax.add_collection(poly_collection)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    # ─────────────────────────────────────────────
    # Colorbar
    # ─────────────────────────────────────────────
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(fig_title if fig_title else feature_name)

    plt.title(fig_title if fig_title else feature_name, fontsize=14)

    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        safe = feature_name.replace(" ", "_").lower()
        bg_suffix = "_with_bg" if background_img is not None else ""
        filename = f"polygon_{safe}_global{bg_suffix}.png"

    save_path = os.path.join(save_dir, filename)
    
    print(f"Saving figure...")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig

def plot_global_polygon_by_categorical(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    edge_color: str = "black",
    edge_width: float = 0.5,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    alpha: float = 0.8,
    legend_loc: str = 'center left',
    legend_bbox_to_anchor: Tuple[float, float] = (1.01, 0.5)
) -> plt.Figure:
    """
    Plot cell polygons colored by a categorical feature in global coordinates.
    
    Optimized version using PolyCollection for fast rendering.
    """
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")
    
    cell_id_col = spatioloji_obj.config.cell_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns("global")
    
    polygon_df = spatioloji_obj.polygons.copy()
    
    for col in [cell_id_col, x_col, y_col]:
        if col not in polygon_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Feature extraction
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(
            spatioloji_obj, feature, feature_df, feature_column
        )
        feature_subset.rename(columns={"feature_value": feature_column}, inplace=True)
        feature = feature_column
        feature_name = feature_column
    else:
        feature_subset = _get_feature_data(
            spatioloji_obj, feature, feature_df, feature_column
        )
        feature_name = feature
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:
            raise ValueError(
                f"Feature '{feature}' has {n_unique} unique values, "
                "too many for categorical plotting"
            )
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Global offsets
    if min_x is None or min_y is None:
        min_x, min_y = _calculate_global_offsets(spatioloji_obj)
    
    polygon_df["x_plot"] = polygon_df[x_col] - min_x
    polygon_df["y_plot"] = polygon_df[y_col] - min_y
    
    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)
    
    if merged_df.empty:
        raise ValueError("No matching cells found between polygons and features")
    
    # Get categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors
        color_map = {cat: color_palette[i % len(color_palette)] 
                     for i, cat in enumerate(categories)}
    
    # ─────────────────────────────────────────────
    # Pre-compute all polygons and colors (FAST!)
    # ─────────────────────────────────────────────
    print(f"Preparing {merged_df[cell_id_col].nunique()} cell polygons...")
    
    polygons = []
    polygon_colors = []
    categories_present = set()
    
    for cell_id, group in merged_df.groupby(cell_id_col):
        # Get coordinates for this cell
        coords = group[["x_plot", "y_plot"]].values
        polygons.append(coords)
        
        # Get color for this cell's category
        category = group[feature].iloc[0]
        categories_present.add(category)
        color = color_map[category]
        polygon_colors.append(color)
    
    print(f"Creating PolyCollection...")
    
    # ─────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    
    if background_img is not None:
        img_h, img_w = background_img.shape[:2]
        ax.imshow(
            cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB),
            origin="upper"
        )
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
    else:
        ax.set_xlim(0, merged_df["x_plot"].max() * 1.05)
        ax.set_ylim(merged_df["y_plot"].max() * 1.05, 0)
    
    # ─────────────────────────────────────────────
    # Draw ALL polygons at once using PolyCollection (FAST!)
    # ─────────────────────────────────────────────
    poly_collection = PolyCollection(
        polygons,
        facecolors=polygon_colors,
        edgecolors=edge_color,
        linewidths=edge_width,
        alpha=alpha
    )
    ax.add_collection(poly_collection)
    
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    
    # Add legend
    if len(categories_present) <= 20:
        legend_patches = [
            Patch(
                facecolor=color_map[cat],
                edgecolor=edge_color,
                alpha=alpha,
                label=str(cat)
            )
            for cat in sorted(categories_present)
        ]
        
        ax.legend(
            handles=legend_patches,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            title=feature_name,
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
        print(f"Warning: Too many categories ({len(categories_present)}) for legend.")
    
    plt.title(
        fig_title if fig_title else f"{feature_name} Distribution",
        fontsize=14
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        safe = feature_name.replace(" ", "_").lower()
        bg_suffix = "_with_bg" if background_img is not None else ""
        filename = f"polygon_{safe}_categorical{bg_suffix}.png"
    
    save_path = os.path.join(save_dir, filename)
    
    print(f"Saving figure...")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


# ============================================================================
# GLOBAL DOT PLOTTING
# ============================================================================

def plot_global_dot_by_features(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    save_dir: str = "./",
    colormap: str = "viridis",
    dot_size: int = 20,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0
) -> plt.Figure:
    """
    Plot cell dots colored by a continuous feature value in global coordinates.
    Uses global center coordinates (CenterX_global_px, CenterY_global_px).
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing cell metadata
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : np.ndarray, optional
        Stitched background image (from stitch_fov_images)
    min_x : float, optional
        X-offset for global coordinates (from stitch_fov_images)
    min_y : float, optional
        Y-offset for global coordinates (from stitch_fov_images)
    save_dir : str, optional
        Directory to save figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    dot_size : int, optional
        Size of dots in points squared, by default 20
    figsize : tuple, optional
        Figure size (width, height), by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    vmin : float, optional
        Minimum value for color scale, by default None
    vmax : float, optional
        Maximum value for color scale, by default None
    alpha : float, optional
        Dot transparency, by default 1.0
    edge_color : str, optional
        Dot edge color, by default None
    edge_width : float, optional
        Dot edge width, by default 0.0
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    
    # Create copy of cell metadata
    cell_meta = spatioloji_obj.cell_meta.reset_index().copy()
    
    # Check required columns for global center coordinates
    required_cols = [cell_id_col, 'CenterX_global_px', 'CenterY_global_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in cell_meta: {missing_cols}")
    
    cell_meta = cell_meta[required_cols]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Calculate offsets if not provided
    if min_x is None or min_y is None:
        # Calculate from cell centers
        min_x = cell_meta['CenterX_global_px'].min()
        min_y = cell_meta['CenterY_global_px'].min()
    
    # Apply offsets
    cell_meta['CenterX_offset'] = cell_meta['CenterX_global_px'] - min_x
    cell_meta['CenterY_offset'] = cell_meta['CenterY_global_px'] - min_y
    
    # Merge with feature data
    merged_df = cell_meta.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Check if feature is numeric
    if not pd.api.types.is_numeric_dtype(merged_df[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous plotting")
    
    # Normalize feature values for coloring
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display background image if provided
    if background_img is not None:
        ax.imshow(cv2.cvtColor(cv2.flip(background_img, 0), cv2.COLOR_BGR2RGB))
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Calculate plot limits
    max_x = merged_df['CenterX_offset'].max() * 1.05
    max_y = merged_df['CenterY_offset'].max() * 1.05
    
    # Create scatter plot
    print(f"Creating scatter plot for {len(merged_df)} cells...")
    
    scatter = ax.scatter(
        merged_df['CenterX_offset'],
        merged_df['CenterY_offset'],
        c=merged_df[feature],
        cmap=colormap,
        norm=norm,
        s=dot_size,
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edge_width
    )
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(feature_name)
    
    # Set title
    title = fig_title if fig_title is not None else feature_name
    plt.title(title, fontsize=14)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img is not None else ""
        filename = f"global_dot_{safe_name}{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_global_dot_by_categorical(
    spatioloji_obj,
    feature: str,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: Optional[np.ndarray] = None,
    min_x: Optional[float] = None,
    min_y: Optional[float] = None,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    dot_size: int = 20,
    figsize: tuple = (12, 12),
    fig_title: Optional[str] = None,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_width: float = 0.0
) -> plt.Figure:
    """
    Plot cell dots colored by a categorical feature in global coordinates.
    Uses global center coordinates (CenterX_global_px, CenterY_global_px).
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing cell metadata
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : np.ndarray, optional
        Stitched background image (from stitch_fov_images)
    min_x : float, optional
        X-offset for global coordinates (from stitch_fov_images)
    min_y : float, optional
        Y-offset for global coordinates (from stitch_fov_images)
    save_dir : str, optional
        Directory to save figure, by default "./"
    color_map : Dict[str, tuple], optional
        Category to color mapping, by default None (auto-assigned)
    dot_size : int, optional
        Size of dots in points squared, by default 20
    figsize : tuple, optional
        Figure size (width, height), by default (12, 12)
    fig_title : str, optional
        Figure title, by default None (uses feature name)
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the figure, by default True
    alpha : float, optional
        Dot transparency, by default 1.0
    edge_color : str, optional
        Dot edge color, by default None
    edge_width : float, optional
        Dot edge width, by default 0.0
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    
    # Create copy of cell metadata
    cell_meta = spatioloji_obj.cell_meta.reset_index().copy()
    
    # Check required columns for global center coordinates
    required_cols = [cell_id_col, 'CenterX_global_px', 'CenterY_global_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in cell_meta: {missing_cols}")
    
    cell_meta = cell_meta[required_cols]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, too many for categorical plotting")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Calculate offsets if not provided
    if min_x is None or min_y is None:
        min_x = cell_meta['CenterX_global_px'].min()
        min_y = cell_meta['CenterY_global_px'].min()
    
    # Apply offsets
    cell_meta['CenterX_offset'] = cell_meta['CenterX_global_px'] - min_x
    cell_meta['CenterY_offset'] = cell_meta['CenterY_global_px'] - min_y
    
    # Merge with feature data
    merged_df = cell_meta.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Get categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display background image if provided
    if background_img is not None:
        ax.imshow(cv2.cvtColor(cv2.flip(background_img, 0), cv2.COLOR_BGR2RGB))
    
    # Calculate plot limits
    max_x = merged_df['CenterX_offset'].max() * 1.05
    max_y = merged_df['CenterY_offset'].max() * 1.05
    
    # Track present categories
    categories_present = []
    
    # Plot each category separately for legend control
    for category in categories:
        category_cells = merged_df[merged_df[feature] == category]
        
        if len(category_cells) == 0:
            continue
        
        categories_present.append(category)
        
        ax.scatter(
            category_cells['CenterX_offset'],
            category_cells['CenterY_offset'],
            c=[color_map[category]],
            s=dot_size,
            alpha=alpha,
            edgecolors=edge_color,
            linewidths=edge_width,
            label=str(category)
        )
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.axis('off')
    
    # Add legend if not too many categories
    if len(categories_present) <= 20:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            title=feature_name,
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.tight_layout()
        print(f"Warning: Too many categories ({len(categories_present)}) to display in legend.")
    
    # Set title
    title = fig_title if fig_title is not None else f"{feature_name} Distribution"
    plt.title(title, fontsize=14)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img is not None else ""
        filename = f"global_dot_{safe_name}_categorical{bg_suffix}.png"
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

# ============================================================================
# LOCAL POLYGON PLOTTING
# ============================================================================

def plot_local_polygon_by_features(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    colormap: str = "viridis",
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_position: str = 'right'
) -> plt.Figure:
    """
    Create polygon plots colored by continuous features across multiple FOVs.
    Uses local coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing polygon data and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Polygon edge color, by default 'black'
    edge_width : float, optional
        Polygon edge width, by default 0.5
    alpha : float, optional
        Polygon transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    vmin : float, optional
        Minimum value for color scale, by default None
    vmax : float, optional
        Maximum value for color scale, by default None
    colorbar_position : str, optional
        Position of colorbar ('right' or 'bottom'), by default 'right'
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if polygon data exists
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")
    
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns('local')
    
    # Create copy of polygon data
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns
    required_cols = [cell_id_col, fov_id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            # Fall back to all FOVs in polygon_df
            fov_ids = sorted(polygon_df[fov_id_col].unique())
    
    # Convert FOV IDs to strings for consistency
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is numeric
    if not pd.api.types.is_numeric_dtype(feature_subset[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous plotting")
    
    # Merge with feature data
    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Determine global color normalization
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)  # Flip vertically
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data[x_col].max() * 1.1
            image_height = fov_data[y_col].max() * 1.1
        
        # Group cells by cell ID
        cell_groups = fov_data.groupby(cell_id_col)
        
        # Process each cell
        for cell_id, cell_vertices in cell_groups:
            # Get feature value for this cell
            feature_value = cell_vertices[feature].iloc[0]
            
            # Get color for this value
            color = cmap(norm(feature_value))
            
            # Skip cells with fewer than 3 vertices
            if len(cell_vertices) < 3:
                print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                continue
            
            # Create polygon coordinates
            coords = cell_vertices[[x_col, y_col]].values
            
            # Create and add polygon
            polygon = mpl.patches.Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha
            )
            ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add colorbar to the figure
    if colorbar_position == 'right':
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax)
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    else:
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax, orientation='horizontal')
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Set title
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{safe_name}_continuous{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_polygon_by_categorical(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create polygon plots colored by categorical features across multiple FOVs.
    Uses local coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing polygon data and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    color_map : Dict[str, tuple], optional
        Category to color mapping, by default None (auto-assigned)
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Polygon edge color, by default 'black'
    edge_width : float, optional
        Polygon edge width, by default 0.5
    alpha : float, optional
        Polygon transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if polygon data exists
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")
    
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns('local')
    
    # Create copy of polygon data
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns
    required_cols = [cell_id_col, fov_id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            fov_ids = sorted(polygon_df[fov_id_col].unique())
    
    # Convert FOV IDs to strings
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, too many for categorical plotting")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Merge with feature data
    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Get categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Track present categories
    categories_present = set()
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data[x_col].max() * 1.1
            image_height = fov_data[y_col].max() * 1.1
        
        # Group cells by category for this FOV
        cell_categories = {}
        
        for category in categories:
            category_cells = fov_data[fov_data[feature] == category][cell_id_col].unique()
            
            if len(category_cells) == 0:
                continue
            
            categories_present.add(category)
            cell_categories[category] = category_cells
        
        # Process all cells by category
        for category, cells in cell_categories.items():
            category_color = color_map[category]
            
            for cell_id in cells:
                cell_vertices = fov_data[fov_data[cell_id_col] == cell_id]
                
                if len(cell_vertices) < 3:
                    print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                    continue
                
                coords = cell_vertices[[x_col, y_col]].values
                
                polygon = mpl.patches.Polygon(
                    coords,
                    closed=True,
                    facecolor=category_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=alpha
                )
                ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add legend for all present categories
    legend_patches = [Patch(facecolor=color_map[cat],
                           edgecolor=edge_color,
                           alpha=alpha,
                           label=str(cat))
                     for cat in sorted(categories_present)]
    
    fig.legend(
        handles=legend_patches,
        loc='center right',
        bbox_to_anchor=(1.02, 0.5),
        title=feature_name,
        fontsize=20
    )
    
    # Set title and adjust layout
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{safe_name}_categorical{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

# ============================================================================
# LOCAL POLYGON PLOTTING
# ============================================================================

def plot_local_polygon_by_features(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    colormap: str = "viridis",
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_position: str = 'right'
) -> plt.Figure:
    """
    Create polygon plots colored by continuous features across multiple FOVs.
    Uses local coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing polygon data and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Polygon edge color, by default 'black'
    edge_width : float, optional
        Polygon edge width, by default 0.5
    alpha : float, optional
        Polygon transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    vmin : float, optional
        Minimum value for color scale, by default None
    vmax : float, optional
        Maximum value for color scale, by default None
    colorbar_position : str, optional
        Position of colorbar ('right' or 'bottom'), by default 'right'
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if polygon data exists
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")
    
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns('local')
    
    # Create copy of polygon data
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns
    required_cols = [cell_id_col, fov_id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            # Fall back to all FOVs in polygon_df
            fov_ids = sorted(polygon_df[fov_id_col].unique())
    
    # Convert FOV IDs to strings for consistency
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is numeric
    if not pd.api.types.is_numeric_dtype(feature_subset[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous plotting")
    
    # Merge with feature data
    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Determine global color normalization
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)  # Flip vertically
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data[x_col].max() * 1.1
            image_height = fov_data[y_col].max() * 1.1
        
        # Group cells by cell ID
        cell_groups = fov_data.groupby(cell_id_col)
        
        # Process each cell
        for cell_id, cell_vertices in cell_groups:
            # Get feature value for this cell
            feature_value = cell_vertices[feature].iloc[0]
            
            # Get color for this value
            color = cmap(norm(feature_value))
            
            # Skip cells with fewer than 3 vertices
            if len(cell_vertices) < 3:
                print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                continue
            
            # Create polygon coordinates
            coords = cell_vertices[[x_col, y_col]].values
            
            # Create and add polygon
            polygon = mpl.patches.Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=alpha
            )
            ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add colorbar to the figure
    if colorbar_position == 'right':
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax)
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    else:
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax, orientation='horizontal')
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Set title
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{safe_name}_continuous{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_polygon_by_categorical(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create polygon plots colored by categorical features across multiple FOVs.
    Uses local coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing polygon data and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    color_map : Dict[str, tuple], optional
        Category to color mapping, by default None (auto-assigned)
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    edge_color : str, optional
        Polygon edge color, by default 'black'
    edge_width : float, optional
        Polygon edge width, by default 0.5
    alpha : float, optional
        Polygon transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if polygon data exists
    if spatioloji_obj.polygons is None:
        raise ValueError("No polygon data available in spatioloji object")
    
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    x_col, y_col = spatioloji_obj.config.get_coordinate_columns('local')
    
    # Create copy of polygon data
    polygon_df = spatioloji_obj.polygons.copy()
    
    # Check required columns
    required_cols = [cell_id_col, fov_id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in polygon_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in polygons: {missing_cols}")
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            fov_ids = sorted(polygon_df[fov_id_col].unique())
    
    # Convert FOV IDs to strings
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, too many for categorical plotting")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Merge with feature data
    merged_df = polygon_df.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between polygon data and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Get categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Track present categories
    categories_present = set()
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data[x_col].max() * 1.1
            image_height = fov_data[y_col].max() * 1.1
        
        # Group cells by category for this FOV
        cell_categories = {}
        
        for category in categories:
            category_cells = fov_data[fov_data[feature] == category][cell_id_col].unique()
            
            if len(category_cells) == 0:
                continue
            
            categories_present.add(category)
            cell_categories[category] = category_cells
        
        # Process all cells by category
        for category, cells in cell_categories.items():
            category_color = color_map[category]
            
            for cell_id in cells:
                cell_vertices = fov_data[fov_data[cell_id_col] == cell_id]
                
                if len(cell_vertices) < 3:
                    print(f"Warning: Cell {cell_id} in FOV {fov_id} has fewer than 3 vertices. Skipping.")
                    continue
                
                coords = cell_vertices[[x_col, y_col]].values
                
                polygon = mpl.patches.Polygon(
                    coords,
                    closed=True,
                    facecolor=category_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    alpha=alpha
                )
                ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add legend for all present categories
    legend_patches = [Patch(facecolor=color_map[cat],
                           edgecolor=edge_color,
                           alpha=alpha,
                           label=str(cat))
                     for cat in sorted(categories_present)]
    
    fig.legend(
        handles=legend_patches,
        loc='center right',
        bbox_to_anchor=(1.02, 0.5),
        title=feature_name,
        fontsize=20
    )
    
    # Set title and adjust layout
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_polygon_{safe_name}_categorical{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

# ============================================================================
# LOCAL DOT PLOTTING
# ============================================================================

def plot_local_dots_by_features(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    colormap: str = "viridis",
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    dot_size: float = 20,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_position: str = 'right'
) -> plt.Figure:
    """
    Create dot plots colored by continuous features across multiple FOVs.
    Uses local center coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing cell metadata and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    colormap : str, optional
        Matplotlib colormap name, by default "viridis"
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    dot_size : float, optional
        Size of dots representing cells, by default 20
    edge_color : str, optional
        Dot edge color, by default 'black'
    edge_width : float, optional
        Dot edge width, by default 0.5
    alpha : float, optional
        Dot transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    vmin : float, optional
        Minimum value for color scale, by default None
    vmax : float, optional
        Maximum value for color scale, by default None
    colorbar_position : str, optional
        Position of colorbar ('right' or 'bottom'), by default 'right'
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    
    # Create copy of cell metadata
    cell_meta = spatioloji_obj.cell_meta.reset_index().copy()
    
    # Check required columns for local center coordinates
    required_cols = [cell_id_col, fov_id_col, 'CenterX_local_px', 'CenterY_local_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in cell_meta: {missing_cols}")
    
    cell_meta = cell_meta[required_cols]
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            fov_ids = sorted(cell_meta[fov_id_col].unique())
    
    # Convert FOV IDs to strings
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is numeric
    if not pd.api.types.is_numeric_dtype(feature_subset[feature]):
        raise ValueError(f"Feature '{feature}' must be numeric for continuous plotting")
    
    # Merge with feature data
    merged_df = cell_meta.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Determine global color normalization
    if vmin is None:
        vmin = merged_df[feature].min()
    if vmax is None:
        vmax = merged_df[feature].max()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)  # Flip vertically
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data['CenterX_local_px'].max() * 1.1
            image_height = fov_data['CenterY_local_px'].max() * 1.1
        
        # Create scatter plot with feature-based coloring
        scatter = ax.scatter(
            fov_data['CenterX_local_px'],
            fov_data['CenterY_local_px'],
            s=dot_size,
            c=fov_data[feature],
            cmap=colormap,
            norm=norm,
            edgecolors=edge_color,
            linewidths=edge_width,
            alpha=alpha
        )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add colorbar to the figure
    if colorbar_position == 'right':
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax)
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    else:
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cbar_ax, orientation='horizontal')
        cbar.set_label(feature_name)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Set title
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_dots_{safe_name}_continuous{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_local_dots_by_categorical(
    spatioloji_obj,
    feature: str,
    fov_ids: Optional[List[Union[str, int]]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    feature_column: Optional[str] = None,
    background_img: bool = True,
    save_dir: str = "./",
    color_map: Optional[Dict[str, tuple]] = None,
    figure_width: int = 7,
    figure_height: int = 7,
    grid_layout: Optional[Tuple[int, int]] = None,
    dot_size: float = 20,
    edge_color: str = 'black',
    edge_width: float = 0.5,
    alpha: float = 0.8,
    title_fontsize: int = 20,
    suptitle_fontsize: int = 24,
    filename: Optional[str] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create dot plots colored by categorical features across multiple FOVs.
    Uses local center coordinates to show cells in each FOV.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object containing cell metadata and FOV images
    feature : str
        Feature name to visualize, or "metadata" to use feature_column
    fov_ids : List[Union[str, int]], optional
        List of FOV IDs to plot. If None, all FOVs with images are used
    feature_df : pd.DataFrame, optional
        External DataFrame with 'cell' column and feature column
    feature_column : str, optional
        Column name from cell_meta when feature="metadata"
    background_img : bool, optional
        Whether to display FOV images as background, by default True
    save_dir : str, optional
        Directory to save figure, by default "./"
    color_map : Dict[str, tuple], optional
        Category to color mapping, by default None (auto-assigned)
    figure_width : int, optional
        Width of each subplot in inches, by default 7
    figure_height : int, optional
        Height of each subplot in inches, by default 7
    grid_layout : Tuple[int, int], optional
        Custom grid layout (rows, columns), by default None (auto-determined)
    dot_size : float, optional
        Size of dots representing cells, by default 20
    edge_color : str, optional
        Dot edge color, by default 'black'
    edge_width : float, optional
        Dot edge width, by default 0.5
    alpha : float, optional
        Dot transparency, by default 0.8
    title_fontsize : int, optional
        Font size for FOV titles, by default 20
    suptitle_fontsize : int, optional
        Font size for main title, by default 24
    filename : str, optional
        Custom filename, by default None (auto-generated)
    dpi : int, optional
        Resolution for saved figure, by default 300
    show_plot : bool, optional
        Whether to display the plot, by default True
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Get configuration
    cell_id_col = spatioloji_obj.config.cell_id_col
    fov_id_col = spatioloji_obj.config.fov_id_col
    
    # Create copy of cell metadata
    cell_meta = spatioloji_obj.cell_meta.reset_index().copy()
    
    # Check required columns for local center coordinates
    required_cols = [cell_id_col, fov_id_col, 'CenterX_local_px', 'CenterY_local_px']
    missing_cols = [col for col in required_cols if col not in cell_meta.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in cell_meta: {missing_cols}")
    
    cell_meta = cell_meta[required_cols]
    
    # If no FOV IDs provided, use all FOVs with images
    if fov_ids is None:
        if spatioloji_obj.images.n_total > 0:
            fov_ids = spatioloji_obj.images.fov_ids
        else:
            fov_ids = sorted(cell_meta[fov_id_col].unique())
    
    # Convert FOV IDs to strings
    fov_ids = [str(fid) for fid in fov_ids]
    
    # Get feature data
    if feature == "metadata" and feature_column is not None:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature_column
        feature_subset.rename(columns={'feature_value': feature_column}, inplace=True)
        feature = feature_column
    else:
        feature_subset = _get_feature_data(spatioloji_obj, feature, feature_df, feature_column)
        feature_name = feature
    
    # Check if feature is categorical or convert it
    if not pd.api.types.is_categorical_dtype(feature_subset[feature]):
        n_unique = feature_subset[feature].nunique()
        
        if n_unique > 20:
            raise ValueError(f"Feature '{feature}' has {n_unique} unique values, too many for categorical plotting")
        
        print(f"Converting '{feature}' to categorical with {n_unique} categories.")
        feature_subset[feature] = feature_subset[feature].astype('category')
    
    # Merge with feature data
    merged_df = cell_meta.merge(feature_subset, on=cell_id_col)
    
    if len(merged_df) == 0:
        raise ValueError("No matching cells found between cell_meta and feature values")
    
    # Filter for selected FOVs
    merged_df = merged_df[merged_df[fov_id_col].astype(str).isin(fov_ids)]
    
    if len(merged_df) == 0:
        raise ValueError(f"No cells found in the selected FOVs: {fov_ids}")
    
    # Get categories
    categories = feature_subset[feature].cat.categories
    
    # Create color map if not provided
    if color_map is None:
        color_palette = plt.cm.tab10.colors
        color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(categories)}
    
    # Determine grid layout
    if grid_layout is None:
        n_plots = len(fov_ids)
        grid_size = int(np.ceil(np.sqrt(n_plots)))
        rows, cols = grid_size, grid_size
    else:
        rows, cols = grid_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=(figure_width*cols, figure_height*rows))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    # Track present categories
    categories_present = set()
    
    # Plot each FOV
    for idx, fov_id in enumerate(fov_ids):
        if idx >= len(axs):
            print(f"Warning: Not enough subplots for FOV {fov_id}. Increase grid size.")
            break
        
        ax = axs[idx]
        
        # Display FOV image as background if requested
        image_width = None
        image_height = None
        
        if background_img:
            img = spatioloji_obj.get_image(str(fov_id))
            if img is not None:
                img = cv2.flip(img, 0)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_height, image_width = img.shape[:2]
            else:
                print(f"Warning: No image found for FOV {fov_id}")
        
        # Get cells in this FOV
        fov_data = merged_df[merged_df[fov_id_col].astype(str) == str(fov_id)]
        
        if len(fov_data) == 0:
            print(f"Warning: No cells found for FOV {fov_id}")
            ax.text(0.5, 0.5, f"No data for FOV {fov_id}", ha='center', va='center')
            if image_width and image_height:
                ax.set_xlim(0, image_width)
                ax.set_ylim(0, image_height)
            ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
            ax.axis('off')
            continue
        
        # If no image dimensions, determine from cell positions
        if not image_width or not image_height:
            image_width = fov_data['CenterX_local_px'].max() * 1.1
            image_height = fov_data['CenterY_local_px'].max() * 1.1
        
        # Plot dots for each category
        for category in categories:
            category_cells = fov_data[fov_data[feature] == category]
            
            if len(category_cells) == 0:
                continue
            
            categories_present.add(category)
            
            # Plot dots for this category
            ax.scatter(
                category_cells['CenterX_local_px'],
                category_cells['CenterY_local_px'],
                s=dot_size,
                c=[color_map[category]],
                edgecolors=edge_color,
                linewidths=edge_width,
                alpha=alpha,
                label=str(category)
            )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_title(f'FOV {fov_id}', fontsize=title_fontsize)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(len(fov_ids), len(axs)):
        axs[j].axis('off')
    
    # Add legend for all present categories
    legend_patches = [Patch(facecolor=color_map[cat],
                           edgecolor=edge_color,
                           alpha=alpha,
                           label=str(cat))
                     for cat in sorted(categories_present)]
    
    fig.legend(
        handles=legend_patches,
        loc='center right',
        bbox_to_anchor=(1.02, 0.5),
        title=feature_name,
        fontsize=20
    )
    
    # Set title and adjust layout
    plt.suptitle(feature_name, fontsize=suptitle_fontsize)
    plt.tight_layout(rect=[0, 0, 0.95, 0.98])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        safe_name = feature_name.replace(' ', '_').lower()
        bg_suffix = "_with_bg" if background_img else ""
        filename = f'local_dots_{safe_name}_categorical{bg_suffix}.png'
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig