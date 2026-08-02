"""
_spatial_helpers.py — Private shared helpers for point_plots and polygon_plots.

Not part of the public API.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm


def finalize_plot(fig: plt.Figure, save_path: str | None, dpi: int, show: bool) -> plt.Figure:
    """Shared save / show / close logic.

    Always returns ``fig``, so the ~50 public plotting functions that delegate
    here honour their documented ``Returns: matplotlib Figure`` contract on both
    the interactive and the scripted path.

    When ``show`` is False the figure is closed to drop it from pyplot's global
    registry — this matters when plotting in a loop — but the returned object
    remains usable for ``fig.savefig(...)`` or a later ``display(fig)``.

    Args:
        fig: The figure to finalize.
        save_path: If given, write the figure here before showing or closing.
        dpi: Resolution used when saving.
        show: If True, call ``plt.show()``; otherwise close the figure.

    Returns:
        The same ``fig`` that was passed in.
    """
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def clean_axes(ax: plt.Axes) -> None:
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def categorical_colors(
    values: pd.Series,
    palette: str | list | dict | None,
) -> tuple[list, dict, list]:
    """Map a categorical Series to colours.

    Returns:
        Tuple of (color_list, color_dict, legend_handles).
    """
    cats: list = sorted(values.dropna().unique().tolist())
    if isinstance(palette, dict):
        color_dict = {k: palette.get(k, "lightgrey") for k in cats}
    else:
        pal = sns.color_palette(palette or "tab20", len(cats))
        color_dict = dict(zip(cats, pal, strict=False))
    colors = [color_dict.get(v, "lightgrey") for v in values]
    handles = [mpatches.Patch(color=c, label=str(k)) for k, c in color_dict.items()]
    return colors, color_dict, handles


def continuous_colors(
    values: pd.Series,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    vcenter: float | None,
) -> tuple[list, Normalize, ScalarMappable]:
    """Map a continuous Series to colours.

    Returns:
        Tuple of (color_list, norm, ScalarMappable).
    """
    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax
    norm: Normalize
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    colors = [cm(norm(v)) if pd.notna(v) else (0.85, 0.85, 0.85, 1.0) for v in values]
    mappable = ScalarMappable(norm=norm, cmap=cm)
    mappable.set_array([])
    return colors, norm, mappable
