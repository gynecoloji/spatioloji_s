"""
visualization/__init__.py - Visualization subpackage for spatioloji

Provides static plotting methods for expression data, embeddings,
and spatial maps.

Usage
-----
    import spatioloji as sj
    sj.visualization.plot_umap(sp)
    sj.visualization.plot_global_dots(sp, gene="EPCAM")
"""

from .basic_plots import (
    plot_dotplot,
    plot_heatmap,
    plot_pca,
    plot_umap,
    plot_umap_by_clusters,
    plot_umap_by_feature,
    plot_umap_by_gene,
    plot_umap_grid,
    plot_violin,
)
from .plots import (
    plot_global_dots,
    plot_global_dots_gene,
    plot_global_polygon,
    plot_global_polygon_gene,
    plot_local_dots,
    plot_local_dots_gene,
    plot_local_polygon,
    plot_local_polygon_gene,
    stitch_fov_images,
)

__all__ = [
    # basic_plots
    "plot_dotplot",
    "plot_heatmap",
    "plot_pca",
    "plot_umap",
    "plot_umap_by_clusters",
    "plot_umap_by_feature",
    "plot_umap_by_gene",
    "plot_umap_grid",
    "plot_violin",
    # plots
    "plot_global_dots",
    "plot_global_dots_gene",
    "plot_global_polygon",
    "plot_global_polygon_gene",
    "plot_local_dots",
    "plot_local_dots_gene",
    "plot_local_polygon",
    "plot_local_polygon_gene",
    "stitch_fov_images",
]
