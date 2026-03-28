# Visualization — `spatioloji_s.visualization`

40+ plotting functions for embeddings, spatial maps, and analysis results. All static plots return `matplotlib.Figure` and support `show`, `save_path`, `dpi`, and `figsize` parameters.

## Embedding plots

```python
from spatioloji_s.visualization import plot_umap, plot_pca, plot_violin, plot_heatmap, plot_dotplot

plot_umap(sp, color_by="leiden")
plot_umap_by_gene(sp, "MKI67")
plot_violin(sp, genes=["MKI67", "VIM"], group_by="cell_type")
plot_heatmap(sp, genes=top_genes, group_by="cell_type")
plot_dotplot(sp, genes=markers, group_by="leiden")
```

```{eval-rst}
.. automodule:: spatioloji_s.visualization.basic_plots
   :members:
```

## Spatial maps

Dot (scatter) and polygon (cell boundary) rendering for global and per-FOV views.

```python
from spatioloji_s.visualization import (
    plot_global_dots, plot_global_polygon,
    plot_global_dots_gene, plot_global_polygon_gene,
    plot_local_dots, plot_local_polygon,
)

plot_global_polygon(sp, color_by="cell_type")
plot_global_dots_gene(sp, "EPCAM")
plot_local_polygon(sp, fov_id="fov_1", color_by="leiden")
```

```{eval-rst}
.. automodule:: spatioloji_s.visualization.plots
   :members:
```

## Point-based analysis plots

Plots for neighborhood analysis, Ripley's statistics, spatial patterns, and interface analysis.

```python
from spatioloji_s.visualization import (
    plot_spatial_graph, plot_neighborhood_enrichment,
    plot_niches, plot_morans_i_map, plot_ripley,
    plot_interface_point_map,
)
```

```{eval-rst}
.. automodule:: spatioloji_s.visualization.point_plots
   :members:
```

## Polygon-based analysis plots

Plots for morphology, contact analysis, interface, gradient, infiltration, and motif analysis.

```python
from spatioloji_s.visualization import (
    # Morphology
    plot_morphology_distribution, plot_morphology_map,
    # Contact
    plot_contact_summary, plot_free_boundary_map,
    # Interface & gradient
    plot_interface_polygon_map, plot_gradient_curve, plot_spatial_distance,
    # Infiltration
    plot_infiltration_summary,
    # Motifs
    plot_motif_map, plot_motif_composition, plot_assembly_map, plot_structure_matches,
)
```

```{eval-rst}
.. automodule:: spatioloji_s.visualization.polygon_plots
   :members:
```

## Interactive plots (Plotly)

All interactive plots return `plotly.graph_objects.Figure` and can be displayed in Jupyter notebooks.

```python
from spatioloji_s.visualization import (
    iplot_umap, iplot_global_dots, iplot_global_polygon,
    iplot_global_dots_gene, iplot_global_polygon_gene,
)

iplot_umap(sp, color_by="leiden")
iplot_global_polygon(sp, color_by="cell_type")
iplot_global_polygon_gene(sp, "EPCAM")
```

```{eval-rst}
.. automodule:: spatioloji_s.visualization.interactive_plots
   :members:
```
