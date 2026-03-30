Usage Principles
================

Import convention
-----------------

.. code-block:: python

   import spatioloji_s as sj

The central data object
-----------------------

The ``spatioloji`` object is the core data container. All functions take it as their
first argument and modify it in-place (storing results in ``cell_meta``, ``gene_meta``,
``embeddings``, or ``layers``).

.. code-block:: python

   sp = sj.spatioloji(
       expression=expression_df,       # genes x cells
       cell_meta=cell_metadata_df,     # per-cell annotations
       spatial_coords=spatial_df,      # x, y coordinates
       polygons=polygon_dict,          # cell boundary polygons (optional)
   )

Module organization
-------------------

spatioloji_s follows a modular design similar to scanpy:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Module
     - Access
     - Purpose
   * - Data
     - ``sj.data``
     - Loading, QC, export, configuration
   * - Processing
     - ``sj.processing``
     - Normalization, HVG, PCA/UMAP, clustering, DEG, batch correction
   * - Spatial (point)
     - ``sj.spatial.point``
     - Centroid-based graphs, neighborhoods, Moran's I, Ripley's K/L
   * - Spatial (polygon)
     - ``sj.spatial.polygon``
     - Polygon adjacency, morphology, contact analysis, boundaries
   * - CCC
     - ``sj.ccc``
     - Ligand-receptor database, edge scoring, significance testing
   * - Visualization
     - ``sj.visualization``
     - 80+ plotting functions for all analysis results

Two spatial modes
-----------------

spatioloji_s provides **two complementary spatial analysis modes**:

**Point-based** (``sj.spatial.point``):
   Uses cell centroids. Fast, scalable to millions of cells.
   Best for neighborhood composition, spatial autocorrelation, Ripley's statistics.

**Polygon-based** (``sj.spatial.polygon``):
   Uses full cell boundary geometry. More biologically accurate.
   Best for contact analysis, morphology, cell-cell communication.

Both modes share the same interface for gradient, infiltration, and motif analysis.

Expression layers
-----------------

Processing steps store results as named layers:

.. code-block:: python

   sj.processing.normalize_total(sp)       # stores 'normalized'
   sj.processing.log_transform(sp)         # stores 'log_normalized'
   sj.processing.scale(sp)                 # stores 'scaled'

   # Access a specific layer
   sp.get_layer("log_normalized")

Typical workflow
----------------

.. code-block:: python

   import spatioloji_s as sj

   # 1. Load data
   sp = sj.spatioloji(...)

   # 2. Quality control
   sj.data.spatioloji_qc(sp, sj.data.QCConfig())

   # 3. Normalize and transform
   sj.processing.normalize_total(sp)
   sj.processing.log_transform(sp)

   # 4. Feature selection and dimensionality reduction
   sj.processing.highly_variable_genes(sp)
   sj.processing.pca(sp)
   sj.processing.umap(sp)

   # 5. Clustering
   sj.processing.leiden_clustering(sp)

   # 6. Spatial analysis
   from spatioloji_s.spatial.polygon import build_buffer_graph, neighborhood_enrichment
   graph = build_buffer_graph(sp, buffer_distance=15)
   neighborhood_enrichment(sp, graph, "cell_type")

   # 7. Cell-cell communication
   from spatioloji_s.ccc import CCCConfig, run_ccc
   result = run_ccc(sp, CCCConfig(group_col="cell_type"))

   # 8. Visualize
   sj.visualization.plot_umap(sp, color_by="cell_type")
   sj.visualization.plot_ccc_heatmap(result)
