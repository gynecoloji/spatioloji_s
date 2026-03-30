Basic Workflow
==============

This tutorial walks through the standard spatioloji_s analysis pipeline
using Xenium data as an example.

1. Loading data
---------------

.. code-block:: python

   import spatioloji_s as sj

   sp = sj.spatioloji(
       expression=expression_df,
       cell_meta=cell_meta_df,
       spatial_coords=spatial_df,
       polygons=polygon_dict,
   )
   print(sp)  # summary of the object

2. Quality control
------------------

.. code-block:: python

   from spatioloji_s.data import spatioloji_qc, QCConfig

   qc_config = QCConfig(
       min_counts=10,
       min_genes=5,
       max_pct_mito=20,
   )
   spatioloji_qc(sp, qc_config)

3. Normalization
----------------

.. code-block:: python

   sj.processing.normalize_total(sp)
   sj.processing.log_transform(sp)

4. Feature selection
--------------------

.. code-block:: python

   sj.processing.highly_variable_genes(sp, method="seurat_v3", n_top_genes=2000)

5. Dimensionality reduction
----------------------------

.. code-block:: python

   sj.processing.pca(sp, n_comps=30)
   sj.processing.umap(sp)

   # Visualize
   sj.visualization.plot_umap(sp, color_by="cell_type")

6. Clustering
-------------

.. code-block:: python

   sj.processing.leiden_clustering(sp, resolution=0.5)
   sj.visualization.plot_umap(sp, color_by="leiden")

7. Differential expression
--------------------------

.. code-block:: python

   deg_results = sj.processing.run_deg(sp, group_col="leiden", method="wilcoxon")

8. Visualization
----------------

.. code-block:: python

   # Marker gene expression
   sj.visualization.plot_dotplot(sp, genes=["EPCAM", "CD3D", "VIM"], group_by="leiden")
   sj.visualization.plot_violin(sp, genes=["EPCAM", "CD3D"], group_by="cell_type")

   # Spatial maps
   sj.visualization.plot_global_dots(sp, feature="cell_type")
   sj.visualization.plot_global_polygon(sp, feature="leiden")
