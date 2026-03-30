Processing: ``sj.processing``
=============================

Normalization
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.normalize_total
   spatioloji_s.processing.log_transform
   spatioloji_s.processing.scale
   spatioloji_s.processing.normalize_pearson_residuals
   spatioloji_s.processing.normalize_standard_workflow
   spatioloji_s.processing.scale_by_batch_normalization

Feature selection
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.highly_variable_genes
   spatioloji_s.processing.compare_hvg_methods
   spatioloji_s.processing.select_genes_by_pattern

Dimensionality reduction
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.pca
   spatioloji_s.processing.umap
   spatioloji_s.processing.tsne
   spatioloji_s.processing.diffusion_map
   spatioloji_s.processing.plot_pca_variance

Clustering
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.leiden_clustering
   spatioloji_s.processing.leiden_resolution_sweep
   spatioloji_s.processing.kmeans_clustering
   spatioloji_s.processing.hierarchical_clustering
   spatioloji_s.processing.spatial_clustering
   spatioloji_s.processing.spatially_constrained_clustering
   spatioloji_s.processing.find_optimal_clusters
   spatioloji_s.processing.assess_clustering_quality

Batch correction
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.combat
   spatioloji_s.processing.harmony
   spatioloji_s.processing.regress_out
   spatioloji_s.processing.scale_by_batch
   spatioloji_s.processing.scvi_integrate
   spatioloji_s.processing.cca_integrate
   spatioloji_s.processing.rpca_integrate
   spatioloji_s.processing.evaluate_batch_correction

Differential expression
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.run_deg
   spatioloji_s.processing.deg_wilcoxon
   spatioloji_s.processing.deg_ttest
   spatioloji_s.processing.deg_mast
   spatioloji_s.processing.deg_nb_glm
   spatioloji_s.processing.deg_deseq2

Imputation
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.magic_impute
   spatioloji_s.processing.knn_smooth
   spatioloji_s.processing.alra_impute
   spatioloji_s.processing.scvi_impute
   spatioloji_s.processing.compare_imputation_methods

Gene set scoring
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.processing.load_gene_sets
   spatioloji_s.processing.make_gene_set_net
   spatioloji_s.processing.score_gene_sets
