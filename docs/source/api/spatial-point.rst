Spatial — Point: ``sj.spatial.point``
=====================================

Centroid-based spatial analysis. Fast and scalable for large datasets.

Graph construction
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.PointSpatialGraph
   spatioloji_s.spatial.point.build_knn_graph
   spatioloji_s.spatial.point.build_weighted_knn_graph
   spatioloji_s.spatial.point.build_radius_graph
   spatioloji_s.spatial.point.build_delaunay_graph

Neighborhood analysis
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.neighborhood_composition
   spatioloji_s.spatial.point.neighborhood_enrichment
   spatioloji_s.spatial.point.identify_niches
   spatioloji_s.spatial.point.neighborhood_diversity

Spatial patterns
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.morans_i
   spatioloji_s.spatial.point.getis_ord_gi
   spatioloji_s.spatial.point.co_occurrence
   spatioloji_s.spatial.point.spatially_variable_genes

Ripley's statistics
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.RipleyResult
   spatioloji_s.spatial.point.ripleys_k
   spatioloji_s.spatial.point.ripleys_l
   spatioloji_s.spatial.point.cross_k
   spatioloji_s.spatial.point.cross_l
   spatioloji_s.spatial.point.simulation_envelope

Distance statistics
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.nearest_neighbor_distances
   spatioloji_s.spatial.point.cross_type_distances
   spatioloji_s.spatial.point.proximity_score
   spatioloji_s.spatial.point.permutation_test

Interface & gradient
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.identify_interface
   spatioloji_s.spatial.point.filter_interface
   spatioloji_s.spatial.point.compute_gradient
   spatioloji_s.spatial.point.score_infiltration

Spatial motifs
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.point.discover_motifs
   spatioloji_s.spatial.point.detect_assemblies
   spatioloji_s.spatial.point.match_known_structures
   spatioloji_s.spatial.point.run_motif_pipeline
