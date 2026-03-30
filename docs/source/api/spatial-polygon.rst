Spatial — Polygon: ``sj.spatial.polygon``
=========================================

Polygon/boundary-based spatial analysis. Uses full cell geometry for
biologically accurate topology.

Graph construction
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.PolygonSpatialGraph
   spatioloji_s.spatial.polygon.build_contact_graph
   spatioloji_s.spatial.polygon.build_buffer_graph
   spatioloji_s.spatial.polygon.build_knn_graph
   spatioloji_s.spatial.polygon.build_weighted_knn_graph

Cell morphology
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.compute_morphology
   spatioloji_s.spatial.polygon.compute_context_morphology
   spatioloji_s.spatial.polygon.classify_morphology
   spatioloji_s.spatial.polygon.morphology_by_group
   spatioloji_s.spatial.polygon.morphology_gene_correlation

Cell-cell contacts
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.contact_length
   spatioloji_s.spatial.polygon.contact_fraction
   spatioloji_s.spatial.polygon.free_boundary_fraction
   spatioloji_s.spatial.polygon.contact_summary

Neighborhood analysis
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.neighborhood_composition
   spatioloji_s.spatial.polygon.neighborhood_enrichment
   spatioloji_s.spatial.polygon.niche_identification
   spatioloji_s.spatial.polygon.boundary_cells

Spatial patterns
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.cell_density_map
   spatioloji_s.spatial.polygon.hotspot_detection
   spatioloji_s.spatial.polygon.spatial_autocorrelation
   spatioloji_s.spatial.polygon.colocalization

Statistical tests
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.contact_permutation_test
   spatioloji_s.spatial.polygon.morphology_association_test
   spatioloji_s.spatial.polygon.spatial_autocorrelation_test
   spatioloji_s.spatial.polygon.boundary_enrichment_test

Interface & gradient
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.identify_interface
   spatioloji_s.spatial.polygon.filter_interface
   spatioloji_s.spatial.polygon.compute_gradient
   spatioloji_s.spatial.polygon.score_infiltration

Spatial motifs
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.spatial.polygon.discover_motifs
   spatioloji_s.spatial.polygon.detect_assemblies
   spatioloji_s.spatial.polygon.match_known_structures
   spatioloji_s.spatial.polygon.run_motif_pipeline
