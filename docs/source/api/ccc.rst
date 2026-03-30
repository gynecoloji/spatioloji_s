Cell-Cell Communication: ``sj.ccc``
===================================

Polygon-native cell-cell communication analysis with edge-level scoring,
significance testing, and interface-aware stratification.

Pipeline
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.ccc.run_ccc
   spatioloji_s.ccc.CCCConfig
   spatioloji_s.ccc.CCCResult

Database
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.ccc.LRPair
   spatioloji_s.ccc.load_lr_database
   spatioloji_s.ccc.load_from_cellchatdb_csv
   spatioloji_s.ccc.filter_to_expressed
   spatioloji_s.ccc.lr_pairs_to_dataframe

Scoring
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.ccc.score_edges
   spatioloji_s.ccc.aggregate_scores
   spatioloji_s.ccc.test_significance

Zone & gradient analysis
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.ccc.compare_zones
   spatioloji_s.ccc.communication_gradient
   spatioloji_s.ccc.compare_morphology
