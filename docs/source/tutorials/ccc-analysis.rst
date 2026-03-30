Cell-Cell Communication Analysis
================================

spatioloji_s uses polygon geometry for biologically accurate CCC inference,
with edge-level scoring and interface-aware stratification.

Basic CCC pipeline
------------------

.. code-block:: python

   from spatioloji_s.ccc import CCCConfig, run_ccc

   config = CCCConfig(
       group_col="cell_type",
       layer="log_normalized",
       db_source="builtin",        # 50 curated LR pairs
       test_method="analytical",    # fast z-score test
   )
   result = run_ccc(sp, config)

   # Significant interactions
   sig = result.scores[result.scores["fdr"] < 0.05]
   print(sig.sort_values("mean_score", ascending=False).head(10))

Visualizing results
-------------------

.. code-block:: python

   import spatioloji_s as sj

   # Heatmap of interaction scores
   sj.visualization.plot_ccc_heatmap(result)

   # Top interactions as dot plot
   sj.visualization.plot_ccc_dotplot(result, top_n=20)

   # Network diagram
   sj.visualization.plot_ccc_network(result)

   # Spatial map of a specific LR pair
   sj.visualization.plot_ccc_spatial(sp, result, lr_name="TGFB1_TGFBR1|TGFBR2")

Using CellChatDB
-----------------

Load the full CellChatDB database (3,234 interactions) instead of the built-in set:

.. code-block:: python

   config = CCCConfig(
       group_col="cell_type",
       db_source="cellchatdb",
       db_csv_path="path/to/interaction_CellChatDB.csv",
   )
   result = run_ccc(sp, config)

Interface-aware analysis
------------------------

Stratify CCC by spatial zones when you have an interface result:

.. code-block:: python

   from spatioloji_s.spatial.polygon import identify_interface

   iface = identify_interface(
       sp, group_col="cell_type",
       region_a="Tumor", region_b="Stroma",
   )

   config = CCCConfig(
       group_col="cell_type",
       layer="log_normalized",
       interface_result=iface,     # enables zone analysis
   )
   result = run_ccc(sp, config)

   # Zone comparison: interface vs interior
   sj.visualization.plot_ccc_zones(result, top_n=10)

   # Communication gradient across interface
   sj.visualization.plot_ccc_gradient(result)

Morphology stratification
-------------------------

Stratify communication by sender cell morphology:

.. code-block:: python

   from spatioloji_s.spatial.polygon import compute_morphology, classify_morphology

   compute_morphology(sp, store=True)
   classify_morphology(sp)

   config = CCCConfig(
       group_col="cell_type",
       layer="log_normalized",
       morphology_col="morph_class",   # round/elongated/intermediate
   )
   result = run_ccc(sp, config)

   sj.visualization.plot_ccc_morphology(result, top_n=10)

Key parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``secreted_radius``
     - 200.0
     - Radius for secreted signaling graph (coordinate units)
   * - ``ecm_radius``
     - 200.0
     - Radius for ECM signaling graph
   * - ``buffer_distance``
     - None (0)
     - Buffer for juxtacrine contact graph. 0 = touching cells only
   * - ``sigma_secreted``
     - None (auto)
     - Distance decay sigma. Auto = median edge distance
   * - ``sigma_ecm``
     - None (auto)
     - ECM distance decay sigma
   * - ``min_pct``
     - 0.05
     - Min fraction of cells expressing a gene for LR pair inclusion
   * - ``test_method``
     - "analytical"
     - "analytical" (fast z-score) or "permutation"
   * - ``n_permutations``
     - 1000
     - Number of permutations (when ``test_method="permutation"``)
