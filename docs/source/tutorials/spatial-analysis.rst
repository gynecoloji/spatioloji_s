Spatial Analysis
================

spatioloji_s provides two complementary spatial analysis modes.

Point-based analysis
--------------------

Uses cell centroids. Fast and scalable.

.. code-block:: python

   from spatioloji_s.spatial.point import (
       build_knn_graph,
       neighborhood_enrichment,
       morans_i,
       ripleys_k,
   )

   # Build a KNN graph
   graph = build_knn_graph(sp, k=15)

   # Neighborhood enrichment
   result = neighborhood_enrichment(sp, graph, group_col="cell_type")
   sj.visualization.plot_neighborhood_enrichment(result)

   # Spatial autocorrelation
   mi = morans_i(sp, graph, feature="EPCAM", layer="log_normalized")
   sj.visualization.plot_morans_i_map(sp, mi)

   # Ripley's K function
   rk = ripleys_k(sp, cell_type="Tumor", n_simulations=99)
   sj.visualization.plot_ripley(rk)

Polygon-based analysis
----------------------

Uses full cell geometry. More biologically accurate.

.. code-block:: python

   from spatioloji_s.spatial.polygon import (
       build_buffer_graph,
       build_contact_graph,
       compute_morphology,
       contact_fraction,
       neighborhood_enrichment,
   )

   # Build a contact graph
   graph = build_contact_graph(sp)

   # Or a buffer graph (includes near-contact cells)
   graph = build_buffer_graph(sp, buffer_distance=15)

   # Cell morphology
   compute_morphology(sp, store=True)
   sj.visualization.plot_morphology_distribution(sp)
   sj.visualization.plot_morphology_map(sp, feature="circularity")

   # Contact analysis
   frac = contact_fraction(sp, graph)
   sj.visualization.plot_contact_summary(frac)

   # Neighborhood enrichment (polygon adjacency)
   result = neighborhood_enrichment(sp, graph, group_col="cell_type")

Interface detection
-------------------

Identify boundaries between tissue regions.

.. code-block:: python

   from spatioloji_s.spatial.polygon import identify_interface, compute_gradient

   # Detect tumor-stroma interface
   iface = identify_interface(
       sp, group_col="cell_type",
       region_a="Tumor", region_b="Stroma",
   )
   sj.visualization.plot_interface_polygon_map(sp, iface)

   # Compute expression gradient across interface
   grad = compute_gradient(sp, iface, genes=["TGFB1", "VIM"])
   sj.visualization.plot_gradient_curve(grad)

Spatial motifs
--------------

Discover recurring multicellular patterns.

.. code-block:: python

   from spatioloji_s.spatial.polygon import run_motif_pipeline

   motifs = run_motif_pipeline(
       sp, graph,
       group_col="cell_type",
       match_builtin="TME",
   )
   sj.visualization.plot_motif_map(sp, motifs)
   sj.visualization.plot_assembly_map(sp, motifs)
