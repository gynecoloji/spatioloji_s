Changelog
=========

v0.3.0 (2026)
--------------

New features
^^^^^^^^^^^^

- **Spatial motif discovery** -- Multi-scale tissue architecture analysis with
  ``discover_motifs``, ``detect_assemblies``, and ``match_known_structures``.
  Built-in TME presets for TLS, tumor buds, immune desert detection.
- **Interface detection** -- Graph-based and density-based tissue interface
  identification between cell-type regions.
- **Expression gradients** -- OLS regression of gene expression vs signed
  distance from interface, with NMF/PCA gene program discovery.
- **Immune infiltration scoring** -- Penetration depth, density gradient, and
  infiltration fraction per immune cell type.
- **CCC module rewritten** -- Modular scoring + zones design with
  ``CCCConfig``/``CCCResult`` API. Edge-level scoring with polygon geometry.
- **Differential expression** -- Five methods: Wilcoxon, t-test, MAST, NB-GLM, DESeq2.
- **Pathway activity scoring** -- Gene set scoring via decoupler integration.
- **80+ visualization functions** -- Comprehensive plotting for all analysis results.

v0.2.0 (2025)
--------------

- 3-layer CCC framework (Bivariate Moran's I, Polygon OT, Contrastive NMF)
- Polygon morphology (8 shape metrics including contour entropy)
- Multi-FOV CCC with cross-FOV graph support

v0.1.0 (2024)
--------------

- Initial release
- Core ``spatioloji`` data structure
- QC pipeline with NegProbe-aware filtering
- Processing pipeline (normalization, clustering, batch correction, imputation)
- Point and polygon spatial analysis
