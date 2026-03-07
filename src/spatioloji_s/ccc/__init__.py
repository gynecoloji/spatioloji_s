"""
ccc - Cell-Cell Communication analysis for spatioloji

Polygon-native CCC framework built around three layers:

  Layer 1 — Discovery (Bivariate Moran's I + Spatial Lag Regression)
    Identifies which LR pairs show significant spatial coupling
    between cell types, ranked by effect size.

  Layer 2 — Cell-Pair Scoring (Polygon OT + Message Passing)
    Scores every contacting cell pair for each significant LR pair.
    Detects hub sender and receiver cells.

  Layer 3 — Pattern Detection (Contrastive Scoring + NMF)
    Classifies LR pairs by mechanistic driver (expression vs geometry)
    and decomposes communication into K recurring spatial programs.

Typical usage
-------------
>>> from spatioloji_s.ccc import CCCConfig, run_ccc, summarize_ccc
>>>
>>> config = CCCConfig(
...     cell_type_col = 'cell_type',
...     layer         = 'log_normalized',
...     db_source     = 'cellchatdb',
...     db_csv_path   = 'CellChatDB.csv',
...     K             = 5,
... )
>>> ccc_results = run_ccc(sp_fov, config)
>>> summarize_ccc(ccc_results)
"""

from .database import (
    LRPair,
    filter_to_expressed,
    load_from_cellchatdb_csv,
    load_lr_database,
    lr_pairs_to_dataframe,
)
from .layer1 import (
    build_weight_matrices,
    compute_bivariate_moran,
    compute_spatial_lag,
    get_top_pairs,
    run_layer1,
    summarize_layer1,
)
from .layer2 import run_layer2
from .layer3 import (
    compute_contrastive_scores,
    get_program_cells,
    get_program_lr_pairs,
    run_layer3,
)
from .run import (
    CCCConfig,
    compare_fov_results,
    load_ccc_results,
    run_ccc,
    run_ccc_multifov,
    save_ccc_results,
    summarize_ccc,
)

__all__ = [
    "CCCConfig",
    "LRPair",
    "build_weight_matrices",
    "compare_fov_results",
    "compute_bivariate_moran",
    "compute_contrastive_scores",
    "compute_spatial_lag",
    "filter_to_expressed",
    "get_program_cells",
    "get_program_lr_pairs",
    "get_top_pairs",
    "load_ccc_results",
    "load_from_cellchatdb_csv",
    "load_lr_database",
    "lr_pairs_to_dataframe",
    "run_ccc",
    "run_ccc_multifov",
    "run_layer1",
    "run_layer2",
    "run_layer3",
    "save_ccc_results",
    "summarize_ccc",
    "summarize_layer1",
]
