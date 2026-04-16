"""
ccc - Cell-Cell Communication analysis for spatioloji

Scoring-based CCC framework:

  Database  — load LR pairs from builtin / CellChatDB CSV
  Scoring   — score edges, aggregate, test significance
  Zones     — interface zone comparison, gradient, morphology stratification
  Run       — full pipeline orchestrator (run_ccc)

Typical usage
-------------
>>> from spatioloji_s.ccc import CCCConfig, CCCResult, run_ccc
>>>
>>> config = CCCConfig(
...     group_col="cell_type",
...     db_source="builtin",
...     secreted_radius=200.0,
... )
>>> result = run_ccc(sp_fov, config)
>>> result.scores[result.scores["fdr"] < 0.05]
"""

from .database import (
    LRPair,
    filter_to_expressed,
    load_from_cellchatdb_csv,
    load_lr_database,
    lr_pairs_to_dataframe,
)
from .run import (
    CCCConfig,
    CCCResult,
    run_ccc,
)
from .scoring import (
    aggregate_scores,
    score_edges,
    test_significance,
)
from .zones import (
    communication_gradient,
    compare_morphology,
    compare_zones,
)
from .halo import (
    create_halo_subset,
    filter_ccc_to_core,
)

__all__ = [
    # Database
    "LRPair",
    "filter_to_expressed",
    "load_from_cellchatdb_csv",
    "load_lr_database",
    "lr_pairs_to_dataframe",
    # Run
    "CCCConfig",
    "CCCResult",
    "run_ccc",
    # Scoring
    "aggregate_scores",
    "score_edges",
    "test_significance",
    # Zones
    "communication_gradient",
    "compare_morphology",
    "compare_zones",
    # Halo
    "create_halo_subset",
    "filter_ccc_to_core",
]
