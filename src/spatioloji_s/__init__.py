# src/spatioloji_s/__init__.py

"""
spatioloji - Spatial transcriptomics data analysis package
"""

# Core data structures
# Import submodules
from . import (
    ccc,
    data,
    processing,
    spatial,
    visualization,
)
from .data.config import SpatialData, SpatiolojiConfig
from .data.core import spatioloji
from .data.qc import CosmxQCConfig, QCConfig, XeniumQCConfig, spatioloji_qc, xenium_qc

# Bumped automatically by release-please on each release — do not edit by hand.
__version__ = "2.0.0"

__all__ = [
    # Core classes
    "spatioloji",
    "SpatiolojiConfig",
    "SpatialData",
    "spatioloji_qc",
    "QCConfig",
    "xenium_qc",
    "CosmxQCConfig",
    "XeniumQCConfig",
    # Submodules
    "data",
    "processing",
    "visualization",
    "spatial",
    "ccc",
]
