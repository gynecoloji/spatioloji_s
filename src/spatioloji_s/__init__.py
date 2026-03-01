# src/spatioloji_s/__init__.py

"""
spatioloji - Spatial transcriptomics data analysis package
"""

# Core data structures
# Import submodules
from . import (
    data,
    processing,
    spatial,  # <-- ADD THIS LINE
    visualization,
)
from .data.config import SpatialData, SpatiolojiConfig
from .data.core import spatioloji
from .data.qc import QCConfig, spatioloji_qc

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "spatioloji",
    "SpatiolojiConfig",
    "SpatialData",
    "spatioloji_qc",
    "QCConfig",
    # Submodules
    "data",
    "processing",
    "visualization",
    "spatial",  # <-- ADD THIS LINE
]
