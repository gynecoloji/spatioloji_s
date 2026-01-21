# src/spatioloji_s/__init__.py

"""
spatioloji - Spatial transcriptomics data analysis package
"""

# Core data structures
from .data.core import spatioloji
from .data.config import SpatiolojiConfig, SpatialData
from .data.qc import spatioloji_qc, QCConfig

# Import submodules
from . import data
from . import processing
from . import visualization
from . import spatial  # <-- ADD THIS LINE

__version__ = '0.1.0'

__all__ = [
    # Core classes
    'spatioloji',
    'SpatiolojiConfig',
    'SpatialData',
    'spatioloji_qc',
    'QCConfig',
    
    # Submodules
    'data',
    'processing',
    'visualization',
    'spatial',  # <-- ADD THIS LINE
]