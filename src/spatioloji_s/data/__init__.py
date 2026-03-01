"""
data - Core data structures and quality control

This module contains the main spatioloji data structure,
configuration classes, and quality control functionality.
"""

from .config import ConsistencyError, ImageMetadata, SpatialData, SpatiolojiConfig, SpatiolojiError, ValidationError
from .core import spatioloji
from .expression import ExpressionMatrix
from .images import ImageHandler, load_fov_positions_from_images
from .qc import QCConfig, spatioloji_qc
from .utils import export_to_csv_bundle

__all__ = [
    # Core class
    'spatioloji',

    # Configuration
    'SpatiolojiConfig',
    'SpatialData',
    'ImageMetadata',

    # Components
    'ExpressionMatrix',
    'ImageHandler',

    # QC
    'spatioloji_qc',
    'QCConfig',

    # Utilities
    'export_to_csv_bundle',
    'load_fov_positions_from_images',

    # Exceptions
    'SpatiolojiError',
    'ConsistencyError',
    'ValidationError',
]
