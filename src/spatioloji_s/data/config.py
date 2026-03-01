"""
config.py - Configuration and data structures for spatioloji

Contains:
- SpatiolojiConfig: Configuration settings
- SpatialData: Efficient spatial coordinate storage
- ImageMetadata: Image metadata container
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpatiolojiConfig:
    """Configuration for spatioloji column names and settings."""

    # Column names
    cell_id_col: str = "cell"
    fov_id_col: str = "fov"
    x_local_col: str = "x_local_px"
    y_local_col: str = "y_local_px"
    x_global_col: str = "x_global_px"
    y_global_col: str = "y_global_px"

    # Image settings
    img_format: str = "jpg"
    img_prefix: str = "CellComposite_F"

    # Processing settings
    auto_sparse_threshold: float = 0.5  # Convert to sparse if >50% zeros

    def get_coordinate_columns(self, coord_type: str) -> tuple[str, str]:
        """
        Get x, y column names for specified coordinate type.

        Parameters
        ----------
        coord_type : str
            'local' or 'global'

        Returns
        -------
        Tuple[str, str]
            (x_column, y_column)
        """
        if coord_type == "local":
            return self.x_local_col, self.y_local_col
        elif coord_type == "global":
            return self.x_global_col, self.y_global_col
        else:
            raise ValueError(f"Invalid coordinate type: {coord_type}")


@dataclass
class SpatialData:
    """
    Efficient container for spatial coordinates.

    Uses numpy arrays for fast computation.
    """

    x_local: np.ndarray
    y_local: np.ndarray
    x_global: np.ndarray
    y_global: np.ndarray

    def __len__(self) -> int:
        return len(self.x_local)

    def __post_init__(self):
        """Validate all arrays have same length."""
        lengths = [len(self.x_local), len(self.y_local), len(self.x_global), len(self.y_global)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Coordinate arrays have different lengths: {lengths}")

    def subset(self, indices: np.ndarray) -> "SpatialData":
        """
        Efficiently subset by integer indices.

        Parameters
        ----------
        indices : np.ndarray
            Integer indices to keep

        Returns
        -------
        SpatialData
            New SpatialData with subset
        """
        return SpatialData(
            x_local=self.x_local[indices],
            y_local=self.y_local[indices],
            x_global=self.x_global[indices],
            y_global=self.y_global[indices],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"x_local": self.x_local, "y_local": self.y_local, "x_global": self.x_global, "y_global": self.y_global}

    @classmethod
    def from_dataframe(cls, df, config: SpatiolojiConfig) -> "SpatialData":
        """
        Create from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with coordinate columns
        config : SpatiolojiConfig
            Configuration with column names

        Returns
        -------
        SpatialData
        """
        x_local_col, y_local_col = config.get_coordinate_columns("local")
        x_global_col, y_global_col = config.get_coordinate_columns("global")

        return cls(
            x_local=df[x_local_col].values,
            y_local=df[y_local_col].values,
            x_global=df[x_global_col].values,
            y_global=df[y_global_col].values,
        )


@dataclass
class ImageMetadata:
    """Metadata for a single FOV image."""

    fov_id: str
    shape: tuple[int, ...]
    dtype: np.dtype
    path: Path | None = None
    is_loaded: bool = False

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def n_channels(self) -> int:
        return self.shape[2] if len(self.shape) == 3 else 1

    def memory_size_mb(self) -> float:
        """Estimate memory size in MB."""
        n_pixels = np.prod(self.shape)
        bytes_per_pixel = self.dtype.itemsize
        return (n_pixels * bytes_per_pixel) / (1024 * 1024)


class SpatiolojiError(Exception):
    """Base exception for spatioloji errors."""

    pass


class ConsistencyError(SpatiolojiError):
    """Raised when data consistency checks fail."""

    pass


class ValidationError(SpatiolojiError):
    """Raised when data validation fails."""

    pass
