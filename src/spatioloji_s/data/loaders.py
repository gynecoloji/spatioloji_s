from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging
from typing import Set
from pathlib import Path
import cv2
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import anndata

@dataclass
class SpatiolojiConfig:
    """Configuration for spatioloji column names and settings."""
    
    # Column names
    cell_id_col: str = 'cell'
    fov_id_col: str = 'fov'
    x_local_col: str = 'x_local_px'
    y_local_col: str = 'y_local_px'
    x_global_col: str = 'x_global_px'
    y_global_col: str = 'y_global_px'
    
    # Image settings
    img_format: str = 'jpg'
    img_prefix: str = 'CellComposite_F'
    
    # Validation settings
    strict_validation: bool = False  # If True, raise errors instead of warnings
    
    def get_coordinate_columns(self, coord_type: str) -> tuple[str, str]:
        """Get x, y column names for specified coordinate type."""
        if coord_type == 'local':
            return self.x_local_col, self.y_local_col
        elif coord_type == 'global':
            return self.x_global_col, self.y_global_col
        else:
            raise ValueError(f"Invalid coordinate type: {coord_type}")

class SpatiolojiError(Exception):
    """Base exception for spatioloji errors."""
    pass

class ValidationError(SpatiolojiError):
    """Raised when data validation fails."""
    pass

class ColumnNotFoundError(SpatiolojiError):
    """Raised when a required column is missing."""
    def __init__(self, column: str, dataframe_name: str):
        self.column = column
        self.dataframe_name = dataframe_name
        super().__init__(f"Column '{column}' not found in {dataframe_name}")

class CellIDMismatchError(SpatiolojiError):
    """Raised when cell IDs don't match across datasets."""
    def __init__(self, missing_count: int, source: str, target: str):
        super().__init__(
            f"{missing_count} cells in {source} are missing in {target}"
        )

class ImageNotFoundError(SpatiolojiError):
    """Raised when an expected image is not found."""
    def __init__(self, fov_id: str, path: str):
        self.fov_id = fov_id
        self.path = path
        super().__init__(f"Image for FOV '{fov_id}' not found at {path}")

logger = logging.getLogger(__name__)

class SpatiolojiValidator:
    """Handles validation for spatioloji objects."""
    
    def __init__(self, config: SpatiolojiConfig):
        self.config = config
    
    def validate_columns(self, df: pd.DataFrame, 
                        required_cols: list[str], 
                        df_name: str) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            msg = f"Missing columns in {df_name}: {missing}"
            if self.config.strict_validation:
                raise ColumnNotFoundError(missing[0], df_name)
            else:
                logger.warning(msg)
    
    def validate_cell_ids(self, 
                         gdf_cells: Set[str],
                         meta_cells: Set[str],
                         adata_cells: Set[str]) -> None:
        """Validate cell ID consistency across datasets."""
        
        # Check gdf vs meta
        if diff := gdf_cells - meta_cells:
            self._handle_mismatch(len(diff), "gdf_local", "cell_meta")
        
        if diff := meta_cells - gdf_cells:
            self._handle_mismatch(len(diff), "cell_meta", "gdf_local")
        
        # Check adata vs meta
        if diff := adata_cells - meta_cells:
            self._handle_mismatch(len(diff), "adata", "cell_meta")
        
        if diff := meta_cells - adata_cells:
            self._handle_mismatch(len(diff), "cell_meta", "adata")
    
    def _handle_mismatch(self, count: int, source: str, target: str) -> None:
        """Handle cell ID mismatches."""
        msg = f"{count} cells in {source} are not in {target}"
        
        if self.config.strict_validation:
            raise CellIDMismatchError(count, source, target)
        else:
            logger.warning(msg)
            


class ImageHandler:
    """Handles image loading and management for spatioloji."""
    
    def __init__(self, config: SpatiolojiConfig):
        self.config = config
        self.images: Dict[str, np.ndarray] = {}
        self.image_shapes: Dict[str, Tuple[int, int]] = {}
    
    def load_from_folder(self, folder_path: str, fov_ids: List[str]) -> None:
        """Load images from folder for specified FOV IDs."""
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.warning(f"Image folder '{folder_path}' does not exist")
            return
        
        for fov_id in fov_ids:
            self._load_single_image(folder, fov_id)
        
        if not self.images:
            logger.warning(f"No images loaded from '{folder_path}'")
    
    def _load_single_image(self, folder: Path, fov_id: str) -> None:
        """Load a single image for a FOV ID."""
        # Try zero-padded format first
        img_path = self._get_image_path(folder, fov_id, zero_pad=True)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        # Try non-padded format if first attempt fails
        if img is None:
            img_path = self._get_image_path(folder, fov_id, zero_pad=False)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if img is not None:
            self.images[fov_id] = img
            self.image_shapes[fov_id] = (img.shape[1], img.shape[0])
        else:
            logger.warning(f"Image for FOV '{fov_id}' not found")
    
    def _get_image_path(self, folder: Path, fov_id: str, 
                       zero_pad: bool = False) -> Path:
        """Construct image path with optional zero-padding."""
        if zero_pad and fov_id.isdigit():
            fov_id = fov_id.zfill(3)
        
        filename = f"{self.config.img_prefix}{fov_id}.{self.config.img_format}"
        return folder / filename
    
    def get_image(self, fov_id: str) -> Optional[np.ndarray]:
        """Get image for a specific FOV ID."""
        return self.images.get(fov_id)
    
    def subset_by_fovs(self, fov_ids: List[str]) -> 'ImageHandler':
        """Create new ImageHandler with subset of images."""
        new_handler = ImageHandler(self.config)
        new_handler.images = {
            fid: img for fid, img in self.images.items() 
            if fid in fov_ids
        }
        new_handler.image_shapes = {
            fid: shape for fid, shape in self.image_shapes.items() 
            if fid in fov_ids
        }
        return new_handler
    
    def get_missing_fovs(self, fov_ids: List[str]) -> List[str]:
        """Get list of FOV IDs that don't have images."""
        return [fid for fid in fov_ids if fid not in self.images]

class GeoDataFrameBuilder:
    """Builds GeoDataFrames from polygon data."""
    
    def __init__(self, config: SpatiolojiConfig):
        self.config = config
    
    def build(self, polygons_df: pd.DataFrame, 
              coord_type: str = 'local') -> gpd.GeoDataFrame:
        """Build GeoDataFrame from polygon DataFrame."""
        try:
            x_col, y_col = self.config.get_coordinate_columns(coord_type)
            
            if not self._has_required_columns(polygons_df, x_col, y_col):
                logger.warning(f"Missing columns for {coord_type} GeoDataFrame")
                return gpd.GeoDataFrame()
            
            geometries = self._create_geometries(polygons_df, x_col, y_col)
            gdf = self._create_geodataframe(geometries, polygons_df)
            
            return gdf
            
        except ImportError:
            logger.error("geopandas/shapely not installed")
            return self._create_empty_gdf(polygons_df)
    
    def _has_required_columns(self, df: pd.DataFrame, 
                             x_col: str, y_col: str) -> bool:
        """Check if DataFrame has required coordinate columns."""
        return x_col in df.columns and y_col in df.columns
    
    def _create_geometries(self, df: pd.DataFrame, 
                          x_col: str, y_col: str) -> List[Tuple[str, Polygon]]:
        """Create polygon geometries grouped by cell."""
        geometries = []
        
        for cell_id, group in df.groupby(self.config.cell_id_col):
            coords = list(zip(group[x_col], group[y_col]))
            
            # Close polygon if needed
            if coords[0] != coords[-1] and len(coords) > 2:
                coords.append(coords[0])
            
            if len(coords) >= 3:
                geometries.append((cell_id, Polygon(coords)))
            else:
                logger.warning(f"Cell {cell_id} has < 3 points, skipping")
        
        return geometries
    
    def _create_geodataframe(self, geometries: List[Tuple[str, Polygon]], 
                            original_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create GeoDataFrame and merge with original data."""
        gdf = gpd.GeoDataFrame(
            data=[g[0] for g in geometries],
            geometry=[g[1] for g in geometries],
            columns=[self.config.cell_id_col]
        )
        
        # Merge with cell-level data
        cell_info = self._get_cell_info(original_df)
        gdf = gdf.merge(cell_info, on=self.config.cell_id_col)
        
        # Preserve original order
        gdf = gdf.set_index(self.config.cell_id_col)
        gdf = gdf.loc[cell_info[self.config.cell_id_col]]
        gdf = gdf.reset_index()
        
        return gdf
    
    def _get_cell_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract one row per cell with non-coordinate columns."""
        x_col, y_col = self.config.get_coordinate_columns('local')
        drop_cols = [x_col, y_col]
        
        # Also drop global coords if present
        x_global, y_global = self.config.get_coordinate_columns('global')
        if x_global in df.columns:
            drop_cols.extend([x_global, y_global])
        
        return df.drop(drop_cols, axis=1).drop_duplicates(self.config.cell_id_col)
    
    def _create_empty_gdf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create empty DataFrame with geometry column."""
        df_copy = df.copy()
        df_copy['geometry'] = None
        return df_copy
      
class spatioloji:
    """
    A class for managing and analyzing spatial transcriptomics data.
    
    This refactored version uses composition with helper classes for
    better separation of concerns and maintainability.
    """
    
    def __init__(self,
                 polygons: pd.DataFrame,
                 cell_meta: pd.DataFrame,
                 adata: anndata.AnnData,
                 fov_positions: pd.DataFrame,
                 config: Optional[SpatiolojiConfig] = None,
                 images: Optional[Dict[str, np.ndarray]] = None,
                 image_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
                 images_folder: Optional[str] = None):
        """
        Initialize a spatioloji object.
        
        Parameters
        ----------
        polygons : pd.DataFrame
            Polygon data with cell boundaries
        cell_meta : pd.DataFrame
            Cell metadata
        adata : anndata.AnnData
            Gene expression data
        fov_positions : pd.DataFrame
            FOV position data
        config : SpatiolojiConfig, optional
            Configuration object. If None, uses defaults.
        images : Dict[str, np.ndarray], optional
            Pre-loaded images
        image_shapes : Dict[str, Tuple[int, int]], optional
            Image shapes
        images_folder : str, optional
            Folder to load images from
        """
        # Initialize configuration
        self.config = config or SpatiolojiConfig()
        
        # Store core data
        self.polygons = polygons
        self.cell_meta = cell_meta
        self.adata = adata
        self.fov_positions = fov_positions
        self.custom: Dict[str, any] = {}
        
        # Initialize helper components
        self._init_helpers()
        
        # Build GeoDataFrames
        self.gdf_local = self.gdf_builder.build(polygons, 'local')
        self.gdf_global = self.gdf_builder.build(polygons, 'global')
        
        # Handle images
        self._init_images(images, image_shapes, images_folder)
        
        # Validate
        self.validator.validate_columns(
            polygons, 
            [self.config.cell_id_col], 
            'polygons'
        )
        self.validator.validate_columns(
            cell_meta, 
            [self.config.cell_id_col], 
            'cell_meta'
        )
        self._validate_cell_ids()
    
    def _init_helpers(self) -> None:
        """Initialize helper components."""
        self.validator = SpatiolojiValidator(self.config)
        self.gdf_builder = GeoDataFrameBuilder(self.config)
        self.image_handler = ImageHandler(self.config)
    
    def _init_images(self, images: Optional[Dict], 
                    image_shapes: Optional[Dict],
                    images_folder: Optional[str]) -> None:
        """Initialize images from provided data or folder."""
        if images and image_shapes:
            # Use pre-loaded images
            self.image_handler.images = images
            self.image_handler.image_shapes = image_shapes
        elif images_folder:
            # Load from folder
            fov_ids = self._get_fov_ids()
            self.image_handler.load_from_folder(images_folder, fov_ids)
    
    def _get_fov_ids(self) -> List[str]:
        """Extract FOV IDs from fov_positions."""
        fov_col = self._find_fov_column()
        return self.fov_positions[fov_col].astype(str).tolist()
    
    def _find_fov_column(self) -> str:
        """Find the FOV ID column in fov_positions."""
        df = self.fov_positions
        
        # Try configured column name
        if self.config.fov_id_col in df.columns:
            return self.config.fov_id_col
        
        # Look for any column with 'fov' in name
        fov_cols = [col for col in df.columns if 'fov' in col.lower()]
        if fov_cols:
            return fov_cols[0]
        
        # Fallback to first column
        logger.warning(
            f"FOV column not found, using '{df.columns[0]}'"
        )
        return df.columns[0]
    
    def _validate_cell_ids(self) -> None:
        """Validate cell ID consistency across datasets."""
        gdf_cells = set(self.gdf_local[self.config.cell_id_col].astype(str)) \
                    if not self.gdf_local.empty else set()
        meta_cells = set(self.cell_meta[self.config.cell_id_col].astype(str))
        
        # Get adata cells
        if self.config.cell_id_col in self.adata.obs.columns:
            adata_cells = set(self.adata.obs[self.config.cell_id_col].astype(str))
        else:
            adata_cells = set(self.adata.obs.index.astype(str))
        
        self.validator.validate_cell_ids(gdf_cells, meta_cells, adata_cells)
        
    def subset_by_fovs(self, fov_ids: List[str]) -> 'spatioloji':
        """
        Create a new spatioloji object with only specified FOVs.
        
        Parameters
        ----------
        fov_ids : List[str]
            FOV IDs to include
        
        Returns
        -------
        spatioloji
            New object with subset data
        """
        fov_ids = [str(fid) for fid in fov_ids]
        fov_col = self._find_fov_column()
        
        # Validate FOV column exists
        if not self._has_fov_column(fov_col):
            logger.warning("Cannot subset by FOVs: FOV column missing")
            return self
        
        # Subset data components
        subset_data = self._create_fov_subset(fov_ids, fov_col)
        
        # Create new instance
        return self._create_subset_instance(subset_data)
    
    def subset_by_cells(self, cell_ids: List[str]) -> 'spatioloji':
        """
        Create a new spatioloji object with only specified cells.
        
        Parameters
        ----------
        cell_ids : List[str]
            Cell IDs to include
        
        Returns
        -------
        spatioloji
            New object with subset data
        """
        cell_ids = [str(cid) for cid in cell_ids]
        
        # Subset data components
        subset_data = self._create_cell_subset(cell_ids)
        
        # Create new instance
        return self._create_subset_instance(subset_data)
    
    def _has_fov_column(self, fov_col: str) -> bool:
        """Check if FOV column exists in required DataFrames."""
        return (fov_col in self.cell_meta.columns and 
                fov_col in self.polygons.columns)
    
    def _create_fov_subset(self, fov_ids: List[str], 
                          fov_col: str) -> dict:
        """Create subset of all data components by FOVs."""
        # Subset polygons and cell_meta
        polygons = self.polygons[
            self.polygons[fov_col].astype(str).isin(fov_ids)
        ]
        cell_meta = self.cell_meta[
            self.cell_meta[fov_col].astype(str).isin(fov_ids)
        ]
        
        # Get cells in these FOVs
        cell_ids = cell_meta[self.config.cell_id_col].astype(str).unique()
        
        # Subset other components
        adata = self._subset_adata(cell_ids)
        fov_positions = self.fov_positions[
            self.fov_positions[fov_col].astype(str).isin(fov_ids)
        ]
        image_handler = self.image_handler.subset_by_fovs(fov_ids)
        
        return {
            'polygons': polygons,
            'cell_meta': cell_meta,
            'adata': adata,
            'fov_positions': fov_positions,
            'images': image_handler.images,
            'image_shapes': image_handler.image_shapes
        }
    
    def _create_cell_subset(self, cell_ids: List[str]) -> dict:
        """Create subset of all data components by cells."""
        # Subset polygons and cell_meta
        polygons = self.polygons[
            self.polygons[self.config.cell_id_col].astype(str).isin(cell_ids)
        ]
        cell_meta = self.cell_meta[
            self.cell_meta[self.config.cell_id_col].astype(str).isin(cell_ids)
        ]
        
        # Subset other components
        adata = self._subset_adata(cell_ids)
        fov_data = self._subset_fov_data(cell_meta)
        
        return {
            'polygons': polygons,
            'cell_meta': cell_meta,
            'adata': adata,
            'fov_positions': fov_data['fov_positions'],
            'images': fov_data['images'],
            'image_shapes': fov_data['image_shapes']
        }
    
    def _subset_adata(self, cell_ids: List[str]) -> anndata.AnnData:
        """Subset AnnData by cell IDs."""
        if self.config.cell_id_col in self.adata.obs.columns:
            mask = self.adata.obs[self.config.cell_id_col].astype(str).isin(cell_ids)
        else:
            mask = self.adata.obs.index.astype(str).isin(cell_ids)
        
        return self.adata[mask].copy()
    
    def _subset_fov_data(self, cell_meta: pd.DataFrame) -> dict:
        """Subset FOV-related data based on cells."""
        fov_col = self._find_fov_column()
        
        if fov_col not in cell_meta.columns:
            logger.warning("FOV column not found, keeping all FOV data")
            return {
                'fov_positions': self.fov_positions.copy(),
                'images': self.image_handler.images.copy(),
                'image_shapes': self.image_handler.image_shapes.copy()
            }
        
        # Get FOVs for these cells
        fov_ids = cell_meta[fov_col].astype(str).unique().tolist()
        
        fov_positions = self.fov_positions[
            self.fov_positions[fov_col].astype(str).isin(fov_ids)
        ]
        image_handler = self.image_handler.subset_by_fovs(fov_ids)
        
        return {
            'fov_positions': fov_positions,
            'images': image_handler.images,
            'image_shapes': image_handler.image_shapes
        }
    
    def _create_subset_instance(self, subset_data: dict) -> 'spatioloji':
        """Create new spatioloji instance from subset data."""
        return spatioloji(
            polygons=subset_data['polygons'],
            cell_meta=subset_data['cell_meta'],
            adata=subset_data['adata'],
            fov_positions=subset_data['fov_positions'],
            config=self.config,
            images=subset_data['images'],
            image_shapes=subset_data['image_shapes']
        )
    
    def get_cells_in_fov(self, fov_id: str, 
                        fov_column: Optional[str] = None) -> pd.DataFrame:
        """Get all cells within a specific FOV."""
        fov_col = fov_column or self.config.fov_id_col
        
        if fov_col not in self.cell_meta.columns:
            logger.warning(f"Column '{fov_col}' not found in cell_meta")
            return pd.DataFrame()
        
        return self.cell_meta[self.cell_meta[fov_col] == fov_id]
    
    def get_polygon_for_cell(self, cell_id: str) -> pd.DataFrame:
        """Get polygon data for a specific cell."""
        return self.polygons[
            self.polygons[self.config.cell_id_col] == cell_id
        ]
    
    def get_geometry_for_cell(self, cell_id: str, 
                             coordinate_type: str = 'local') -> Optional[any]:
        """Get geometry for a specific cell from GeoDataFrame."""
        gdf = self.gdf_local if coordinate_type == 'local' else self.gdf_global
        
        if gdf.empty:
            logger.warning(f"{coordinate_type} GeoDataFrame is empty")
            return None
        
        mask = gdf[self.config.cell_id_col] == cell_id
        if not mask.any():
            logger.warning(f"Cell '{cell_id}' not found in {coordinate_type} GeoDataFrame")
            return None
        
        return gdf[mask].geometry.iloc[0]
    
    def get_image(self, fov_id: str) -> Optional[np.ndarray]:
        """Get image for a specific FOV."""
        return self.image_handler.get_image(fov_id)
    
    # ========== Summary & Custom Data ==========
    
    def summary(self) -> Dict[str, any]:
        """Get summary of the object's data."""
        return {
            "n_cells": len(self.cell_meta),
            "n_fovs": len(self.fov_positions),
            "n_polygons": len(self.polygons),
            "n_local_geometries": len(self.gdf_local),
            "n_global_geometries": len(self.gdf_global),
            "n_images": len(self.image_handler.images),
            "image_fovs": list(self.image_handler.images.keys()),
            "adata_shape": self.adata.shape,
            "config": self.config
        }
    
    def add_custom(self, key: str, value: any) -> None:
        """Add custom data to the object."""
        self.custom[key] = value
    
    def get_custom(self, key: str, default: any = None) -> any:
        """Retrieve custom data from the object."""
        return self.custom.get(key, default)
    
    # ========== Serialization ==========
    
    def to_pickle(self, filepath: str) -> None:
        """Save object to pickle file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def to_anndata(self) -> anndata.AnnData:
        """Get the AnnData object."""
        return self.adata
    
    # ========== Static Factory Methods ==========
    
    @staticmethod
    def from_pickle(filepath: str) -> 'spatioloji':
        """Load spatioloji object from pickle file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def from_files(polygons_path: str,
                   cell_meta_path: str,
                   adata_path: str,
                   fov_positions_path: str,
                   images_folder: Optional[str] = None,
                   config: Optional[SpatiolojiConfig] = None) -> 'spatioloji':
        """
        Create spatioloji object from file paths.
        
        Parameters
        ----------
        polygons_path : str
            Path to CSV with polygon data
        cell_meta_path : str
            Path to CSV with cell metadata
        adata_path : str
            Path to h5ad file
        fov_positions_path : str
            Path to CSV with FOV positions
        images_folder : str, optional
            Path to folder with images
        config : SpatiolojiConfig, optional
            Configuration object
        
        Returns
        -------
        spatioloji
            New spatioloji object
        """
        # Load data files
        polygons = pd.read_csv(polygons_path)
        cell_meta = pd.read_csv(cell_meta_path)
        adata = anndata.read_h5ad(adata_path)
        fov_positions = pd.read_csv(fov_positions_path)
        
        # Create instance
        return spatioloji(
            polygons=polygons,
            cell_meta=cell_meta,
            adata=adata,
            fov_positions=fov_positions,
            config=config,
            images_folder=images_folder
        )