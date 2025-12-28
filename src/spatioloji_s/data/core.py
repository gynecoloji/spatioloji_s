"""
core.py - Main spatioloji class for spatial transcriptomics data

The spatioloji class is the core data structure for managing spatial 
transcriptomics data with guaranteed consistency across all components.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    import anndata
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
from pathlib import Path

from .config import SpatiolojiConfig, SpatialData, ConsistencyError, ValidationError
from .expression import ExpressionMatrix
from .images import ImageHandler, load_fov_positions_from_images


class spatioloji:
    """
    Efficient spatial transcriptomics data structure.
    
    Core Principles:
    - Master Index: Cell IDs stored once as primary index
    - Efficient Storage: Sparse matrices, numpy arrays
    - Automatic Consistency: All data aligned to master indices
    - Fast Operations: Integer-based indexing internally
    - FOV-Image Integration: Automatic FOV system management
    
    Attributes
    ----------
    _cell_index : pd.Index
        Master cell index (single source of truth)
    _gene_index : pd.Index
        Master gene index
    _fov_index : pd.Index
        Master FOV index
    _expression : ExpressionMatrix
        Gene expression data
    _cell_meta : pd.DataFrame
        Cell metadata (aligned to _cell_index)
    _gene_meta : pd.DataFrame
        Gene metadata (aligned to _gene_index)
    _spatial : SpatialData
        Spatial coordinates
    _polygons : pd.DataFrame
        Cell boundary polygons
    _fov_positions : pd.DataFrame
        FOV positions
    _image_handler : ImageHandler
        Image manager
    _layers : np.ndarray
        Processed expression matrices
    _embeddings : Dict
        Low-dimensional embeddings
    """
    
    def __init__(self,
                 expression: Union[np.ndarray, sparse.spmatrix, ExpressionMatrix],
                 cell_ids: Union[List[str], pd.Index],
                 gene_names: Union[List[str], pd.Index],
                 cell_metadata: Optional[pd.DataFrame] = None,
                 gene_metadata: Optional[pd.DataFrame] = None,
                 spatial_coords: Optional[Union[Dict, SpatialData, pd.DataFrame]] = None,
                 polygons: Optional[pd.DataFrame] = None,
                 fov_positions: Optional[pd.DataFrame] = None,
                 images: Optional[Union[Dict[str, np.ndarray], ImageHandler]] = None,
                 images_folder: Optional[str] = None,
                 image_pattern: str = "CellComposite_F{fov_id}.{ext}",
                 image_extensions: List[str] = ['jpg', 'png', 'tif', 'tiff'],
                 lazy_load_images: bool = True,
                 image_cache_size: Optional[int] = 10,
                 config: Optional[SpatiolojiConfig] = None):
        """
        Initialize spatioloji object.
        
        Parameters
        ----------
        expression : array-like or ExpressionMatrix
            Expression matrix (cells × genes)
        cell_ids : list or Index
            Cell identifiers
        gene_names : list or Index
            Gene names
        cell_metadata : DataFrame, optional
            Cell annotations. Will be aligned to cell_ids.
        gene_metadata : DataFrame, optional
            Gene annotations. Will be aligned to gene_names.
        spatial_coords : dict, SpatialData, or DataFrame, optional
            Spatial coordinates. Can be:
            - Dict with keys: x_local, y_local, x_global, y_global
            - SpatialData object
            - DataFrame with coordinate columns
        polygons : DataFrame, optional
            Cell boundary polygons with 'cell' column
        fov_positions : DataFrame, optional
            FOV positions
        images : dict or ImageHandler, optional
            Pre-loaded images
        images_folder : str, optional
            Path to image folder (auto-loads)
        image_pattern : str
            Image filename pattern
        image_extensions : list
            Image file extensions to try
        lazy_load_images : bool
            Enable lazy image loading
        image_cache_size : int, optional
            Max images in memory
        config : SpatiolojiConfig, optional
            Configuration object
        """
        self.config = config or SpatiolojiConfig()
        
        print("\n" + "="*70)
        print("Initializing spatioloji")
        print("="*70)
        
        # STEP 1: Establish master indices
        self._cell_index = pd.Index(cell_ids, name=self.config.cell_id_col)
        self._gene_index = pd.Index(gene_names, name='gene')
        self._n_cells = len(self._cell_index)
        self._n_genes = len(self._gene_index)
        
        print(f"[1/11] Master indices: {self._n_cells:,} cells × {self._n_genes:,} genes")
        
        # STEP 2: Store expression efficiently
        if isinstance(expression, ExpressionMatrix):
            self._expression = expression
        else:
            self._expression = ExpressionMatrix(
                data=expression,
                cell_ids=self._cell_index,
                gene_names=self._gene_index,
                auto_sparse=True,
                sparse_threshold=self.config.auto_sparse_threshold
            )
        print(f"[2/11] Expression matrix: {'sparse' if self._expression.is_sparse else 'dense'} "
              f"({self._expression.memory_usage_mb():.1f} MB)")
        
        # STEP 3: Store and align cell metadata
        self._cell_meta = self._prepare_cell_metadata(cell_metadata)
        print(f"[3/10] Cell metadata: {len(self._cell_meta.columns)} columns")
        
        # STEP 4: Store and align gene metadata
        self._gene_meta = self._prepare_gene_metadata(gene_metadata)
        print(f"[4/11] Gene metadata: {len(self._gene_meta.columns)} columns")
        
        # STEP 5: Store spatial coordinates
        self._spatial = self._prepare_spatial_coords(spatial_coords)
        print(f"[5/11] Spatial coordinates: {len(self._spatial)} cells")
        
        # STEP 6: Store and validate polygons
        self._polygons = self._prepare_polygons(polygons)
        if self._polygons is not None:
            n_poly_cells = self._polygons[self.config.cell_id_col].nunique()
            print(f"[6/11] Polygons: {len(self._polygons)} vertices for {n_poly_cells} cells")
        else:
            print(f"[6/11] Polygons: None")
        
        # STEP 7: Initialize ImageHandler
        self._image_handler = self._prepare_image_handler(
            images=images,
            images_folder=images_folder,
            pattern=image_pattern,
            extensions=image_extensions,
            lazy_load=lazy_load_images,
            cache_size=image_cache_size
        )
        print(f"[7/11] Images: {self._image_handler.n_total} total, "
              f"{self._image_handler.n_loaded} loaded")
        
        # STEP 8: Prepare FOV system
        self._fov_positions, self._fov_index = self._prepare_fov_system(fov_positions)
        print(f"[8/11] FOV system: {len(self._fov_index)} FOVs")
        
        # STEP 9: Create fast lookup structures
        self._create_index_maps()
        print(f"[9/11] Index maps created")
        
        # STEP 10: Validate consistency
        self._gdf_local = None
        self._gdf_global = None
        
        status = self.validate_consistency(raise_error=False)
        if status['overall']:
            print(f"[10/11] Validation: ✓ All consistent")
        else:
            failed = [k for k, v in status.items() if not v and k != 'overall']
            print(f"[10/11] Validation: ⚠ Issues with: {', '.join(failed)}")

        # STEP 11: Initialize layers and embeddings dictionary for additional expression matrices
        self._layers = {}
        self._embeddings = {}
        print(f"[11/11] Layers and Embeddings initialized (empty)")
        
        print("="*70)
        self._print_summary()
    
    # ========== Data Preparation Methods ==========
    
    def _prepare_cell_metadata(self, cell_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Prepare and align cell metadata to master index."""
        if cell_metadata is None:
            return pd.DataFrame(index=self._cell_index)
        
        cell_id_col = self.config.cell_id_col
        
        # Set cell ID as index if it's a column
        if cell_id_col in cell_metadata.columns:
            cell_metadata = cell_metadata.set_index(cell_id_col)
        
        # Reindex to master (handles missing and extra cells)
        aligned = cell_metadata.reindex(self._cell_index)
        
        # Report alignment
        n_matched = aligned.notna().any(axis=1).sum()
        n_missing = self._n_cells - n_matched
        n_extra = len(cell_metadata) - n_matched
        
        if n_missing > 0:
            print(f"    ⚠ {n_missing} cells missing metadata (filled with NaN)")
        if n_extra > 0:
            print(f"    ⚠ {n_extra} metadata rows not in expression (dropped)")
        
        return aligned
    
    def _prepare_gene_metadata(self, gene_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Prepare and align gene metadata to master gene index."""
        if gene_metadata is None:
            return pd.DataFrame(index=self._gene_index)
        
        if 'gene' in gene_metadata.columns:
            gene_metadata = gene_metadata.set_index('gene')
        
        aligned = gene_metadata.reindex(self._gene_index)
        
        n_matched = aligned.notna().any(axis=1).sum()
        n_missing = self._n_genes - n_matched
        
        if n_missing > 0:
            print(f"    ⚠ {n_missing} genes missing metadata")
        
        return aligned
    
    def _prepare_spatial_coords(self, 
                                spatial_coords: Optional[Union[Dict, SpatialData, pd.DataFrame]]) -> SpatialData:
        """Prepare spatial coordinates aligned to master index."""
        
        # Case 1: None provided
        if spatial_coords is None:
            return SpatialData(
                x_local=np.full(self._n_cells, np.nan),
                y_local=np.full(self._n_cells, np.nan),
                x_global=np.full(self._n_cells, np.nan),
                y_global=np.full(self._n_cells, np.nan)
            )
        
        # Case 2: Already SpatialData
        if isinstance(spatial_coords, SpatialData):
            if len(spatial_coords) != self._n_cells:
                raise ValueError(
                    f"SpatialData length ({len(spatial_coords)}) != n_cells ({self._n_cells})"
                )
            return spatial_coords
        
        # Case 3: DataFrame
        if isinstance(spatial_coords, pd.DataFrame):
            # Check if indexed by cell IDs
            if spatial_coords.index.name == self.config.cell_id_col:
                # Align to master index
                spatial_coords = spatial_coords.reindex(self._cell_index)
            
            return SpatialData.from_dataframe(spatial_coords, self.config)
        
        # Case 4: Dictionary
        if isinstance(spatial_coords, dict):
            required_keys = ['x_local', 'y_local', 'x_global', 'y_global']
            missing = [k for k in required_keys if k not in spatial_coords]
            if missing:
                raise ValueError(f"Missing spatial keys: {missing}")
            
            # Validate lengths
            for key in required_keys:
                arr = np.array(spatial_coords[key])
                if len(arr) != self._n_cells:
                    raise ValueError(
                        f"Coordinate '{key}' length ({len(arr)}) != n_cells ({self._n_cells})"
                    )
            
            return SpatialData(
                x_local=np.array(spatial_coords['x_local']),
                y_local=np.array(spatial_coords['y_local']),
                x_global=np.array(spatial_coords['x_global']),
                y_global=np.array(spatial_coords['y_global'])
            )
        
        raise TypeError(f"Unsupported spatial_coords type: {type(spatial_coords)}")
    
    def _prepare_polygons(self, polygons: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare polygons aligned to master cell index."""
        if polygons is None:
            return None
        
        cell_id_col = self.config.cell_id_col
        
        if cell_id_col not in polygons.columns:
            raise ValueError(f"Polygons must have '{cell_id_col}' column")
        
        # Keep only polygons for cells in master index
        mask = polygons[cell_id_col].astype(str).isin(self._cell_index)
        aligned = polygons[mask].copy()
        
        n_unique = aligned[cell_id_col].nunique()
        n_dropped = polygons[cell_id_col].nunique() - n_unique
        
        if n_dropped > 0:
            print(f"    ⚠ Dropped polygons for {n_dropped} cells not in expression")
        
        if n_unique < self._n_cells:
            print(f"    ⚠ {self._n_cells - n_unique} cells missing polygon data")
        
        return aligned
    
    def _prepare_image_handler(self,
                               images: Optional[Union[Dict, ImageHandler]],
                               images_folder: Optional[str],
                               pattern: str,
                               extensions: List[str],
                               lazy_load: bool,
                               cache_size: Optional[int]) -> ImageHandler:
        """Initialize ImageHandler from various sources."""
        
        # Case 1: ImageHandler provided
        if isinstance(images, ImageHandler):
            return images
        
        # Create new handler
        handler = ImageHandler(lazy_load=lazy_load, cache_size=cache_size)
        
        # Case 2: Dict of images
        if isinstance(images, dict):
            for fov_id, img in images.items():
                from .config import ImageMetadata
                handler._images[str(fov_id)] = img
                handler._metadata[str(fov_id)] = ImageMetadata(
                    fov_id=str(fov_id),
                    shape=img.shape,
                    dtype=img.dtype,
                    path=None,
                    is_loaded=True
                )
            return handler
        
        # Case 3: Load from folder
        if images_folder is not None:
            # Get FOV IDs from cell metadata if available
            fov_id_col = self.config.fov_id_col
            if fov_id_col in self._cell_meta.columns:
                cell_fovs = self._cell_meta[fov_id_col].dropna().astype(str).unique().tolist()
                handler.load_from_folder(
                    folder_path=images_folder,
                    fov_ids=cell_fovs,
                    pattern=pattern,
                    extensions=extensions
                )
            else:
                # Auto-detect
                handler.load_from_folder(
                    folder_path=images_folder,
                    fov_ids=None,
                    pattern=pattern,
                    extensions=extensions
                )
        
        return handler
    
    def _prepare_fov_system(self, 
                           fov_positions: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Index]:
        """Prepare FOV system ensuring consistency."""
        fov_id_col = self.config.fov_id_col
        
        # STEP 1: Get FOVs from cells
        cell_fovs = set()
        if fov_id_col in self._cell_meta.columns:
            cell_fovs = set(self._cell_meta[fov_id_col].dropna().astype(str).unique())
        
        # STEP 2: Get FOVs from images
        image_fovs = set(self._image_handler.fov_ids)
        
        # STEP 3: Combine FOVs
        if cell_fovs and image_fovs:
            only_cells = cell_fovs - image_fovs
            only_images = image_fovs - cell_fovs
            
            if only_cells:
                print(f"    ⚠ {len(only_cells)} FOVs have cells but no images")
            if only_images:
                print(f"    ⚠ {len(only_images)} FOVs have images but no cells")
            
            all_fovs = cell_fovs | image_fovs
        elif cell_fovs:
            all_fovs = cell_fovs
        elif image_fovs:
            all_fovs = image_fovs
            # Assign cells to first FOV
            if not self._cell_meta.empty:
                self._cell_meta[fov_id_col] = sorted(all_fovs)[0]
        else:
            # No FOVs - create default
            all_fovs = {'0'}
            if not self._cell_meta.empty:
                self._cell_meta[fov_id_col] = '0'
        
        # Create master FOV index
        fov_master_index = pd.Index(sorted(all_fovs), name=fov_id_col)
        
        # STEP 4: Prepare FOV positions
        if fov_positions is None:
            if self._image_handler.n_total > 0:
                fov_positions = load_fov_positions_from_images(self._image_handler)
            else:
                fov_positions = pd.DataFrame(index=fov_master_index)
        else:
            if fov_id_col in fov_positions.columns:
                fov_positions = fov_positions.set_index(fov_id_col)
            
            fov_positions.index = fov_positions.index.astype(str)
            fov_positions = fov_positions.reindex(fov_master_index)
        
        # STEP 5: Filter ImageHandler to master FOVs
        if self._image_handler.n_total > 0:
            self._image_handler = self._image_handler.subset(fov_master_index.tolist())
        
        return fov_positions, fov_master_index
    
    def _create_index_maps(self) -> None:
        """Create fast lookup maps."""
        # Cell ID -> position
        self._cell_id_to_idx = {cid: idx for idx, cid in enumerate(self._cell_index)}
        
        # Gene name -> position
        self._gene_name_to_idx = {gene: idx for idx, gene in enumerate(self._gene_index)}
        
        # FOV ID -> position
        self._fov_id_to_idx = {fid: idx for idx, fid in enumerate(self._fov_index)}
        
        # FOV -> cells mapping
        fov_id_col = self.config.fov_id_col
        if fov_id_col in self._cell_meta.columns:
            self._fov_to_cells = {}
            for fov_id in self._fov_index:
                mask = self._cell_meta[fov_id_col].astype(str) == str(fov_id)
                self._fov_to_cells[str(fov_id)] = np.where(mask)[0]
        else:
            self._fov_to_cells = {}
    
    # ========== Consistency Validation ==========
    
    def validate_consistency(self, raise_error: bool = False) -> Dict[str, bool]:
        """
        Validate all components are consistent.
        
        Parameters
        ----------
        raise_error : bool
            If True, raise error on inconsistency
        
        Returns
        -------
        dict
            Status of each component
        """
        status = {}
        issues = []
        
        # Expression matrix
        status['expression_cells'] = self._expression.shape[0] == self._n_cells
        status['expression_genes'] = self._expression.shape[1] == self._n_genes
        
        if not status['expression_cells']:
            issues.append(f"Expression has {self._expression.shape[0]} cells, expected {self._n_cells}")
        if not status['expression_genes']:
            issues.append(f"Expression has {self._expression.shape[1]} genes, expected {self._n_genes}")
        
        # Cell metadata
        status['cell_metadata'] = len(self._cell_meta) == self._n_cells
        if not status['cell_metadata']:
            issues.append(f"Cell metadata has {len(self._cell_meta)} rows, expected {self._n_cells}")
        
        # Gene metadata
        status['gene_metadata'] = len(self._gene_meta) == self._n_genes
        if not status['gene_metadata']:
            issues.append(f"Gene metadata has {len(self._gene_meta)} rows, expected {self._n_genes}")
        
        # Spatial coordinates
        status['spatial'] = len(self._spatial) == self._n_cells
        if not status['spatial']:
            issues.append(f"Spatial coords have {len(self._spatial)} cells, expected {self._n_cells}")
        
        # Polygons
        if self._polygons is not None:
            cell_id_col = self.config.cell_id_col
            poly_cells = set(self._polygons[cell_id_col].astype(str))
            extra = poly_cells - set(self._cell_index)
            status['polygons'] = len(extra) == 0
            if not status['polygons']:
                issues.append(f"Polygons have {len(extra)} cells not in master index")
        else:
            status['polygons'] = True
        
        # FOV system
        fov_id_col = self.config.fov_id_col
        if fov_id_col in self._cell_meta.columns:
            cell_fovs = set(self._cell_meta[fov_id_col].dropna().astype(str))
            master_fovs = set(str(f) for f in self._fov_index)
            
            # Cell FOVs should be subset of master
            status['cell_fovs'] = cell_fovs.issubset(master_fovs)
            if not status['cell_fovs']:
                extra = cell_fovs - master_fovs
                issues.append(f"Cells reference {len(extra)} FOVs not in master index")
        else:
            status['cell_fovs'] = False
            issues.append(f"Cell metadata missing '{fov_id_col}' column")
        
        # FOV positions
        fov_pos_ids = set(self._fov_positions.index.astype(str))
        master_fovs = set(str(f) for f in self._fov_index)
        status['fov_positions'] = fov_pos_ids == master_fovs
        if not status['fov_positions']:
            issues.append("FOV positions index doesn't match master FOV index")
        
        # Images
        image_fovs = set(self._image_handler.fov_ids)
        extra_images = image_fovs - master_fovs
        status['images'] = len(extra_images) == 0
        if not status['images']:
            issues.append(f"Images have {len(extra_images)} FOVs not in master index")
        
        # Overall
        status['overall'] = all(status.values())
        
        if issues and raise_error:
            raise ConsistencyError("\n".join(issues))
        
        return status
    
    # ========== Properties ==========
    
    @property
    def cell_index(self) -> pd.Index:
        """Get master cell index."""
        return self._cell_index
    
    @property
    def gene_index(self) -> pd.Index:
        """Get master gene index."""
        return self._gene_index
    
    @property
    def fov_index(self) -> pd.Index:
        """Get master FOV index."""
        return self._fov_index
    
    @property
    def n_cells(self) -> int:
        """Get number of cells."""
        return self._n_cells
    
    @property
    def n_genes(self) -> int:
        """Get number of genes."""
        return self._n_genes
    
    @property
    def n_fovs(self) -> int:
        """Get number of FOVs."""
        return len(self._fov_index)
    
    @property
    def expression(self) -> ExpressionMatrix:
        """Get expression matrix."""
        return self._expression
    
    @property
    def cell_meta(self) -> pd.DataFrame:
        """Get cell metadata (aligned to master index)."""
        return self._cell_meta
    
    @property
    def gene_meta(self) -> pd.DataFrame:
        """Get gene metadata (aligned to master index)."""
        return self._gene_meta
    
    @property
    def spatial(self) -> SpatialData:
        """Get spatial coordinates."""
        return self._spatial
    
    @property
    def polygons(self) -> Optional[pd.DataFrame]:
        """Get cell polygons."""
        return self._polygons
    
    @property
    def fov_positions(self) -> pd.DataFrame:
        """Get FOV positions."""
        return self._fov_positions
    
    @property
    def images(self) -> ImageHandler:
        """Get image handler."""
        return self._image_handler

    @property
    def layers(self) -> Dict[str, Union[np.ndarray, sparse.spmatrix]]:
        """
        Get dictionary of expression layers (e.g., normalized, scaled).
        
        Returns
        -------
        dict
            Dictionary mapping layer names to expression matrices
        """
        return self._layers

    @property
    def embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get dictionary of low-dimensional embeddings (PCA, t-SNE, UMAP, etc.).
        
        Returns
        -------
        dict
            Dictionary with embedding names as keys and coordinate arrays as values
        """
        if not hasattr(self, '_embeddings'):
            self._embeddings = {}
        return self._embeddings
    
    def _print_summary(self) -> None:
        """Print initialization summary."""
        print("\nSpatioloji Object Summary:")
        print(f"  Cells:              {self._n_cells:,}")
        print(f"  Genes:              {self._n_genes:,}")
        print(f"  FOVs:               {len(self._fov_index)}")
        print(f"  Images:             {self._image_handler.n_total} "
              f"({self._image_handler.n_loaded} loaded)")
        print(f"  Original Expression:         {'sparse' if self._expression.is_sparse else 'dense'}")
        print(f"  Has polygons:       {self._polygons is not None}")
        print(f"  Memory (approx):    {self._estimate_memory_usage():.1f} MB")
        print()
    
    def _estimate_memory_usage(self) -> float:
        """Estimate total memory usage in MB."""
        total_bytes = 0
        
        # Expression
        total_bytes += self._expression.memory_usage_mb() * 1024 * 1024
        
        # Spatial
        total_bytes += self._spatial.x_local.nbytes * 4  # 4 arrays
        
        # Cell metadata
        total_bytes += self._cell_meta.memory_usage(deep=True).sum()
        
        # Gene metadata
        total_bytes += self._gene_meta.memory_usage(deep=True).sum()
        
        # Polygons
        if self._polygons is not None:
            total_bytes += self._polygons.memory_usage(deep=True).sum()
        
        # FOV positions
        total_bytes += self._fov_positions.memory_usage(deep=True).sum()
        
        # Loaded images
        for img in self._image_handler._images.values():
            total_bytes += img.nbytes
        
        return total_bytes / (1024 * 1024)
    
    def _get_cell_indices(self, cell_ids: Union[List[str], np.ndarray, pd.Index]) -> np.ndarray:
        """
        Convert cell IDs to integer indices.
        
        Parameters
        ----------
        cell_ids : list, array, or Index
            Cell identifiers
        
        Returns
        -------
        np.ndarray
            Integer indices (positions in master index)
        """
        cell_ids = pd.Index(cell_ids).astype(str)
        
        # Fast lookup using map
        indices = [self._cell_id_to_idx[cid] for cid in cell_ids 
                  if cid in self._cell_id_to_idx]
        
        if len(indices) != len(cell_ids):
            n_missing = len(cell_ids) - len(indices)
            print(f"  ⚠ {n_missing} cell IDs not found")
        
        return np.array(indices, dtype=np.int64)
    
    def _get_gene_indices(self, gene_names: Union[List[str], np.ndarray, pd.Index]) -> np.ndarray:
        """
        Convert gene names to integer indices.
        
        Parameters
        ----------
        gene_names : list, array, or Index
            Gene names
        
        Returns
        -------
        np.ndarray
            Integer indices
        """
        gene_names = pd.Index(gene_names).astype(str)
        
        indices = [self._gene_name_to_idx[gene] for gene in gene_names 
                  if gene in self._gene_name_to_idx]
        
        if len(indices) != len(gene_names):
            n_missing = len(gene_names) - len(indices)
            print(f"  ⚠ {n_missing} gene names not found")
        
        return np.array(indices, dtype=np.int64)
    
    def _get_fov_indices(self, fov_ids: Union[List[str], np.ndarray, pd.Index]) -> np.ndarray:
        """
        Convert FOV IDs to integer indices.
        
        Parameters
        ----------
        fov_ids : list, array, or Index
            FOV identifiers
        
        Returns
        -------
        np.ndarray
            Integer indices
        """
        fov_ids = pd.Index(fov_ids).astype(str)
        
        indices = [self._fov_id_to_idx[fid] for fid in fov_ids 
                  if fid in self._fov_id_to_idx]
        
        if len(indices) != len(fov_ids):
            n_missing = len(fov_ids) - len(indices)
            print(f"  ⚠ {n_missing} FOV IDs not found")
        
        return np.array(indices, dtype=np.int64)
    
    # ========== Subsetting Methods ==========
    
    def subset_by_cells(self, 
                       cell_ids: Union[List[str], np.ndarray, pd.Index],
                       copy: bool = True) -> 'spatioloji':
        """
        Create new spatioloji with subset of cells.
        
        All components are automatically subsetted and kept consistent:
        - Expression matrix
        - Cell metadata
        - Spatial coordinates
        - Polygons
        - FOVs (only FOVs with remaining cells)
        - Images (only for remaining FOVs)
        
        Parameters
        ----------
        cell_ids : list, array, or Index
            Cell IDs to keep
        copy : bool
            If True, create deep copy of data
        
        Returns
        -------
        spatioloji
            New object with subset of cells
        """
        print(f"\nSubsetting by cells: {len(cell_ids)} cells requested")
        
        # Get integer indices
        indices = self._get_cell_indices(cell_ids)
        
        if len(indices) == 0:
            raise ValueError("No valid cell IDs found")
        
        # Subset expression
        subset_expr = self._expression.subset_cells(indices)
        
        # Subset cell metadata
        subset_cell_meta = self._cell_meta.iloc[indices].copy() if copy else self._cell_meta.iloc[indices]
        
        # Subset spatial coordinates
        subset_spatial = self._spatial.subset(indices)
        
        # Subset polygons
        if self._polygons is not None:
            cell_id_col = self.config.cell_id_col
            keep_cells = self._cell_index[indices]
            mask = self._polygons[cell_id_col].isin(keep_cells)
            subset_polygons = self._polygons[mask].copy() if copy else self._polygons[mask]
        else:
            subset_polygons = None
        
        # Determine which FOVs to keep
        fov_id_col = self.config.fov_id_col
        if fov_id_col in subset_cell_meta.columns:
            kept_fovs = subset_cell_meta[fov_id_col].dropna().unique()
            subset_fov_positions = self._fov_positions.loc[kept_fovs].copy() if copy else self._fov_positions.loc[kept_fovs]
            subset_image_handler = self._image_handler.subset(kept_fovs.tolist())
        else:
            subset_fov_positions = self._fov_positions.copy() if copy else self._fov_positions
            subset_image_handler = self._image_handler
        
        # Create new spatioloji object
        new_obj = spatioloji(
            expression=subset_expr,
            cell_ids=self._cell_index[indices],
            gene_names=self._gene_index,
            cell_metadata=subset_cell_meta,
            gene_metadata=self._gene_meta.copy() if copy else self._gene_meta,
            spatial_coords=subset_spatial,
            polygons=subset_polygons,
            fov_positions=subset_fov_positions,
            images=subset_image_handler,
            lazy_load_images=self._image_handler.lazy_load,
            image_cache_size=self._image_handler.cache_size,
            config=self.config
        )
        
        print(f"✓ Created subset: {new_obj.n_cells} cells, {new_obj.n_fovs} FOVs")
        
        return new_obj
    
    def subset_by_genes(self,
                       gene_names: Union[List[str], np.ndarray, pd.Index],
                       copy: bool = True) -> 'spatioloji':
        """
        Create new spatioloji with subset of genes.
        
        Parameters
        ----------
        gene_names : list, array, or Index
            Gene names to keep
        copy : bool
            If True, create deep copy
        
        Returns
        -------
        spatioloji
            New object with subset of genes
        """
        print(f"\nSubsetting by genes: {len(gene_names)} genes requested")
        
        # Get integer indices
        indices = self._get_gene_indices(gene_names)
        
        if len(indices) == 0:
            raise ValueError("No valid gene names found")
        
        # Subset expression
        subset_expr = self._expression.subset_genes(indices)
        
        # Subset gene metadata
        subset_gene_meta = self._gene_meta.iloc[indices].copy() if copy else self._gene_meta.iloc[indices]
        
        # Create new object (cell data unchanged)
        new_obj = spatioloji(
            expression=subset_expr,
            cell_ids=self._cell_index,
            gene_names=self._gene_index[indices],
            cell_metadata=self._cell_meta.copy() if copy else self._cell_meta,
            gene_metadata=subset_gene_meta,
            spatial_coords=self._spatial,
            polygons=self._polygons.copy() if copy and self._polygons is not None else self._polygons,
            fov_positions=self._fov_positions.copy() if copy else self._fov_positions,
            images=self._image_handler,
            lazy_load_images=self._image_handler.lazy_load,
            image_cache_size=self._image_handler.cache_size,
            config=self.config
        )
        
        print(f"✓ Created subset: {new_obj.n_genes} genes")
        
        return new_obj
    
    def subset_by_fovs(self,
                      fov_ids: Union[List[str], np.ndarray, pd.Index],
                      copy: bool = True) -> 'spatioloji':
        """
        Create new spatioloji with subset of FOVs.
        
        Automatically subsets:
        - Cells belonging to these FOVs
        - FOV positions
        - Images
        
        Parameters
        ----------
        fov_ids : list, array, or Index
            FOV IDs to keep
        copy : bool
            If True, create deep copy
        
        Returns
        -------
        spatioloji
            New object with subset of FOVs
        """
        print(f"\nSubsetting by FOVs: {len(fov_ids)} FOVs requested")
        
        fov_ids = [str(fid) for fid in fov_ids]
        fov_id_col = self.config.fov_id_col
        
        if fov_id_col not in self._cell_meta.columns:
            raise ValueError(f"Cannot subset by FOVs: '{fov_id_col}' not in cell metadata")
        
        # Get cells in these FOVs
        mask = self._cell_meta[fov_id_col].astype(str).isin(fov_ids)
        cell_indices = np.where(mask)[0]
        
        if len(cell_indices) == 0:
            raise ValueError("No cells found in specified FOVs")
        
        # Use subset_by_cells (which handles everything)
        subset_cell_ids = self._cell_index[cell_indices]
        
        return self.subset_by_cells(subset_cell_ids, copy=copy)
    
    # ========== Access Methods ==========
    
    def get_cells_in_fov(self, fov_id: str) -> pd.Index:
        """
        Get cell IDs for a specific FOV.
        
        Fast O(1) lookup using pre-computed mapping.
        
        Parameters
        ----------
        fov_id : str
            FOV identifier
        
        Returns
        -------
        pd.Index
            Cell IDs in this FOV
        """
        fov_id = str(fov_id)
        
        if fov_id in self._fov_to_cells:
            indices = self._fov_to_cells[fov_id]
            return self._cell_index[indices]
        
        return pd.Index([])
    
    def get_fov_for_cell(self, cell_id: str) -> Optional[str]:
        """
        Get FOV ID for a specific cell.
        
        Parameters
        ----------
        cell_id : str
            Cell identifier
        
        Returns
        -------
        str or None
            FOV ID
        """
        fov_id_col = self.config.fov_id_col
        
        if cell_id in self._cell_index and fov_id_col in self._cell_meta.columns:
            return str(self._cell_meta.loc[cell_id, fov_id_col])
        
        return None
    
    def get_expression(self, 
                      cell_ids: Optional[Union[List[str], str]] = None,
                      gene_names: Optional[Union[List[str], str]] = None,
                      as_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get expression data for specific cells and/or genes.
        
        Parameters
        ----------
        cell_ids : list or str, optional
            Cell ID(s). If None, all cells.
        gene_names : list or str, optional
            Gene name(s). If None, all genes.
        as_dataframe : bool
            If True, return as DataFrame with labels
        
        Returns
        -------
        np.ndarray or pd.DataFrame
            Expression data
        """
        # Handle single values
        if isinstance(cell_ids, str):
            cell_ids = [cell_ids]
        if isinstance(gene_names, str):
            gene_names = [gene_names]
        
        # Get indices
        if cell_ids is not None:
            cell_indices = self._get_cell_indices(cell_ids)
        else:
            cell_indices = np.arange(self._n_cells)
        
        if gene_names is not None:
            gene_indices = self._get_gene_indices(gene_names)
        else:
            gene_indices = np.arange(self._n_genes)
        
        # Extract data
        if self._expression.is_sparse:
            data = self._expression._data[np.ix_(cell_indices, gene_indices)]
            if not sparse.issparse(data):
                data = data.toarray()
            else:
                data = data.toarray()
        else:
            data = self._expression._data[np.ix_(cell_indices, gene_indices)]
        
        if as_dataframe:
            return pd.DataFrame(
                data,
                index=self._cell_index[cell_indices],
                columns=self._gene_index[gene_indices]
            )
        
        return data
    
    def get_spatial_coords(self,
                          cell_ids: Optional[Union[List[str], str]] = None,
                          coord_type: str = 'local',
                          as_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Get spatial coordinates for cells.
        
        Parameters
        ----------
        cell_ids : list or str, optional
            Cell ID(s). If None, all cells.
        coord_type : str
            'local' or 'global'
        as_dataframe : bool
            If True, return as DataFrame
        
        Returns
        -------
        np.ndarray or pd.DataFrame
            Spatial coordinates (N × 2)
        """
        if isinstance(cell_ids, str):
            cell_ids = [cell_ids]
        
        if cell_ids is not None:
            indices = self._get_cell_indices(cell_ids)
        else:
            indices = np.arange(self._n_cells)
        
        if coord_type == 'local':
            x = self._spatial.x_local[indices]
            y = self._spatial.y_local[indices]
        elif coord_type == 'global':
            x = self._spatial.x_global[indices]
            y = self._spatial.y_global[indices]
        else:
            raise ValueError(f"Invalid coord_type: {coord_type}")
        
        coords = np.column_stack([x, y])
        
        if as_dataframe:
            x_col, y_col = self.config.get_coordinate_columns(coord_type)
            return pd.DataFrame(
                coords,
                index=self._cell_index[indices],
                columns=[x_col, y_col]
            )
        
        return coords
    
    def get_polygon_for_cell(self, cell_id: str) -> Optional[pd.DataFrame]:
        """
        Get polygon data for a specific cell.
        
        Parameters
        ----------
        cell_id : str
            Cell identifier
        
        Returns
        -------
        pd.DataFrame or None
            Polygon vertices
        """
        if self._polygons is None:
            return None
        
        cell_id_col = self.config.cell_id_col
        mask = self._polygons[cell_id_col] == cell_id
        
        return self._polygons[mask] if mask.any() else None
        
    def to_geopandas(self, 
                 coord_type: str = 'global',
                 include_metadata: bool = True) -> 'gpd.GeoDataFrame':
        """
        Convert polygon data to GeoPandas GeoDataFrame.
        
        Parameters
        ----------
        coord_type : str
            'local' or 'global' coordinates
        include_metadata : bool
            Whether to include cell metadata columns
        
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with Polygon geometries, one row per cell
        
        Examples
        --------
        >>> # Get polygons as GeoDataFrame
        >>> gdf = sp.to_geopandas(coord_type='global', include_metadata=True)
        >>> gdf.plot()
        >>> 
        >>> # Export to shapefile
        >>> gdf.to_file('cells.shp')
        >>> 
        >>> # Export to GeoJSON
        >>> gdf.to_file('cells.geojson', driver='GeoJSON')
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except ImportError:
            raise ImportError(
                "GeoPandas required. Install with: pip install geopandas"
            )
        
        if self._polygons is None:
            raise ValueError("No polygon data available")
        
        # Get coordinate columns
        cell_id_col = self.config.cell_id_col
        x_col, y_col = self.config.get_coordinate_columns(coord_type)
        
        print(f"Converting to GeoDataFrame ({coord_type} coordinates)...")
        
        # Create Polygon geometries
        geometries = []
        cell_ids = []
        
        for cell_id, group in self._polygons.groupby(cell_id_col):
            coords = list(zip(group[x_col], group[y_col]))
            geometries.append(Polygon(coords))
            cell_ids.append(cell_id)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {cell_id_col: cell_ids},
            geometry=geometries,
            crs=None
        )
        
        # Set index
        gdf = gdf.set_index(cell_id_col).reindex(self._cell_index)
        
        # Add metadata if requested
        if include_metadata:
            for col in self._cell_meta.columns:
                if col not in gdf.columns:
                    gdf[col] = self._cell_meta[col]
        
        print(f"  ✓ Created GeoDataFrame: {len(gdf)} cells")
        
        return gdf
        
    def get_image(self, fov_id: str) -> Optional[np.ndarray]:
        """
        Get image for a specific FOV.
        
        Loads on-demand if lazy loading enabled.
        
        Parameters
        ----------
        fov_id : str
            FOV identifier
        
        Returns
        -------
        np.ndarray or None
            Image array
        """
        return self._image_handler.get(fov_id)
    
    # ========== Image Management ==========
    
    def load_all_images(self) -> None:
        """Load all images into memory."""
        print(f"\nLoading all {self._image_handler.n_total} images...")
        self._image_handler.load_all()
        print(f"✓ Loaded {self._image_handler.n_loaded} images")
    
    def unload_all_images(self) -> None:
        """Unload all images from memory."""
        print(f"\nUnloading all images...")
        self._image_handler.unload_all()
        print(f"✓ Images unloaded (lazy loading available)")
    
    # ========== Summary & Info Methods ==========
    
    def summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of the object.
        
        Returns
        -------
        dict
            Summary statistics
        """
        fov_id_col = self.config.fov_id_col
        
        # Cells per FOV
        if fov_id_col in self._cell_meta.columns:
            cells_per_fov = self._cell_meta[fov_id_col].value_counts()
            avg_cells_per_fov = cells_per_fov.mean()
            min_cells_per_fov = cells_per_fov.min()
            max_cells_per_fov = cells_per_fov.max()
        else:
            avg_cells_per_fov = 0
            min_cells_per_fov = 0
            max_cells_per_fov = 0
        
        # Expression stats
        if self._expression.is_sparse:
            expr_data = self._expression._data
            sparsity = 1 - (expr_data.nnz / (expr_data.shape[0] * expr_data.shape[1]))
        else:
            sparsity = 1 - np.count_nonzero(self._expression._data) / self._expression._data.size
        
        return {
            'n_cells': self._n_cells,
            'n_genes': self._n_genes,
            'n_fovs': len(self._fov_index),
            'n_images': self._image_handler.n_total,
            'n_images_loaded': self._image_handler.n_loaded,
            'avg_cells_per_fov': avg_cells_per_fov,
            'min_cells_per_fov': min_cells_per_fov,
            'max_cells_per_fov': max_cells_per_fov,
            'has_polygons': self._polygons is not None,
            'n_polygon_vertices': len(self._polygons) if self._polygons is not None else 0,
            'is_sparse': self._expression.is_sparse,
            'sparsity': sparsity,
            'memory_usage_mb': self._estimate_memory_usage(),
            'cell_meta_columns': list(self._cell_meta.columns),
            'gene_meta_columns': list(self._gene_meta.columns),
            'fov_positions_columns': list(self._fov_positions.columns)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"spatioloji object\n"
                f"  Cells: {self._n_cells:,}\n"
                f"  Genes: {self._n_genes:,}\n"
                f"  FOVs:  {len(self._fov_index)}\n"
                f"  Images: {self._image_handler.n_total} "
                f"({self._image_handler.n_loaded} loaded)")
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()
      
    # ========== Serialization Methods ==========
    
    def to_pickle(self, filepath: str) -> None:
        """
        Save spatioloji object to pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving spatioloji to: {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Saved ({file_size_mb:.1f} MB)")
    
    @staticmethod
    def from_pickle(filepath: str) -> 'spatioloji':
        """
        Load spatioloji object from pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to pickle file
        
        Returns
        -------
        spatioloji
            Loaded object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"\nLoading spatioloji from: {filepath}")
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        if not isinstance(obj, spatioloji):
            raise TypeError(f"File does not contain a spatioloji object")
        
        print(f"✓ Loaded: {obj.n_cells:,} cells × {obj.n_genes:,} genes")
        
        return obj
    
    def save_components(self, output_dir: str) -> None:
        """
        Save all components to separate files.
        
        Creates directory structure:
        output_dir/
          ├── expression.npz (or .npy)
          ├── cell_metadata.csv
          ├── gene_metadata.csv
          ├── spatial_coords.csv
          ├── polygons.csv
          ├── fov_positions.csv
          └── config.pkl
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving components to: {output_dir}")
        
        # Save expression matrix
        if self._expression.is_sparse:
            sparse.save_npz(
                output_path / 'expression.npz',
                self._expression.get_sparse()
            )
            print("  ✓ expression.npz (sparse)")
        else:
            np.save(
                output_path / 'expression.npy',
                self._expression.get_dense()
            )
            print("  ✓ expression.npy (dense)")
        
        # Save indices
        pd.DataFrame({'cell_id': self._cell_index}).to_csv(
            output_path / 'cell_ids.csv', index=False
        )
        pd.DataFrame({'gene': self._gene_index}).to_csv(
            output_path / 'gene_names.csv', index=False
        )
        print("  ✓ cell_ids.csv, gene_names.csv")
        
        # Save metadata
        self._cell_meta.to_csv(output_path / 'cell_metadata.csv')
        self._gene_meta.to_csv(output_path / 'gene_metadata.csv')
        print("  ✓ cell_metadata.csv, gene_metadata.csv")
        
        # Save spatial coordinates
        spatial_df = pd.DataFrame(
            self._spatial.to_dict(),
            index=self._cell_index
        )
        spatial_df.to_csv(output_path / 'spatial_coords.csv')
        print("  ✓ spatial_coords.csv")
        
        # Save polygons
        if self._polygons is not None:
            self._polygons.to_csv(output_path / 'polygons.csv', index=False)
            print("  ✓ polygons.csv")
        
        # Save FOV positions
        self._fov_positions.to_csv(output_path / 'fov_positions.csv')
        print("  ✓ fov_positions.csv")
        
        # Save config
        with open(output_path / 'config.pkl', 'wb') as f:
            pickle.dump(self.config, f)
        print("  ✓ config.pkl")
        
        # Save image metadata (not actual images)
        image_meta_list = []
        for fov_id in self._image_handler.fov_ids:
            meta = self._image_handler.get_metadata(fov_id)
            if meta:
                image_meta_list.append({
                    'fov_id': fov_id,
                    'path': str(meta.path) if meta.path else None,
                    'shape': meta.shape,
                    'dtype': str(meta.dtype)
                })
        
        if image_meta_list:
            pd.DataFrame(image_meta_list).to_csv(
                output_path / 'image_metadata.csv', index=False
            )
            print("  ✓ image_metadata.csv")
        
        print(f"\n✓ All components saved to {output_dir}")
        
    # ========== Layer Methods ==========
    def add_layer(self, 
                  layer_name: str,
                  data: Union[np.ndarray, sparse.spmatrix],
                  overwrite: bool = False) -> None:
        """
        Add a new expression layer (e.g., normalized, scaled data).
        
        Parameters
        ----------
        layer_name : str
            Name for this layer (e.g., 'normalized', 'scaled', 'log_normalized')
        data : np.ndarray or sparse matrix
            Expression matrix with same shape as main expression
            (n_cells × n_genes)
        overwrite : bool
            Whether to overwrite if layer already exists
        
        Raises
        ------
        ValueError
            If shape doesn't match or layer exists without overwrite
        
        Examples
        --------
        >>> # Add log-normalized data
        >>> import scanpy as sc
        >>> expr_dense = sp.expression.get_dense()
        >>> normalized = sc.pp.normalize_total(expr_dense, target_sum=1e4, inplace=False)
        >>> normalized_log = np.log1p(normalized)
        >>> sp.add_layer('normalized', normalized_log)
        
        >>> # Add scaled data
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaled = scaler.fit_transform(expr_dense)
        >>> sp.add_layer('scaled', scaled)
        """
        # Validate shape
        if isinstance(data, np.ndarray):
            if data.shape != (self._n_cells, self._n_genes):
                raise ValueError(
                    f"Layer shape {data.shape} doesn't match "
                    f"expression shape ({self._n_cells}, {self._n_genes})"
                )
        elif sparse.issparse(data):
            if data.shape != (self._n_cells, self._n_genes):
                raise ValueError(
                    f"Layer shape {data.shape} doesn't match "
                    f"expression shape ({self._n_cells}, {self._n_genes})"
                )
        else:
            raise TypeError("Layer data must be numpy array or sparse matrix")
        
        # Check if exists
        if layer_name in self._layers and not overwrite:
            raise ValueError(
                f"Layer '{layer_name}' already exists. "
                "Use overwrite=True to replace it."
            )
        
        self._layers[layer_name] = data
        print(f"✓ Added layer '{layer_name}' "
              f"({'sparse' if sparse.issparse(data) else 'dense'}, "
              f"{data.nbytes / (1024**2):.1f} MB)")
    
    def remove_layer(self, layer_name: str) -> None:
        """
        Remove an expression layer.
        
        Parameters
        ----------
        layer_name : str
            Name of layer to remove
        """
        if layer_name not in self._layers:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        del self._layers[layer_name]
        print(f"✓ Removed layer '{layer_name}'")
    
    def get_layer(self, layer_name: str) -> Union[np.ndarray, sparse.spmatrix]:
        """
        Get a specific expression layer.
        
        Parameters
        ----------
        layer_name : str
            Name of layer to retrieve
        
        Returns
        -------
        np.ndarray or sparse matrix
            Expression matrix for this layer
        """
        if layer_name not in self._layers:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {list(self._layers.keys())}"
            )
        
        return self._layers[layer_name]
    
    def list_layers(self) -> List[str]:
        """
        List all available expression layers.
        
        Returns
        -------
        list
            Names of all layers
        """
        return list(self._layers.keys())
    
    # ========== Factory Methods ==========
    
    @staticmethod
    def from_components(output_dir: str,
                       images_folder: Optional[str] = None,
                       lazy_load_images: bool = True) -> 'spatioloji':
        """
        Load spatioloji from component files.
        
        Parameters
        ----------
        output_dir : str
            Directory with component files
        images_folder : str, optional
            Path to images folder
        lazy_load_images : bool
            Enable lazy image loading
        
        Returns
        -------
        spatioloji
            Loaded object
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Directory not found: {output_dir}")
        
        print(f"\nLoading spatioloji from components: {output_dir}")
        
        # Load expression
        if (output_path / 'expression.npz').exists():
            expression = sparse.load_npz(output_path / 'expression.npz')
            print("  ✓ Loaded expression.npz (sparse)")
        elif (output_path / 'expression.npy').exists():
            expression = np.load(output_path / 'expression.npy')
            print("  ✓ Loaded expression.npy (dense)")
        else:
            raise FileNotFoundError("Expression file not found")
        
        # Load indices
        cell_ids = pd.read_csv(output_path / 'cell_ids.csv')['cell_id'].values
        gene_names = pd.read_csv(output_path / 'gene_names.csv')['gene'].values
        print("  ✓ Loaded cell_ids.csv, gene_names.csv")
        
        # Load metadata
        cell_metadata = pd.read_csv(output_path / 'cell_metadata.csv', index_col=0)
        gene_metadata = pd.read_csv(output_path / 'gene_metadata.csv', index_col=0)
        print("  ✓ Loaded metadata files")
        
        # Load spatial coords
        spatial_coords = pd.read_csv(output_path / 'spatial_coords.csv', index_col=0)
        print("  ✓ Loaded spatial_coords.csv")
        
        # Load polygons
        polygons_path = output_path / 'polygons.csv'
        polygons = pd.read_csv(polygons_path) if polygons_path.exists() else None
        
        # Load FOV positions
        fov_positions = pd.read_csv(output_path / 'fov_positions.csv', index_col=0)
        print("  ✓ Loaded fov_positions.csv")
        
        # Load config
        config_path = output_path / 'config.pkl'
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
        else:
            config = None
        
        # Create object
        return spatioloji(
            expression=expression,
            cell_ids=cell_ids,
            gene_names=gene_names,
            cell_metadata=cell_metadata,
            gene_metadata=gene_metadata,
            spatial_coords=spatial_coords,
            polygons=polygons,
            fov_positions=fov_positions,
            images_folder=images_folder,
            lazy_load_images=lazy_load_images,
            config=config
        )
    
    @staticmethod
    def from_files(polygons_path: str,
                  cell_meta_path: str,
                  expression_path: str,
                  fov_positions_path: str,
                  gene_names: Optional[List[str]] = None,
                  images_folder: Optional[str] = None,
                  image_pattern: str = "CellComposite_F{fov_id}.{ext}",
                  image_extensions: List[str] = ['jpg', 'png', 'tif', 'tiff'],
                  lazy_load_images: bool = True,
                  config: Optional[SpatiolojiConfig] = None) -> 'spatioloji':
        """
        Create spatioloji from separate data files.
        
        Parameters
        ----------
        polygons_path : str
            Path to CSV with polygon data
        cell_meta_path : str
            Path to CSV with cell metadata
        expression_path : str
            Path to expression file (CSV, NPY, or NPZ)
        fov_positions_path : str
            Path to CSV with FOV positions
        gene_names : list, optional
            Gene names (required if expression is matrix without column names)
        images_folder : str, optional
            Path to images folder
        image_pattern : str
            Image filename pattern
        image_extensions : list
            Image file extensions
        lazy_load_images : bool
            Enable lazy loading
        config : SpatiolojiConfig, optional
            Configuration
        
        Returns
        -------
        spatioloji
            New spatioloji object
        """
        print("\n" + "="*70)
        print("Loading spatioloji from files")
        print("="*70)
        
        config = config or SpatiolojiConfig()
        
        # Load polygons
        print(f"Loading polygons: {polygons_path}")
        polygons = pd.read_csv(polygons_path)
        
        # Load cell metadata
        print(f"Loading cell metadata: {cell_meta_path}")
        cell_meta = pd.read_csv(cell_meta_path)
        
        # Extract spatial coordinates from polygons
        print("Extracting spatial coordinates from polygons...")
        spatial_coords = extract_spatial_coords_from_polygons(polygons, config)
        
        # Load expression
        print(f"Loading expression: {expression_path}")
        expression, cell_ids, genes = load_expression_file(
            expression_path, 
            gene_names=gene_names
        )
        
        # Load FOV positions
        print(f"Loading FOV positions: {fov_positions_path}")
        fov_positions = pd.read_csv(fov_positions_path)
        
        print("="*70)
        
        # Create object
        return spatioloji(
            expression=expression,
            cell_ids=cell_ids,
            gene_names=genes,
            cell_metadata=cell_meta,
            spatial_coords=spatial_coords,
            polygons=polygons,
            fov_positions=fov_positions,
            images_folder=images_folder,
            image_pattern=image_pattern,
            image_extensions=image_extensions,
            lazy_load_images=lazy_load_images,
            config=config
        )
    
    @staticmethod
    def from_anndata(adata,
                    polygons: Optional[pd.DataFrame] = None,
                    fov_positions: Optional[pd.DataFrame] = None,
                    images_folder: Optional[str] = None,
                    spatial_key: str = 'spatial',
                    config: Optional[SpatiolojiConfig] = None) -> 'spatioloji':
        """
        Create spatioloji from AnnData object.
        
        Extracts:
        - Expression: adata.X
        - Cell metadata: adata.obs
        - Gene metadata: adata.var
        - Spatial coords: adata.obsm[spatial_key]
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object
        polygons : pd.DataFrame, optional
            Polygon data
        fov_positions : pd.DataFrame, optional
            FOV positions
        images_folder : str, optional
            Images folder
        spatial_key : str
            Key in obsm for spatial coordinates
        config : SpatiolojiConfig, optional
            Configuration
        
        Returns
        -------
        spatioloji
            New spatioloji object
        """
        import anndata
        
        if not isinstance(adata, anndata.AnnData):
            raise TypeError("Input must be AnnData object")
        
        print(f"\nConverting AnnData to spatioloji...")
        print(f"  AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        config = config or SpatiolojiConfig()
        
        # Extract expression
        expression = adata.X
        
        # Extract IDs
        cell_ids = adata.obs_names
        gene_names = adata.var_names
        
        # Extract metadata
        cell_metadata = adata.obs.copy()
        gene_metadata = adata.var.copy()
        
        # Extract spatial coordinates
        if spatial_key in adata.obsm:
            spatial_array = adata.obsm[spatial_key]
            if spatial_array.shape[1] == 2:
                # Assume global coordinates
                spatial_coords = {
                    'x_local': spatial_array[:, 0],
                    'y_local': spatial_array[:, 1],
                    'x_global': spatial_array[:, 0],
                    'y_global': spatial_array[:, 1]
                }
            elif spatial_array.shape[1] == 4:
                # Assume [x_local, y_local, x_global, y_global]
                spatial_coords = {
                    'x_local': spatial_array[:, 0],
                    'y_local': spatial_array[:, 1],
                    'x_global': spatial_array[:, 2],
                    'y_global': spatial_array[:, 3]
                }
            else:
                print(f"  ⚠ Spatial array has unexpected shape: {spatial_array.shape}")
                spatial_coords = None
        else:
            print(f"  ⚠ No spatial coordinates found in obsm['{spatial_key}']")
            spatial_coords = None
        
        # Create object
        return spatioloji(
            expression=expression,
            cell_ids=cell_ids,
            gene_names=gene_names,
            cell_metadata=cell_metadata,
            gene_metadata=gene_metadata,
            spatial_coords=spatial_coords,
            polygons=polygons,
            fov_positions=fov_positions,
            images_folder=images_folder,
            config=config
        )
    
    def to_anndata(self, 
                   include_spatial: bool = True,
                   spatial_key: str = 'spatial') -> 'anndata.AnnData':
        """
        Convert to AnnData object.
        
        Creates AnnData with:
        - X: expression matrix
        - obs: cell metadata
        - var: gene metadata
        - obsm[spatial_key]: spatial coordinates (if include_spatial=True)
        
        Parameters
        ----------
        include_spatial : bool
            Include spatial coordinates in obsm
        spatial_key : str
            Key for spatial coordinates in obsm
        
        Returns
        -------
        anndata.AnnData
            AnnData object
        """
        import anndata
        
        print(f"\nConverting spatioloji to AnnData...")
        
        # Create AnnData
        adata = anndata.AnnData(
            X=self._expression.get_sparse() if self._expression.is_sparse else self._expression.get_dense(),
            obs=self._cell_meta.copy(),
            var=self._gene_meta.copy()
        )
        
        # Add spatial coordinates
        if include_spatial:
            spatial_array = np.column_stack([
                self._spatial.x_local,
                self._spatial.y_local,
                self._spatial.x_global,
                self._spatial.y_global
            ])
            adata.obsm[spatial_key] = spatial_array
        
        # Add FOV info to uns
        adata.uns['fov_positions'] = self._fov_positions.to_dict()
        adata.uns['n_fovs'] = len(self._fov_index)
        
        print(f"✓ Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        return adata


# ========== Helper Functions for File Loading ==========

def load_expression_file(filepath: str,
                        gene_names: Optional[List[str]] = None) -> Tuple[Union[np.ndarray, sparse.spmatrix], pd.Index, pd.Index]:
    """
    Load expression data from various file formats.
    
    Supports: CSV, NPY, NPZ
    
    Parameters
    ----------
    filepath : str
        Path to expression file
    gene_names : list, optional
        Gene names (for files without headers)
    
    Returns
    -------
    Tuple[array, Index, Index]
        (expression_matrix, cell_ids, gene_names)
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        # Load CSV
        df = pd.read_csv(filepath, index_col=0)
        return df.values, pd.Index(df.index), pd.Index(df.columns)
    
    elif suffix == '.npy':
        # Load numpy array
        arr = np.load(filepath)
        cell_ids = pd.Index([f"cell_{i}" for i in range(arr.shape[0])])
        
        if gene_names is None:
            gene_names = pd.Index([f"gene_{i}" for i in range(arr.shape[1])])
        else:
            gene_names = pd.Index(gene_names)
        
        return arr, cell_ids, gene_names
    
    elif suffix == '.npz':
        # Load sparse matrix
        arr = sparse.load_npz(filepath)
        cell_ids = pd.Index([f"cell_{i}" for i in range(arr.shape[0])])
        
        if gene_names is None:
            gene_names = pd.Index([f"gene_{i}" for i in range(arr.shape[1])])
        else:
            gene_names = pd.Index(gene_names)
        
        return arr, cell_ids, gene_names
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def extract_spatial_coords_from_polygons(polygons: pd.DataFrame,
                                         config: SpatiolojiConfig) -> pd.DataFrame:
    """
    Extract spatial coordinates from polygon data.
    
    Calculates cell centroids from polygon vertices.
    
    Parameters
    ----------
    polygons : pd.DataFrame
        Polygon data with coordinate columns
    config : SpatiolojiConfig
        Configuration with column names
    
    Returns
    -------
    pd.DataFrame
        Spatial coordinates per cell
    """
    cell_id_col = config.cell_id_col
    x_local_col, y_local_col = config.get_coordinate_columns('local')
    x_global_col, y_global_col = config.get_coordinate_columns('global')
    
    # Calculate centroids
    coords = polygons.groupby(cell_id_col).agg({
        x_local_col: 'mean',
        y_local_col: 'mean',
        x_global_col: 'mean',
        y_global_col: 'mean'
    })
    
    coords.columns = ['x_local', 'y_local', 'x_global', 'y_global']
    
    return coords
