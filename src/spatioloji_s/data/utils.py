"""
utils.py - Utility functions for spatioloji objects

Provides helper functions for data conversion, export, and common
operations that complement the core spatioloji functionality.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata
    from spatioloji_s.data.core import spatioloji
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json


# ========== Data Export Utilities ==========

def export_to_csv_bundle(sp, output_dir: str, include_qc: bool = True) -> None:
    """
    Export all spatioloji components to CSV files.
    
    Creates directory structure with all data components in CSV format.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object to export
    output_dir : str
        Directory to save files
    include_qc : bool
        Whether to include QC flags from metadata
    
    Examples
    --------
    >>> from spatioloji import spatioloji
    >>> from spatioloji.data.utils import export_to_csv_bundle
    >>> sp = spatioloji.from_pickle('data.pkl')
    >>> export_to_csv_bundle(sp, 'exported_data/')
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting spatioloji to CSV bundle: {output_dir}")
    
    # 1. Expression matrix
    print("  → Exporting expression matrix...")
    expr_df = sp.expression.to_dataframe()
    expr_df.to_csv(output_path / 'expression_matrix.csv')
    
    # 2. Cell metadata
    print("  → Exporting cell metadata...")
    sp.cell_meta.to_csv(output_path / 'cell_metadata.csv')
    
    # 3. Gene metadata
    print("  → Exporting gene metadata...")
    sp.gene_meta.to_csv(output_path / 'gene_metadata.csv')
    
    # 4. Spatial coordinates
    print("  → Exporting spatial coordinates...")
    spatial_df = pd.DataFrame({
        'x_local': sp.spatial.x_local,
        'y_local': sp.spatial.y_local,
        'x_global': sp.spatial.x_global,
        'y_global': sp.spatial.y_global
    }, index=sp.cell_index)
    spatial_df.to_csv(output_path / 'spatial_coords.csv')
    
    # 5. Polygons
    if sp.polygons is not None:
        print("  → Exporting polygons...")
        sp.polygons.to_csv(output_path / 'polygons.csv', index=False)
    
    # 6. FOV positions
    print("  → Exporting FOV positions...")
    sp.fov_positions.to_csv(output_path / 'fov_positions.csv')
    
    # 7. Image metadata
    if sp.images.n_total > 0:
        print("  → Exporting image metadata...")
        image_meta_list = []
        for fov_id in sp.images.fov_ids:
            meta = sp.images.get_metadata(fov_id)
            if meta:
                image_meta_list.append({
                    'fov_id': fov_id,
                    'width': meta.width,
                    'height': meta.height,
                    'n_channels': meta.n_channels,
                    'path': str(meta.path) if meta.path else None,
                    'is_loaded': meta.is_loaded
                })
        
        pd.DataFrame(image_meta_list).to_csv(
            output_path / 'image_metadata.csv', index=False
        )
    
    # 8. Summary info
    print("  → Exporting summary...")
    summary = sp.summary()
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Export complete: {output_dir}")
    print(f"  Files created:")
    print(f"    - expression_matrix.csv")
    print(f"    - cell_metadata.csv")
    print(f"    - gene_metadata.csv")
    print(f"    - spatial_coords.csv")
    if sp.polygons is not None:
        print(f"    - polygons.csv")
    print(f"    - fov_positions.csv")
    if sp.images.n_total > 0:
        print(f"    - image_metadata.csv")
    print(f"    - summary.json")


def export_to_seurat_files(sp, output_dir: str) -> None:
    """
    Export spatioloji to Seurat-compatible files.
    
    Creates files that can be loaded into Seurat (R):
    - counts.csv: Expression matrix
    - metadata.csv: Cell metadata with spatial coordinates
    - spatial_coords.csv: Separate spatial coordinates file
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    output_dir : str
        Directory to save files
    
    Examples
    --------
    >>> export_to_seurat_files(sp, 'for_seurat/')
    # In R:
    # counts <- read.csv('for_seurat/counts.csv', row.names=1)
    # metadata <- read.csv('for_seurat/metadata.csv', row.names=1)
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to Seurat-compatible format: {output_dir}")
    
    # Expression matrix (genes as rows for Seurat)
    print("  → Creating counts.csv (genes × cells)...")
    expr_df = sp.expression.to_dataframe().T  # Transpose for Seurat
    expr_df.to_csv(output_path / 'counts.csv')
    
    # Metadata with coordinates
    print("  → Creating metadata.csv...")
    metadata = sp.cell_meta.copy()
    metadata['x_local'] = sp.spatial.x_local
    metadata['y_local'] = sp.spatial.y_local
    metadata['x_global'] = sp.spatial.x_global
    metadata['y_global'] = sp.spatial.y_global
    metadata.to_csv(output_path / 'metadata.csv')
    
    # Separate spatial coordinates
    print("  → Creating spatial_coords.csv...")
    spatial_df = pd.DataFrame({
        'cell': sp.cell_index,
        'x': sp.spatial.x_global,
        'y': sp.spatial.y_global
    })
    spatial_df.to_csv(output_path / 'spatial_coords.csv', index=False)
    
    print(f"\n✓ Seurat export complete!")


def export_to_squidpy(sp) -> 'anndata.AnnData':
    """
    Convert spatioloji to Squidpy/Scanpy-compatible AnnData.
    
    Adds spatial coordinates to adata.obsm['spatial'] for compatibility
    with Squidpy spatial analysis functions.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    
    Returns
    -------
    anndata.AnnData
        AnnData with spatial information
    
    Examples
    --------
    >>> adata = export_to_squidpy(sp)
    >>> import squidpy as sq
    >>> sq.pl.spatial_scatter(adata, color='Gene1')
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    print("\nConverting to Squidpy-compatible AnnData...")
    
    # Use spatioloji's built-in conversion
    adata = sp.to_anndata(include_spatial=True, spatial_key='spatial')
    
    # Add Squidpy-specific metadata
    adata.uns['spatial'] = {
        'spatioloji': {
            'images': {},
            'scalefactors': {
                'tissue_hires_scalef': 1.0,
                'spot_diameter_fullres': 1.0
            }
        }
    }
    
    print(f"✓ Created AnnData with spatial coordinates")
    print(f"  Use: adata.obsm['spatial'] for coordinates")
    
    return adata


# ========== Data Alignment Utilities ==========

def align_dataframe_to_cells(sp, df: pd.DataFrame, 
                            id_column: Optional[str] = None) -> pd.DataFrame:
    """
    Align any DataFrame to spatioloji's master cell index.
    
    Useful for adding external data to spatioloji.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    df : pd.DataFrame
        DataFrame to align
    id_column : str, optional
        Column containing cell IDs. If None, uses index.
    
    Returns
    -------
    pd.DataFrame
        Aligned DataFrame
    
    Examples
    --------
    >>> external_data = pd.DataFrame({
    ...     'cell': ['cell_1', 'cell_2'],
    ...     'cluster': ['A', 'B']
    ... })
    >>> aligned = align_dataframe_to_cells(sp, external_data, id_column='cell')
    >>> sp.cell_meta['cluster'] = aligned['cluster']
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("First argument must be a spatioloji object")
    
    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in DataFrame")
        df = df.set_index(id_column)
    
    # Reindex to master cell index
    aligned = df.reindex(sp.cell_index)
    
    # Report alignment
    n_matched = aligned.notna().any(axis=1).sum()
    n_missing = len(sp.cell_index) - n_matched
    n_extra = len(df) - n_matched
    
    print(f"Alignment results:")
    print(f"  Matched:  {n_matched:,} cells")
    if n_missing > 0:
        print(f"  Missing:  {n_missing:,} cells (filled with NaN)")
    if n_extra > 0:
        print(f"  Extra:    {n_extra:,} rows (dropped)")
    
    return aligned


def align_dataframe_to_genes(sp, df: pd.DataFrame,
                            id_column: Optional[str] = None) -> pd.DataFrame:
    """
    Align any DataFrame to spatioloji's master gene index.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    df : pd.DataFrame
        DataFrame to align
    id_column : str, optional
        Column containing gene names. If None, uses index.
    
    Returns
    -------
    pd.DataFrame
        Aligned DataFrame
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("First argument must be a spatioloji object")
    
    if id_column is not None:
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in DataFrame")
        df = df.set_index(id_column)
    
    aligned = df.reindex(sp.gene_index)
    
    n_matched = aligned.notna().any(axis=1).sum()
    n_missing = len(sp.gene_index) - n_matched
    
    print(f"Alignment results:")
    print(f"  Matched:  {n_matched:,} genes")
    if n_missing > 0:
        print(f"  Missing:  {n_missing:,} genes (filled with NaN)")
    
    return aligned


# ========== Coordinate Conversion Utilities ==========

def convert_coordinates(coords: np.ndarray, 
                       from_type: str,
                       to_type: str,
                       fov_positions: pd.DataFrame,
                       fov_ids: np.ndarray) -> np.ndarray:
    """
    Convert between local and global coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinates to convert (N × 2)
    from_type : str
        'local' or 'global'
    to_type : str
        'local' or 'global'
    fov_positions : pd.DataFrame
        FOV positions with x_global_px, y_global_px
    fov_ids : np.ndarray
        FOV IDs for each coordinate (length N)
    
    Returns
    -------
    np.ndarray
        Converted coordinates (N × 2)
    
    Examples
    --------
    >>> local_coords = sp.get_spatial_coords(coord_type='local')
    >>> fov_ids = sp.cell_meta[sp.config.fov_id_col].values
    >>> global_coords = convert_coordinates(
    ...     local_coords, 'local', 'global',
    ...     sp.fov_positions, fov_ids
    ... )
    """
    if from_type == to_type:
        return coords.copy()
    
    if from_type == 'local' and to_type == 'global':
        # Local → Global: add FOV positions
        result = coords.copy()
        for fov_id in np.unique(fov_ids):
            mask = fov_ids == fov_id
            if fov_id in fov_positions.index:
                fov_x = fov_positions.loc[fov_id, 'x_global_px']
                fov_y = fov_positions.loc[fov_id, 'y_global_px']
                result[mask, 0] += fov_x
                result[mask, 1] += fov_y
        return result
    
    elif from_type == 'global' and to_type == 'local':
        # Global → Local: subtract FOV positions
        result = coords.copy()
        for fov_id in np.unique(fov_ids):
            mask = fov_ids == fov_id
            if fov_id in fov_positions.index:
                fov_x = fov_positions.loc[fov_id, 'x_global_px']
                fov_y = fov_positions.loc[fov_id, 'y_global_px']
                result[mask, 0] -= fov_x
                result[mask, 1] -= fov_y
        return result
    
    else:
        raise ValueError(f"Invalid conversion: {from_type} → {to_type}")


def calculate_cell_centroids(sp, use_polygons: bool = True) -> pd.DataFrame:
    """
    Calculate cell centroids.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    use_polygons : bool
        If True, calculate from polygon vertices.
        If False, use stored spatial coordinates.
    
    Returns
    -------
    pd.DataFrame
        Centroids with columns: x_local, y_local, x_global, y_global
    
    Examples
    --------
    >>> centroids = calculate_cell_centroids(sp, use_polygons=True)
    >>> sp.cell_meta['centroid_x'] = centroids['x_global']
    >>> sp.cell_meta['centroid_y'] = centroids['y_global']
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    if use_polygons and sp.polygons is not None:
        # Calculate from polygons
        cell_id_col = sp.config.cell_id_col
        
        centroids = sp.polygons.groupby(cell_id_col).agg({
            sp.config.x_local_col: 'mean',
            sp.config.y_local_col: 'mean',
            sp.config.x_global_col: 'mean',
            sp.config.y_global_col: 'mean'
        })
        
        centroids.columns = ['x_local', 'y_local', 'x_global', 'y_global']
        
        # Align to master cell index
        centroids = centroids.reindex(sp.cell_index)
    else:
        # Use stored coordinates
        centroids = pd.DataFrame({
            'x_local': sp.spatial.x_local,
            'y_local': sp.spatial.y_local,
            'x_global': sp.spatial.x_global,
            'y_global': sp.spatial.y_global
        }, index=sp.cell_index)
    
    return centroids


# ========== Spatial Query Utilities ==========

def get_cells_in_polygon(sp, 
                         polygon_coords: np.ndarray,
                         coord_type: str = 'global') -> List[str]:
    """
    Get cells within a polygon region.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    polygon_coords : np.ndarray
        Polygon vertices (N × 2)
    coord_type : str
        'local' or 'global'
    
    Returns
    -------
    list
        Cell IDs within polygon
    
    Examples
    --------
    >>> # Define a square region
    >>> polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    >>> cells_in_region = get_cells_in_polygon(sp, polygon, coord_type='global')
    >>> subset = sp.subset_by_cells(cells_in_region)
    """
    from matplotlib.path import Path as MPLPath
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    # Get cell coordinates
    cell_coords = sp.get_spatial_coords(coord_type=coord_type)
    
    # Create polygon path
    polygon_path = MPLPath(polygon_coords)
    
    # Test which points are inside
    mask = polygon_path.contains_points(cell_coords)
    
    # Get cell IDs
    cell_ids = sp.cell_index[mask].tolist()
    
    print(f"Found {len(cell_ids)} cells in polygon")
    
    return cell_ids


def get_cells_in_circle(sp,
                        center: Tuple[float, float],
                        radius: float,
                        coord_type: str = 'global') -> List[str]:
    """
    Get cells within a circular region.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    center : tuple
        (x, y) coordinates of circle center
    radius : float
        Circle radius
    coord_type : str
        'local' or 'global'
    
    Returns
    -------
    list
        Cell IDs within circle
    
    Examples
    --------
    >>> cells_near_center = get_cells_in_circle(
    ...     sp, center=(500, 500), radius=100, coord_type='global'
    ... )
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    # Get cell coordinates
    cell_coords = sp.get_spatial_coords(coord_type=coord_type)
    
    # Calculate distances from center
    dx = cell_coords[:, 0] - center[0]
    dy = cell_coords[:, 1] - center[1]
    distances = np.sqrt(dx**2 + dy**2)
    
    # Get cells within radius
    mask = distances <= radius
    cell_ids = sp.cell_index[mask].tolist()
    
    print(f"Found {len(cell_ids)} cells within radius {radius}")
    
    return cell_ids


def get_fov_bounds(sp, fov_id: str) -> Dict[str, float]:
    """
    Get bounding box for a FOV.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    fov_id : str
        FOV identifier
    
    Returns
    -------
    dict
        Bounding box: {'xmin', 'xmax', 'ymin', 'ymax'}
    
    Examples
    --------
    >>> bounds = get_fov_bounds(sp, '001')
    >>> print(f"FOV 001: x={bounds['xmin']}-{bounds['xmax']}")
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    # Get cells in FOV
    cells_in_fov = sp.get_cells_in_fov(fov_id)
    
    if len(cells_in_fov) == 0:
        return {'xmin': 0, 'xmax': 0, 'ymin': 0, 'ymax': 0}
    
    # Get coordinates (global)
    coords = sp.get_spatial_coords(cell_ids=cells_in_fov, coord_type='global')
    
    bounds = {
        'xmin': coords[:, 0].min(),
        'xmax': coords[:, 0].max(),
        'ymin': coords[:, 1].min(),
        'ymax': coords[:, 1].max()
    }
    
    return bounds


# ========== Gene List Utilities ==========

def parse_gene_list(gene_input: Union[str, List[str], Path]) -> List[str]:
    """
    Parse gene names from various formats.
    
    Parameters
    ----------
    gene_input : str, list, or Path
        Can be:
        - List of gene names
        - Comma-separated string: "Gene1,Gene2,Gene3"
        - Path to text file (one gene per line)
    
    Returns
    -------
    list
        List of gene names
    
    Examples
    --------
    >>> genes = parse_gene_list("Gene1,Gene2,Gene3")
    >>> genes = parse_gene_list(['Gene1', 'Gene2'])
    >>> genes = parse_gene_list('gene_list.txt')
    """
    if isinstance(gene_input, list):
        return gene_input
    
    if isinstance(gene_input, (str, Path)):
        gene_path = Path(gene_input)
        
        # Check if it's a file
        if gene_path.exists() and gene_path.is_file():
            with open(gene_path, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
            return genes
        
        # Treat as comma-separated string
        genes = [g.strip() for g in str(gene_input).split(',')]
        return genes
    
    raise TypeError(f"Unsupported gene_input type: {type(gene_input)}")


def filter_genes_by_list(sp, gene_names: Union[str, List[str], Path]) -> 'spatioloji':
    """
    Filter spatioloji to keep only specified genes.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    gene_names : str, list, or Path
        Gene names (see parse_gene_list for formats)
    
    Returns
    -------
    spatioloji
        Filtered spatioloji object
    
    Examples
    --------
    >>> marker_genes = ['CD4', 'CD8', 'CD19', 'CD3']
    >>> sp_markers = filter_genes_by_list(sp, marker_genes)
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    # Parse gene list
    genes = parse_gene_list(gene_names)
    
    # Filter to genes present in spatioloji
    genes_present = [g for g in genes if g in sp.gene_index]
    genes_missing = [g for g in genes if g not in sp.gene_index]
    
    if genes_missing:
        print(f"  ⚠ {len(genes_missing)} genes not found: {genes_missing[:5]}...")
    
    if not genes_present:
        raise ValueError("No genes found in spatioloji")
    
    # Subset
    return sp.subset_by_genes(genes_present)


# ========== Validation Utilities ==========

def validate_input_data(polygons: pd.DataFrame,
                       cell_meta: pd.DataFrame,
                       expression: np.ndarray,
                       config) -> Dict[str, bool]:
    """
    Validate input data before creating spatioloji.
    
    Parameters
    ----------
    polygons : pd.DataFrame
        Polygon data
    cell_meta : pd.DataFrame
        Cell metadata
    expression : np.ndarray
        Expression matrix
    config : SpatiolojiConfig
        Configuration
    
    Returns
    -------
    dict
        Validation results
    """
    results = {}
    issues = []
    
    # Check polygons
    if config.cell_id_col not in polygons.columns:
        issues.append(f"Polygons missing '{config.cell_id_col}' column")
        results['polygons_cell_col'] = False
    else:
        results['polygons_cell_col'] = True
    
    # Check cell_meta
    if config.cell_id_col not in cell_meta.columns:
        issues.append(f"Cell metadata missing '{config.cell_id_col}' column")
        results['cell_meta_cell_col'] = False
    else:
        results['cell_meta_cell_col'] = True
    
    # Check dimensions
    n_cells_expr = expression.shape[0]
    n_cells_meta = len(cell_meta)
    
    if n_cells_expr != n_cells_meta:
        issues.append(
            f"Expression rows ({n_cells_expr}) != "
            f"cell metadata rows ({n_cells_meta})"
        )
        results['dimension_match'] = False
    else:
        results['dimension_match'] = True
    
    # Overall
    results['valid'] = len(issues) == 0
    
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("✓ Validation passed")
    
    return results


# ========== Miscellaneous Utilities ==========

def format_cell_ids(cell_ids: List[str], 
                   prefix: str = '',
                   pad_length: Optional[int] = None) -> List[str]:
    """
    Format cell IDs with prefix and padding.
    
    Parameters
    ----------
    cell_ids : list
        Cell identifiers
    prefix : str
        Prefix to add
    pad_length : int, optional
        Zero-pad numbers to this length
    
    Returns
    -------
    list
        Formatted cell IDs
    
    Examples
    --------
    >>> ids = ['1', '2', '100']
    >>> format_cell_ids(ids, prefix='cell_', pad_length=4)
    ['cell_0001', 'cell_0002', 'cell_0100']
    """
    formatted = []
    
    for cid in cell_ids:
        # Try to extract number
        try:
            num = int(''.join(filter(str.isdigit, str(cid))))
            if pad_length:
                formatted_id = f"{prefix}{num:0{pad_length}d}"
            else:
                formatted_id = f"{prefix}{num}"
        except ValueError:
            # Not a number, just add prefix
            formatted_id = f"{prefix}{cid}"
        
        formatted.append(formatted_id)
    
    return formatted


def chunk_data(data: Union[List, np.ndarray, pd.Index],
               chunk_size: int) -> List:
    """
    Split data into chunks.
    
    Parameters
    ----------
    data : list, array, or Index
        Data to chunk
    chunk_size : int
        Size of each chunk
    
    Returns
    -------
    list
        List of chunks
    
    Examples
    --------
    >>> cell_ids = sp.cell_index.tolist()
    >>> chunks = chunk_data(cell_ids, chunk_size=1000)
    >>> for chunk in chunks:
    ...     process_cells(chunk)
    """
    data_list = list(data)
    chunks = []
    
    for i in range(0, len(data_list), chunk_size):
        chunks.append(data_list[i:i + chunk_size])
    
    return chunks


def safe_divide(numerator: np.ndarray, 
               denominator: np.ndarray,
               fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator values
    denominator : np.ndarray
        Denominator values
    fill_value : float
        Value to use when denominator is zero
    
    Returns
    -------
    np.ndarray
        Result of division
    
    Examples
    --------
    >>> ratio = safe_divide(counts, total_counts, fill_value=0.0)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = fill_value
    
    return result


def get_summary_stats(sp, metric: str) -> pd.Series:
    """
    Get summary statistics for a metric.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    metric : str
        Column name in cell_meta
    
    Returns
    -------
    pd.Series
        Summary statistics
    
    Examples
    --------
    >>> stats = get_summary_stats(sp, 'total_counts')
    >>> print(f"Mean: {stats['mean']:.1f}")
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    if metric not in sp.cell_meta.columns:
        raise ValueError(f"Metric '{metric}' not found in cell_meta")
    
    data = sp.cell_meta[metric]
    
    stats = pd.Series({
        'count': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        '25%': data.quantile(0.25),
        '50%': data.quantile(0.50),
        '75%': data.quantile(0.75),
        'max': data.max()
    })
    
    return stats


# ========== Quick Summary Functions ==========

def quick_summary(sp) -> None:
    """
    Print a quick summary of spatioloji object.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    
    Examples
    --------
    >>> from spatioloji.data.utils import quick_summary
    >>> quick_summary(sp)
    """
    from .core import spatioloji
    
    if not isinstance(sp, spatioloji):
        raise TypeError("Input must be a spatioloji object")
    
    summary = sp.summary()
    
    print("\n" + "="*50)
    print("SPATIOLOJI SUMMARY")
    print("="*50)
    print(f"Cells:              {summary['n_cells']:>12,}")
    print(f"Genes:              {summary['n_genes']:>12,}")
    print(f"FOVs:               {summary['n_fovs']:>12}")
    print(f"Images:             {summary['n_images']:>12} "
          f"({summary['n_images_loaded']} loaded)")
    print(f"-" * 50)
    print(f"Avg cells/FOV:      {summary['avg_cells_per_fov']:>12.1f}")
    print(f"Expression format:  {'Sparse' if summary['is_sparse'] else 'Dense':>12}")
    print(f"Sparsity:           {summary['sparsity']:>11.1%}")
    print(f"Has polygons:       {'Yes' if summary['has_polygons'] else 'No':>12}")
    print(f"Memory usage:       {summary['memory_usage_mb']:>11.1f} MB")
    print("="*50 + "\n")