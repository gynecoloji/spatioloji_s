# src/spatioloji_s/spatial/point/neighborhoods.py

"""
neighborhoods.py - Neighborhood queries and composition analysis

Functions for analyzing cellular neighborhoods using point-based graphs.
All functions use global coordinates.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
    from .graph import SpatialGraph

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import warnings


def get_neighbors(sp: 'spatioloji',
                 graph: 'SpatialGraph',
                 cell_id: str,
                 return_distances: bool = False) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    """
    Get neighbors for a specific cell.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    cell_id : str
        Cell identifier
    return_distances : bool, default=False
        If True, also return distances to neighbors
    
    Returns
    -------
    list or tuple
        Neighbor cell IDs, optionally with distances
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> neighbors = sj.spatial.point.get_neighbors(sp, graph, 'cell_1')
    >>> print(f"Cell 1 has {len(neighbors)} neighbors")
    >>> 
    >>> # Get neighbors with distances
    >>> neighbors, dists = sj.spatial.point.get_neighbors(
    ...     sp, graph, 'cell_1', return_distances=True
    ... )
    """
    neighbor_ids = graph.get_neighbor_ids(cell_id)
    
    if return_distances:
        distances = graph.get_neighbor_distances(cell_id)
        if distances is None:
            warnings.warn("Graph has no distance information, returning None for distances")
        return neighbor_ids, distances
    
    return neighbor_ids


def get_cells_in_radius(sp: 'spatioloji',
                       center_cell: str,
                       radius: float,
                       include_center: bool = False) -> List[str]:
    """
    Get all cells within radius of a center cell.
    
    This is a one-time query that doesn't require pre-built graph.
    Uses global coordinates. For FOV-specific analysis, subset first.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    center_cell : str
        Center cell ID
    radius : float
        Search radius
    include_center : bool, default=False
        Whether to include center cell in results
    
    Returns
    -------
    list
        Cell IDs within radius
    
    Examples
    --------
    >>> # Find cells within 50 units of a specific cell
    >>> nearby = sj.spatial.point.get_cells_in_radius(sp, 'cell_1', radius=50)
    >>> 
    >>> # For FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> nearby = sj.spatial.point.get_cells_in_radius(sp_fov, 'cell_1', radius=50)
    >>> 
    >>> # Get neighborhood for multiple cells
    >>> for cell_id in ['cell_1', 'cell_2', 'cell_3']:
    ...     neighbors = sj.spatial.point.get_cells_in_radius(sp, cell_id, radius=100)
    ...     print(f"{cell_id}: {len(neighbors)} neighbors")
    """
    # Get coordinates (always global)
    coords = sp.get_spatial_coords(coord_type='global')
    
    # Get center cell coordinate
    if center_cell not in sp.cell_index:
        raise ValueError(f"Cell '{center_cell}' not found")
    
    center_idx = sp._cell_id_to_idx[center_cell]
    center_coord = coords[center_idx]
    
    # Build KD-tree
    tree = KDTree(coords)
    
    # Query cells within radius
    indices = tree.query_ball_point(center_coord, radius)
    
    # Get cell IDs
    cell_ids = [sp.cell_index[i] for i in indices]
    
    # Remove center if requested
    if not include_center and center_cell in cell_ids:
        cell_ids.remove(center_cell)
    
    return cell_ids


def get_neighborhood_composition(sp: 'spatioloji',
                                 graph: 'SpatialGraph',
                                 groupby: str,
                                 normalize: bool = True,
                                 add_to_obs: bool = False) -> pd.DataFrame:
    """
    Get neighborhood composition for each cell.
    
    Counts categories (e.g., cell types) in each cell's neighborhood
    based on the spatial graph.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph defining neighborhoods
    groupby : str
        Column in cell_meta to group by (e.g., 'cell_type', 'cluster')
    normalize : bool, default=True
        If True, return proportions instead of counts
    add_to_obs : bool, default=False
        If True, add composition to sp.cell_meta with prefix 'nhood_'
    
    Returns
    -------
    pd.DataFrame
        Composition matrix (cells × categories)
        Index aligned with sp.cell_index
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Get neighborhood composition by cell type
    >>> comp = sj.spatial.point.get_neighborhood_composition(
    ...     sp, graph, groupby='cell_type'
    ... )
    >>> 
    >>> # Cells with high T-cell neighborhood
    >>> high_tcell = comp['T_cell'] > 0.5
    >>> sp.cell_meta['high_tcell_nhood'] = high_tcell
    >>> 
    >>> # Add to cell_meta automatically
    >>> comp = sj.spatial.point.get_neighborhood_composition(
    ...     sp, graph, groupby='cell_type', add_to_obs=True
    ... )
    >>> # Now available as sp.cell_meta['nhood_T_cell'], etc.
    """
    print(f"\n{'='*70}")
    print(f"Computing Neighborhood Composition")
    print(f"{'='*70}")
    
    if groupby not in sp.cell_meta.columns:
        raise ValueError(f"Column '{groupby}' not found in cell_meta")
    
    print(f"Parameters:")
    print(f"  groupby = '{groupby}'")
    print(f"  normalize = {normalize}")
    
    # Get categories
    categories = sp.cell_meta[groupby].astype(str)
    unique_cats = sorted(categories.unique())
    
    print(f"  Found {len(unique_cats)} categories")
    
    # Initialize result matrix
    n_cells = len(sp.cell_index)
    composition = pd.DataFrame(
        0.0,
        index=sp.cell_index,
        columns=unique_cats,
        dtype=np.float32
    )
    
    print(f"\nCounting neighbors for {n_cells:,} cells...")
    
    # Count neighbors for each cell
    for i, cell_id in enumerate(sp.cell_index):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} / {n_cells:,} cells...")
        
        neighbor_indices = graph.get_neighbors(cell_id)
        
        if len(neighbor_indices) > 0:
            neighbor_ids = sp.cell_index[neighbor_indices]
            neighbor_cats = categories.loc[neighbor_ids]
            
            # Count each category
            counts = neighbor_cats.value_counts()
            composition.loc[cell_id, counts.index] = counts.values
    
    # Normalize if requested
    if normalize:
        row_sums = composition.sum(axis=1)
        composition = composition.div(row_sums, axis=0).fillna(0)
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Composition matrix: {n_cells:,} cells × {len(unique_cats)} categories")
    if normalize:
        print(f"  Values are proportions (sum to 1)")
    else:
        print(f"  Values are counts")
    print(f"{'='*70}\n")
    
    # Add to cell_meta if requested
    if add_to_obs:
        for cat in unique_cats:
            col_name = f"nhood_{cat}"
            sp.cell_meta[col_name] = composition[cat]
        print(f"Added to cell_meta with prefix 'nhood_'")
    
    return composition


def get_neighborhood_expression(sp: 'spatioloji',
                                graph: 'SpatialGraph',
                                genes: Optional[Union[str, List[str]]] = None,
                                layer: Optional[str] = None,
                                agg_func: str = 'mean') -> pd.DataFrame:
    """
    Get aggregated gene expression in each cell's neighborhood.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph defining neighborhoods
    genes : str or list, optional
        Gene(s) to analyze. If None, uses all genes.
    layer : str, optional
        Expression layer to use. If None, uses main expression.
    agg_func : str, default='mean'
        Aggregation function: 'mean', 'median', 'sum', 'max'
    
    Returns
    -------
    pd.DataFrame
        Neighborhood expression (cells × genes)
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Mean expression of specific genes in neighborhoods
    >>> nhood_expr = sj.spatial.point.get_neighborhood_expression(
    ...     sp, graph, genes=['CD4', 'CD8', 'CD19']
    ... )
    >>> 
    >>> # Compare cell's own expression vs neighborhood
    >>> cell_expr = sp.get_expression(gene_names=['CD4'], as_dataframe=True)
    >>> nhood_expr = sj.spatial.point.get_neighborhood_expression(
    ...     sp, graph, genes=['CD4']
    ... )
    >>> sp.cell_meta['CD4_self'] = cell_expr['CD4']
    >>> sp.cell_meta['CD4_nhood'] = nhood_expr['CD4']
    """
    print(f"\n{'='*70}")
    print(f"Computing Neighborhood Expression")
    print(f"{'='*70}")
    
    # Get expression data
    if layer is not None:
        if layer not in sp.layers:
            raise ValueError(f"Layer '{layer}' not found")
        expr_data = sp.get_layer(layer)
    else:
        expr_data = sp.expression.get_dense()
    
    # Get genes to analyze
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        gene_indices = sp._get_gene_indices(genes)
        gene_names = sp.gene_index[gene_indices]
        expr_data = expr_data[:, gene_indices]
    else:
        gene_names = sp.gene_index
    
    print(f"Parameters:")
    print(f"  n_genes = {len(gene_names)}")
    print(f"  agg_func = {agg_func}")
    print(f"  layer = {layer if layer else 'main expression'}")
    
    # Initialize result
    nhood_expr = pd.DataFrame(
        0.0,
        index=sp.cell_index,
        columns=gene_names,
        dtype=np.float32
    )
    
    print(f"\nAggregating expression for {len(sp.cell_index):,} cells...")
    
    # Aggregate for each cell's neighborhood
    for i, cell_id in enumerate(sp.cell_index):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i:,} / {len(sp.cell_index):,} cells...")
        
        neighbor_indices = graph.get_neighbors(cell_id)
        
        if len(neighbor_indices) > 0:
            neighbor_expr = expr_data[neighbor_indices, :]
            
            # Aggregate
            if agg_func == 'mean':
                agg_values = neighbor_expr.mean(axis=0)
            elif agg_func == 'median':
                agg_values = np.median(neighbor_expr, axis=0)
            elif agg_func == 'sum':
                agg_values = neighbor_expr.sum(axis=0)
            elif agg_func == 'max':
                agg_values = neighbor_expr.max(axis=0)
            else:
                raise ValueError(f"Unknown agg_func: {agg_func}")
            
            nhood_expr.loc[cell_id] = agg_values
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Neighborhood expression: {len(sp.cell_index):,} cells × {len(gene_names)} genes")
    print(f"{'='*70}\n")
    
    return nhood_expr


def compute_neighborhood_diversity(sp: 'spatioloji',
                                   graph: 'SpatialGraph',
                                   groupby: str,
                                   metric: str = 'shannon') -> pd.Series:
    """
    Compute diversity of cell types in each neighborhood.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    groupby : str
        Column to compute diversity on (e.g., 'cell_type')
    metric : str, default='shannon'
        Diversity metric: 'shannon' (entropy), 'simpson', or 'richness'
    
    Returns
    -------
    pd.Series
        Diversity score for each cell
    
    Notes
    -----
    - Shannon entropy: -sum(p * log(p)) where p is proportion
    - Simpson diversity: 1 - sum(p^2)
    - Richness: Number of unique types
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=20)
    >>> 
    >>> # Compute Shannon diversity
    >>> diversity = sj.spatial.point.compute_neighborhood_diversity(
    ...     sp, graph, groupby='cell_type', metric='shannon'
    ... )
    >>> sp.cell_meta['nhood_diversity'] = diversity
    >>> 
    >>> # High diversity regions (mixed cell types)
    >>> high_diversity = sp.cell_meta[sp.cell_meta['nhood_diversity'] > 1.5].index
    """
    print(f"\n{'='*70}")
    print(f"Computing Neighborhood Diversity")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  groupby = '{groupby}'")
    print(f"  metric = {metric}")
    
    # Get composition
    comp = get_neighborhood_composition(
        sp, graph, groupby, normalize=True
    )
    
    print(f"\nCalculating {metric} diversity...")
    
    # Calculate diversity
    if metric == 'shannon':
        # Shannon entropy: -sum(p * log(p))
        diversity = -np.sum(comp * np.log(comp + 1e-10), axis=1)
    
    elif metric == 'simpson':
        # Simpson diversity: 1 - sum(p^2)
        diversity = 1 - np.sum(comp ** 2, axis=1)
    
    elif metric == 'richness':
        # Number of unique types
        diversity = (comp > 0).sum(axis=1)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    diversity = pd.Series(diversity, index=sp.cell_index, name=f'{metric}_diversity')
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Min diversity: {diversity.min():.3f}")
    print(f"  Max diversity: {diversity.max():.3f}")
    print(f"  Mean diversity: {diversity.mean():.3f}")
    print(f"{'='*70}\n")
    
    return diversity


def compute_spatial_lag(sp: 'spatioloji',
                       graph: 'SpatialGraph',
                       values: Union[str, pd.Series, np.ndarray],
                       weighted: bool = False) -> pd.Series:
    """
    Compute spatial lag (average of neighbors' values).
    
    The spatial lag is the average value of a variable in a cell's
    neighborhood. Used in spatial autocorrelation analysis.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    graph : SpatialGraph
        Spatial graph
    values : str, pd.Series, or np.ndarray
        Values to compute lag for.
        - str: column name in cell_meta
        - pd.Series: values with index matching sp.cell_index
        - np.ndarray: values aligned with sp.cell_index
    weighted : bool, default=False
        If True, weight by inverse distance (requires graph.distances)
    
    Returns
    -------
    pd.Series
        Spatial lag for each cell
    
    Examples
    --------
    >>> graph = sj.spatial.point.build_knn_graph(sp, k=10)
    >>> 
    >>> # Spatial lag of gene expression
    >>> expr = sp.get_expression(gene_names='CD4', as_dataframe=True)['CD4']
    >>> lag = sj.spatial.point.compute_spatial_lag(sp, graph, expr)
    >>> 
    >>> # Compare cell value vs neighborhood average
    >>> sp.cell_meta['CD4'] = expr
    >>> sp.cell_meta['CD4_lag'] = lag
    >>> 
    >>> # Spatial lag from cell_meta column
    >>> lag = sj.spatial.point.compute_spatial_lag(sp, graph, 'total_counts')
    """
    print(f"\n{'='*70}")
    print(f"Computing Spatial Lag")
    print(f"{'='*70}")
    
    # Get values
    if isinstance(values, str):
        if values not in sp.cell_meta.columns:
            raise ValueError(f"Column '{values}' not found in cell_meta")
        values_array = sp.cell_meta[values].values
        values_name = values
    elif isinstance(values, pd.Series):
        # Align to cell_index
        values_array = values.reindex(sp.cell_index).values
        values_name = values.name or 'values'
    elif isinstance(values, np.ndarray):
        values_array = values
        values_name = 'values'
    else:
        raise TypeError("values must be str, pd.Series, or np.ndarray")
    
    print(f"Parameters:")
    print(f"  values = {values_name}")
    print(f"  weighted = {weighted}")
    
    if weighted and graph.distances is None:
        raise ValueError("Graph must have distances for weighted spatial lag")
    
    # Compute spatial lag
    spatial_lag = np.zeros(len(sp.cell_index))
    
    print(f"\nComputing lag for {len(sp.cell_index):,} cells...")
    
    for i, cell_id in enumerate(sp.cell_index):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} / {len(sp.cell_index):,} cells...")
        
        neighbor_indices = graph.get_neighbors(cell_id)
        
        if len(neighbor_indices) > 0:
            neighbor_values = values_array[neighbor_indices]
            
            if weighted:
                # Weight by inverse distance
                distances = graph.get_neighbor_distances(cell_id)
                weights = 1.0 / (distances + 1e-10)
                weights = weights / weights.sum()
                spatial_lag[i] = np.sum(neighbor_values * weights)
            else:
                # Simple mean
                spatial_lag[i] = neighbor_values.mean()
    
    result = pd.Series(spatial_lag, index=sp.cell_index, name=f'{values_name}_lag')
    
    print(f"\n{'='*70}")
    print(f"Result:")
    print(f"  Min lag: {result.min():.3f}")
    print(f"  Max lag: {result.max():.3f}")
    print(f"  Mean lag: {result.mean():.3f}")
    print(f"{'='*70}\n")
    
    return result