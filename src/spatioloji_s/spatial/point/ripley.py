# src/spatioloji_s/spatial/point/ripley.py

"""
ripley.py - Ripley's K and L functions for point pattern analysis

Provides methods for analyzing spatial clustering at multiple scales:
- Ripley's K function: Analyzes clustering/dispersion at different distances
- Ripley's L function: Variance-stabilized transformation of K
- Cross-type K/L: Interactions between different cell types
- Null model testing: Compare to Complete Spatial Randomness (CSR)

All functions use global coordinates. For FOV-specific analysis, subset first:
    sp_fov = sp.subset_by_fovs(['fov1'])
    k_result = sj.spatial.point.ripleys_k(sp_fov)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji

from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import warnings


def ripleys_k(sp: 'spatioloji',
             radii: Optional[np.ndarray] = None,
             n_radii: int = 50,
             cell_ids: Optional[List[str]] = None,
             edge_correction: str = 'ripley',
             fov_id: Optional[str] = None) -> pd.DataFrame:
    """
    Compute Ripley's K function for point pattern analysis.
    
    Ripley's K(r) measures the expected number of points within distance r
    of a typical point, adjusted for study area. It quantifies clustering
    or dispersion at different spatial scales.
    
    - K(r) > πr²: Clustering at distance r
    - K(r) = πr²: Complete Spatial Randomness (CSR)
    - K(r) < πr²: Dispersion/regularity at distance r
    
    Uses global coordinates. For FOV-specific analysis, use fov_id parameter
    or subset first:
        sp_fov = sp.subset_by_fovs(['fov1'])
        k_result = sj.spatial.point.ripleys_k(sp_fov)
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    radii : np.ndarray, optional
        Distance radii to test. If None, creates n_radii evenly spaced values.
    n_radii : int, default=50
        Number of radii to test (if radii not provided)
    cell_ids : list, optional
        Subset of cells to analyze. If None, uses all cells.
    edge_correction : str, default='ripley'
        Edge correction method:
        - 'none': No correction
        - 'ripley': Ripley's isotropic correction
        - 'border': Border method (exclude edge points)
    fov_id : str, optional
        Analyze specific FOV only
    
    Returns
    -------
    pd.DataFrame
        K function values with columns:
        - r: Distance radius
        - K: Ripley's K value
        - K_csr: Expected K under CSR (πr²)
        - K_diff: K - K_csr (deviation from randomness)
    
    Notes
    -----
    Ripley's K is defined as:
        K(r) = λ⁻¹ * E[number of points within distance r]
    
    where λ is the intensity (points per unit area).
    
    Examples
    --------
    >>> import spatioloji as sj
    >>> 
    >>> # Compute K function for all cells
    >>> k_result = sj.spatial.point.ripleys_k(sp, n_radii=50)
    >>> 
    >>> # Plot K function
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k_result['r'], k_result['K'], label='Observed K(r)')
    >>> plt.plot(k_result['r'], k_result['K_csr'], label='CSR', linestyle='--')
    >>> plt.xlabel('Distance r')
    >>> plt.ylabel('K(r)')
    >>> plt.legend()
    >>> 
    >>> # Analyze specific cell type
    >>> tcells = sp.cell_meta[sp.cell_meta['cell_type'] == 'T_cell'].index
    >>> k_tcells = sj.spatial.point.ripleys_k(sp, cell_ids=tcells)
    >>> 
    >>> # Single FOV analysis (method 1: fov_id parameter)
    >>> k_fov1 = sj.spatial.point.ripleys_k(sp, fov_id='fov1')
    >>> 
    >>> # Single FOV analysis (method 2: subset first)
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> k_fov = sj.spatial.point.ripleys_k(sp_fov)
    """
    print(f"\n{'='*70}")
    print(f"Computing Ripley's K Function (Global Coordinates)")
    print(f"{'='*70}")
    
    # Get coordinates (always global)
    if fov_id is not None:
        cell_ids_fov = sp.get_cells_in_fov(fov_id)
        if cell_ids is not None:
            # Intersection
            cell_ids = list(set(cell_ids) & set(cell_ids_fov))
        else:
            cell_ids = cell_ids_fov
        
        print(f"FOV: {fov_id}")
    
    if cell_ids is not None:
        indices = sp._get_cell_indices(cell_ids)
        coords = sp.get_spatial_coords(coord_type='global')[indices]
        print(f"Cells: {len(cell_ids):,}")
    else:
        coords = sp.get_spatial_coords(coord_type='global')
        print(f"Cells: {len(coords):,}")
    
    n_points = len(coords)
    
    print(f"Edge correction: {edge_correction}")
    
    # Determine study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    
    print(f"Study area: {area:.2f} square units")
    
    # Intensity (points per unit area)
    intensity = n_points / area
    
    print(f"Intensity: {intensity:.6f} points/unit²")
    
    # Determine radii
    if radii is None:
        max_dist = np.sqrt(area) / 4  # Use 1/4 of study area diagonal
        radii = np.linspace(0, max_dist, n_radii + 1)[1:]  # Exclude 0
    
    print(f"Radii: {len(radii)} values from {radii.min():.2f} to {radii.max():.2f}")
    
    # Compute pairwise distances
    print(f"\nComputing pairwise distances...")
    dists = distance_matrix(coords, coords)
    
    # Compute K for each radius
    print(f"Computing K function...")
    K_values = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        if i % 10 == 0 and i > 0:
            print(f"  Radius {i}/{len(radii)}...")
        
        # Count pairs within distance r
        within_r = (dists <= r) & (dists > 0)  # Exclude self
        
        if edge_correction == 'none':
            # No edge correction
            count = within_r.sum()
            K_values[i] = count / (n_points * intensity)
        
        elif edge_correction == 'ripley':
            # Ripley's isotropic edge correction
            # Weight by proportion of circle inside study area
            total_weighted = 0
            
            for j in range(n_points):
                for k in range(n_points):
                    if j != k and dists[j, k] <= r:
                        # Compute edge correction weight
                        x_j, y_j = coords[j]
                        
                        # Distance to boundaries
                        d_left = x_j - x_min
                        d_right = x_max - x_j
                        d_bottom = y_j - y_min
                        d_top = y_max - y_j
                        
                        # Approximate correction (simplified)
                        # Full correction requires computing circle-rectangle intersection
                        min_edge_dist = min(d_left, d_right, d_bottom, d_top)
                        
                        if min_edge_dist >= r:
                            weight = 1.0  # Circle fully inside
                        else:
                            # Approximate weight
                            weight = 1.0 + (r - min_edge_dist) / r
                        
                        total_weighted += weight
            
            K_values[i] = total_weighted / (n_points * intensity)
        
        elif edge_correction == 'border':
            # Border method: exclude points too close to edge
            # Only count pairs where both points are > r from edge
            valid_points = []
            
            for j in range(n_points):
                x_j, y_j = coords[j]
                d_left = x_j - x_min
                d_right = x_max - x_j
                d_bottom = y_j - y_min
                d_top = y_max - y_j
                
                if min(d_left, d_right, d_bottom, d_top) >= r:
                    valid_points.append(j)
            
            if len(valid_points) > 0:
                valid_dists = dists[np.ix_(valid_points, valid_points)]
                within_r_valid = (valid_dists <= r) & (valid_dists > 0)
                count = within_r_valid.sum()
                K_values[i] = count / (len(valid_points) * intensity)
            else:
                K_values[i] = 0
        
        else:
            raise ValueError(f"Unknown edge_correction: {edge_correction}")
    
    # Expected K under CSR (Complete Spatial Randomness)
    K_csr = np.pi * radii**2
    
    # Create results DataFrame
    results = pd.DataFrame({
        'r': radii,
        'K': K_values,
        'K_csr': K_csr,
        'K_diff': K_values - K_csr
    })
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Radii tested: {len(radii)}")
    print(f"  Mean K(r): {K_values.mean():.2f}")
    print(f"  Mean deviation from CSR: {(K_values - K_csr).mean():.2f}")
    
    # Interpretation
    mean_diff = (K_values - K_csr).mean()
    if mean_diff > 0:
        print(f"  → Overall clustering detected")
    elif mean_diff < 0:
        print(f"  → Overall dispersion detected")
    else:
        print(f"  → Consistent with random pattern")
    
    print(f"{'='*70}\n")
    
    return results


def ripleys_l(sp: 'spatioloji',
             radii: Optional[np.ndarray] = None,
             n_radii: int = 50,
             cell_ids: Optional[List[str]] = None,
             edge_correction: str = 'ripley',
             fov_id: Optional[str] = None) -> pd.DataFrame:
    """
    Compute Ripley's L function (variance-stabilized K function).
    
    L(r) = sqrt(K(r) / π) - r
    
    The L function is easier to interpret than K:
    - L(r) > 0: Clustering at distance r
    - L(r) = 0: Complete Spatial Randomness (CSR)
    - L(r) < 0: Dispersion/regularity at distance r
    
    Uses global coordinates. For FOV-specific analysis, subset first.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    radii : np.ndarray, optional
        Distance radii to test
    n_radii : int, default=50
        Number of radii
    cell_ids : list, optional
        Subset of cells
    edge_correction : str, default='ripley'
        Edge correction method
    fov_id : str, optional
        Specific FOV
    
    Returns
    -------
    pd.DataFrame
        L function values with columns:
        - r: Distance radius
        - L: Ripley's L value
        - L_csr: Expected L under CSR (0)
        - L_diff: L - L_csr (same as L)
    
    Examples
    --------
    >>> # Compute L function
    >>> l_result = sj.spatial.point.ripleys_l(sp)
    >>> 
    >>> # Plot L function
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(l_result['r'], l_result['L'])
    >>> plt.axhline(0, color='red', linestyle='--', label='CSR')
    >>> plt.xlabel('Distance r')
    >>> plt.ylabel('L(r)')
    >>> plt.legend()
    >>> 
    >>> # Identify scales with clustering
    >>> clustered = l_result[l_result['L'] > 0]
    >>> print(f"Clustering at r = {clustered['r'].min():.1f} to {clustered['r'].max():.1f}")
    >>> 
    >>> # FOV-specific analysis
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> l_fov = sj.spatial.point.ripleys_l(sp_fov)
    """
    print(f"\n{'='*70}")
    print(f"Computing Ripley's L Function (Global Coordinates)")
    print(f"{'='*70}")
    
    # Compute K function first
    k_result = ripleys_k(
        sp, radii=radii, n_radii=n_radii,
        cell_ids=cell_ids, edge_correction=edge_correction, fov_id=fov_id
    )
    
    # Transform to L
    print(f"Transforming K to L...")
    L_values = np.sqrt(k_result['K'] / np.pi) - k_result['r']
    L_csr = np.zeros(len(L_values))  # Under CSR, L(r) = 0
    
    # Create results DataFrame
    results = pd.DataFrame({
        'r': k_result['r'],
        'L': L_values,
        'L_csr': L_csr,
        'L_diff': L_values  # Same as L since CSR is 0
    })
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Mean L(r): {L_values.mean():.3f}")
    print(f"  Min L(r): {L_values.min():.3f}")
    print(f"  Max L(r): {L_values.max():.3f}")
    
    # Interpretation
    if L_values.mean() > 0:
        print(f"  → Overall clustering pattern")
    elif L_values.mean() < 0:
        print(f"  → Overall dispersed pattern")
    else:
        print(f"  → Random pattern")
    
    print(f"{'='*70}\n")
    
    return results


def cross_k_function(sp: 'spatioloji',
                    group1: str,
                    group2: str,
                    groupby: str = 'cell_type',
                    radii: Optional[np.ndarray] = None,
                    n_radii: int = 50,
                    edge_correction: str = 'none') -> pd.DataFrame:
    """
    Compute cross-type K function for two cell populations.
    
    Measures spatial association between two cell types at different
    distances. Tests whether type 1 cells are clustered around type 2 cells.
    
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    group1 : str
        First cell type (reference population)
    group2 : str
        Second cell type (target population)
    groupby : str, default='cell_type'
        Column in cell_meta
    radii : np.ndarray, optional
        Distance radii
    n_radii : int, default=50
        Number of radii
    edge_correction : str, default='none'
        Edge correction method
    
    Returns
    -------
    pd.DataFrame
        Cross-K function with columns:
        - r: Distance radius
        - K_cross: Cross-K value
        - K_indep: Expected K under independence
        - K_diff: K_cross - K_indep
    
    Examples
    --------
    >>> # Test if T cells cluster around tumor cells
    >>> cross_k = sj.spatial.point.cross_k_function(
    ...     sp, group1='T_cell', group2='Tumor', groupby='cell_type'
    ... )
    >>> 
    >>> # Plot
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cross_k['r'], cross_k['K_cross'], label='Observed')
    >>> plt.plot(cross_k['r'], cross_k['K_indep'], label='Independence', linestyle='--')
    >>> plt.xlabel('Distance r')
    >>> plt.ylabel('K_cross(r)')
    >>> plt.legend()
    >>> 
    >>> # Check if T cells are attracted to tumor
    >>> if (cross_k['K_diff'] > 0).mean() > 0.5:
    ...     print("T cells cluster around tumor cells")
    """
    print(f"\n{'='*70}")
    print(f"Computing Cross-Type K Function (Global Coordinates)")
    print(f"{'='*70}")
    print(f"Group 1 (reference): {group1}")
    print(f"Group 2 (target): {group2}")
    
    if groupby not in sp.cell_meta.columns:
        raise ValueError(f"Column '{groupby}' not found in cell_meta")
    
    # Get cell populations
    cell_types = sp.cell_meta[groupby].astype(str)
    
    cells1 = sp.cell_index[cell_types == group1].tolist()
    cells2 = sp.cell_index[cell_types == group2].tolist()
    
    n1 = len(cells1)
    n2 = len(cells2)
    
    print(f"{group1}: {n1:,} cells")
    print(f"{group2}: {n2:,} cells")
    
    if n1 == 0 or n2 == 0:
        raise ValueError("One or both groups have no cells")
    
    # Get coordinates (always global)
    indices1 = sp._get_cell_indices(cells1)
    indices2 = sp._get_cell_indices(cells2)
    
    coords1 = sp.get_spatial_coords(coord_type='global')[indices1]
    coords2 = sp.get_spatial_coords(coord_type='global')[indices2]
    
    # Study area
    all_coords = np.vstack([coords1, coords2])
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    
    print(f"Study area: {area:.2f} square units")
    
    # Intensities
    intensity1 = n1 / area
    intensity2 = n2 / area
    
    print(f"Intensity {group1}: {intensity1:.6f}")
    print(f"Intensity {group2}: {intensity2:.6f}")
    
    # Determine radii
    if radii is None:
        max_dist = np.sqrt(area) / 4
        radii = np.linspace(0, max_dist, n_radii + 1)[1:]
    
    print(f"Radii: {len(radii)} values")
    
    # Compute cross-distances (group1 to group2)
    print(f"\nComputing cross-distances...")
    from scipy.spatial.distance import cdist
    cross_dists = cdist(coords1, coords2)
    
    # Compute cross-K for each radius
    print(f"Computing cross-K function...")
    K_cross = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        if i % 10 == 0 and i > 0:
            print(f"  Radius {i}/{len(radii)}...")
        
        # Count type2 cells within distance r of each type1 cell
        within_r = (cross_dists <= r).sum()
        
        # Cross-K
        K_cross[i] = within_r / (n1 * intensity2)
    
    # Expected K under independence (random labeling)
    K_indep = np.pi * radii**2
    
    # Create results
    results = pd.DataFrame({
        'r': radii,
        'K_cross': K_cross,
        'K_indep': K_indep,
        'K_diff': K_cross - K_indep
    })
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Mean K_cross: {K_cross.mean():.2f}")
    print(f"  Mean deviation from independence: {(K_cross - K_indep).mean():.2f}")
    
    mean_diff = (K_cross - K_indep).mean()
    if mean_diff > 0:
        print(f"  → {group1} cells cluster around {group2} cells")
    elif mean_diff < 0:
        print(f"  → {group1} cells avoid {group2} cells")
    else:
        print(f"  → Random spatial association")
    
    print(f"{'='*70}\n")
    
    return results


def cross_l_function(sp: 'spatioloji',
                    group1: str,
                    group2: str,
                    groupby: str = 'cell_type',
                    radii: Optional[np.ndarray] = None,
                    n_radii: int = 50) -> pd.DataFrame:
    """
    Compute cross-type L function (normalized cross-K).
    
    L_cross(r) = sqrt(K_cross(r) / π) - r
    
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    group1 : str
        First cell type
    group2 : str
        Second cell type
    groupby : str, default='cell_type'
        Column in cell_meta
    radii : np.ndarray, optional
        Distance radii
    n_radii : int, default=50
        Number of radii
    
    Returns
    -------
    pd.DataFrame
        Cross-L function
    
    Examples
    --------
    >>> cross_l = sj.spatial.point.cross_l_function(
    ...     sp, group1='T_cell', group2='Tumor'
    ... )
    >>> 
    >>> # Plot
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(cross_l['r'], cross_l['L_cross'])
    >>> plt.axhline(0, color='red', linestyle='--')
    >>> plt.xlabel('Distance r')
    >>> plt.ylabel('L_cross(r)')
    """
    # Compute cross-K first
    k_result = cross_k_function(
        sp, group1, group2, groupby=groupby,
        radii=radii, n_radii=n_radii
    )
    
    # Transform to L
    print(f"Transforming to L function...")
    L_cross = np.sqrt(k_result['K_cross'] / np.pi) - k_result['r']
    L_indep = np.zeros(len(L_cross))
    
    results = pd.DataFrame({
        'r': k_result['r'],
        'L_cross': L_cross,
        'L_indep': L_indep,
        'L_diff': L_cross
    })
    
    print(f"\nMean L_cross: {L_cross.mean():.3f}")
    
    return results


def test_csr(sp: 'spatioloji',
            cell_ids: Optional[List[str]] = None,
            n_simulations: int = 99,
            radii: Optional[np.ndarray] = None,
            n_radii: int = 30,
            statistic: str = 'L') -> Dict:
    """
    Test Complete Spatial Randomness (CSR) using Monte Carlo simulation.
    
    Compares observed K or L function to distribution under CSR.
    Uses global coordinates.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    cell_ids : list, optional
        Subset of cells
    n_simulations : int, default=99
        Number of CSR simulations
    radii : np.ndarray, optional
        Distance radii
    n_radii : int, default=30
        Number of radii
    statistic : str, default='L'
        'K' or 'L' function
    
    Returns
    -------
    dict
        Results with keys:
        - observed: Observed K or L values
        - simulations: Simulated values
        - envelope_low: Lower envelope (2.5th percentile)
        - envelope_high: Upper envelope (97.5th percentile)
        - p_value: P-value for deviation from CSR
        - p_values_per_r: P-values for each radius
        - radii: Distance radii
    
    Examples
    --------
    >>> # Test T cells for CSR
    >>> tcells = sp.cell_meta[sp.cell_meta['cell_type'] == 'T_cell'].index
    >>> csr_result = sj.spatial.point.test_csr(
    ...     sp, cell_ids=tcells, n_simulations=99
    ... )
    >>> 
    >>> # Plot with confidence envelope
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(csr_result['radii'], csr_result['observed'], label='Observed')
    >>> plt.fill_between(
    ...     csr_result['radii'],
    ...     csr_result['envelope_low'],
    ...     csr_result['envelope_high'],
    ...     alpha=0.3, label='95% CSR envelope'
    ... )
    >>> plt.legend()
    >>> 
    >>> # FOV-specific CSR test
    >>> sp_fov = sp.subset_by_fovs(['fov1'])
    >>> csr_fov = sj.spatial.point.test_csr(sp_fov, n_simulations=99)
    """
    print(f"\n{'='*70}")
    print(f"Testing Complete Spatial Randomness (Global Coordinates)")
    print(f"{'='*70}")
    print(f"Simulations: {n_simulations}")
    print(f"Statistic: {statistic}")
    
    # Get coordinates (always global)
    if cell_ids is not None:
        indices = sp._get_cell_indices(cell_ids)
        coords = sp.get_spatial_coords(coord_type='global')[indices]
    else:
        coords = sp.get_spatial_coords(coord_type='global')
    
    n_points = len(coords)
    print(f"Points: {n_points:,}")
    
    # Study area
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    
    # Compute observed statistic
    print(f"\nComputing observed {statistic} function...")
    
    if statistic == 'K':
        obs_result = ripleys_k(
            sp, radii=radii, n_radii=n_radii,
            cell_ids=cell_ids, edge_correction='none'
        )
        observed = obs_result['K'].values
    else:  # L
        obs_result = ripleys_l(
            sp, radii=radii, n_radii=n_radii,
            cell_ids=cell_ids, edge_correction='none'
        )
        observed = obs_result['L'].values
    
    radii = obs_result['r'].values
    
    # Run simulations
    print(f"\nRunning {n_simulations} CSR simulations...")
    simulations = np.zeros((n_simulations, len(radii)))
    
    for sim in range(n_simulations):
        if sim % 20 == 0 and sim > 0:
            print(f"  Simulation {sim}/{n_simulations}...")
        
        # Generate random points in study area
        sim_coords = np.random.uniform(
            low=[x_min, y_min],
            high=[x_max, y_max],
            size=(n_points, 2)
        )
        
        # Compute statistic for simulated points
        intensity = n_points / area
        dists = distance_matrix(sim_coords, sim_coords)
        
        for i, r in enumerate(radii):
            within_r = (dists <= r) & (dists > 0)
            count = within_r.sum()
            K_sim = count / (n_points * intensity)
            
            if statistic == 'K':
                simulations[sim, i] = K_sim
            else:  # L
                simulations[sim, i] = np.sqrt(K_sim / np.pi) - r
    
    # Compute envelopes
    envelope_low = np.percentile(simulations, 2.5, axis=0)
    envelope_high = np.percentile(simulations, 97.5, axis=0)
    
    # P-value (two-tailed)
    # Proportion of simulations more extreme than observed
    deviations_obs = np.abs(observed - simulations.mean(axis=0))
    deviations_sim = np.abs(simulations - simulations.mean(axis=0))
    
    p_values_per_r = np.mean(deviations_sim >= deviations_obs[np.newaxis, :], axis=0)
    p_value_global = np.mean(p_values_per_r < 0.05)  # Overall significance
    
    results = {
        'observed': observed,
        'simulations': simulations,
        'envelope_low': envelope_low,
        'envelope_high': envelope_high,
        'p_value': p_value_global,
        'p_values_per_r': p_values_per_r,
        'radii': radii
    }
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Global p-value: {p_value_global:.3f}")
    
    # Check if outside envelope
    outside = (observed < envelope_low) | (observed > envelope_high)
    n_outside = outside.sum()
    
    print(f"  Radii outside envelope: {n_outside}/{len(radii)}")
    
    if n_outside > len(radii) * 0.05:
        if np.mean(observed > envelope_high) > 0.5:
            print(f"  → Significant clustering (p < 0.05)")
        else:
            print(f"  → Significant dispersion (p < 0.05)")
    else:
        print(f"  → Consistent with CSR")
    
    print(f"{'='*70}\n")
    
    return results