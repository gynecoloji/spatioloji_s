"""
utils.py - Utility functions for spatioloji package

Provides helper functions for data conversion, visualization, 
spatial analysis, and common operations on spatioloji objects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import anndata
from pathlib import Path

def export_to_csv_bundle(sp, output_dir: str, include_qc: bool = True) -> None:
    """
    Export all spatioloji components to CSV files.
    
    Parameters
    ----------
    sp : spatioloji
        spatioloji object
    output_dir : str
        Directory to save files
    include_qc : bool
        Whether to include QC flags from adata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Core data
    sp.polygons.to_csv(output_path / 'polygons.csv', index=False)
    sp.cell_meta.to_csv(output_path / 'cell_meta.csv', index=False)
    sp.fov_positions.to_csv(output_path / 'fov_positions.csv', index=False)
    
    # AnnData
    sp.adata.write_h5ad(output_path / 'adata.h5ad')
    
    # Expression matrix
    sp.adata.to_df().to_csv(output_path / 'expression_matrix.csv')
    
    # Cell observations (including QC if available)
    sp.adata.obs.to_csv(output_path / 'cell_observations.csv')
    
    # Gene annotations
    sp.adata.var.to_csv(output_path / 'gene_annotations.csv')
    
    print(f"✓ Exported bundle to {output_dir}")