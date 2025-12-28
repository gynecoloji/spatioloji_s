"""
batch_correction.py - Batch correction methods for spatial transcriptomics

Provides methods for correcting batch effects across samples, experiments,
or FOVs while preserving biological variation.
"""

from typing import Optional, Literal, Union, List, Dict
import numpy as np
import pandas as pd
from scipy import sparse
import warnings


def combat(
    spatioloji_obj,
    batch_key: str,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = False,
    covariates: Optional[List[str]] = None,
    output_layer: str = 'combat_corrected',
    inplace: bool = True
):
    """
    ComBat batch correction.
    
    Adjusts for batch effects using empirical Bayes framework.
    This is one of the most widely used batch correction methods.
    Requires `combat` from `pycombat` package.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
        (e.g., 'sample_id', 'batch', 'fov_id')
    layer : str, optional
        Which layer to correct, by default 'log_normalized'
    use_highly_variable : bool, optional
        Correct only highly variable genes, by default False
    covariates : list of str, optional
        Covariates to preserve (e.g., ['cell_type']), by default None
    output_layer : str, optional
        Name for corrected layer, by default 'combat_corrected'
    inplace : bool, optional
        Add corrected layer to spatioloji object, by default True
    
    Returns
    -------
    np.ndarray or None
        Corrected expression if inplace=False, otherwise None
    
    Examples
    --------
    >>> # Basic batch correction by sample
    >>> sp.processing.combat(sp, batch_key='sample_id')
    >>> 
    >>> # Preserve cell type differences
    >>> sp.processing.combat(
    ...     sp, 
    ...     batch_key='sample_id',
    ...     covariates=['cell_type']
    ... )
    >>> 
    >>> # Correct only HVGs (faster)
    >>> sp.processing.combat(
    ...     sp,
    ...     batch_key='batch',
    ...     use_highly_variable=True
    ... )
    
    References
    ----------
    Johnson et al. (2007) Biostatistics
    """
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError(
            "ComBat requires pycombat package. "
            "Install with: pip install pycombat"
        )
    
    print(f"\nComBat batch correction (batch_key='{batch_key}')")
    
    # Check batch column exists
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")
    
    batch = spatioloji_obj.cell_meta[batch_key].values
    n_batches = len(np.unique(batch))
    
    print(f"  Found {n_batches} batches")
    
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()
    
    # Subset to HVGs if requested
    gene_mask = None
    if use_highly_variable:
        if 'highly_variable' in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
            if gene_mask.sum() > 0:
                X = X[:, gene_mask]
                print(f"  Correcting {gene_mask.sum()} highly variable genes")
            else:
                warnings.warn("No HVGs found, using all genes")
        else:
            warnings.warn("No 'highly_variable' column found, using all genes")
    
    # Prepare data for pycombat (genes × cells)
    X_t = X.T
    
    # Create DataFrame for pycombat
    df = pd.DataFrame(
        X_t,
        index=[f"gene_{i}" for i in range(X_t.shape[0])],
        columns=spatioloji_obj.cell_index
    )
    
    # Prepare covariate DataFrame if provided
    covar_df = None
    if covariates is not None:
        covar_df = spatioloji_obj.cell_meta[covariates].copy()
        print(f"  Preserving covariates: {covariates}")
    
    # Run ComBat
    print(f"  Running ComBat...")
    try:
        corrected = pycombat(df, batch, mod=covar_df)
        X_corrected = corrected.T.values
    except Exception as e:
        print(f"  Error running pycombat: {e}")
        print(f"  Trying alternative implementation...")
        
        # Fallback to scanpy's combat if available
        try:
            import scanpy as sc
            import anndata
            
            adata_temp = anndata.AnnData(X=X)
            adata_temp.obs[batch_key] = batch
            
            if covariates is not None:
                for cov in covariates:
                    adata_temp.obs[cov] = spatioloji_obj.cell_meta[cov].values
            
            sc.pp.combat(adata_temp, key=batch_key, covariates=covariates)
            X_corrected = adata_temp.X
            
            if sparse.issparse(X_corrected):
                X_corrected = X_corrected.toarray()
        except:
            raise RuntimeError("Failed to run ComBat with both pycombat and scanpy")
    
    print(f"  ✓ ComBat correction complete")
    
    # If we only corrected HVGs, need to merge back
    if gene_mask is not None:
        X_full = X.copy() if layer is None else spatioloji_obj.get_layer(layer).copy()
        if sparse.issparse(X_full):
            X_full = X_full.toarray()
        X_full[:, gene_mask] = X_corrected
        X_corrected = X_full
    
    if inplace:
        spatioloji_obj.add_layer(output_layer, X_corrected, overwrite=True)
        return None
    else:
        return X_corrected


def harmony(
    spatioloji_obj,
    batch_key: str,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    max_iter_harmony: int = 10,
    theta: float = 2.0,
    sigma: float = 0.1,
    random_state: int = 42,
    output_key: str = 'X_pca_harmony',
    inplace: bool = True
):
    """
    Harmony batch correction in PCA space.
    
    Fast integration method that corrects batch effects in reduced dimension
    space (typically PCA). Preserves more variance than ComBat and is
    scalable to large datasets. Requires `harmonypy` package.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
    use_pca : bool, optional
        Run on PCA coordinates, by default True (recommended)
    n_pcs : int, optional
        Number of PCs to use, by default 50
    layer : str, optional
        Which layer to use if not using PCA, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only HVGs for PCA, by default True
    max_iter_harmony : int, optional
        Maximum Harmony iterations, by default 10
    theta : float, optional
        Diversity clustering penalty (0-inf), by default 2.0
        Higher values = more aggressive correction
    sigma : float, optional
        Ridge regression penalty, by default 0.1
    random_state : int, optional
        Random seed, by default 42
    output_key : str, optional
        Key for storing corrected embeddings, by default 'X_pca_harmony'
    inplace : bool, optional
        Store results in spatioloji object, by default True
    
    Returns
    -------
    np.ndarray or None
        Harmony-corrected coordinates if inplace=False, otherwise None
    
    Examples
    --------
    >>> # Standard Harmony on PCA
    >>> sp.processing.pca(sp, n_comps=50)
    >>> sp.processing.harmony(sp, batch_key='sample_id')
    >>> 
    >>> # More aggressive correction
    >>> sp.processing.harmony(sp, batch_key='batch', theta=5.0)
    >>> 
    >>> # Then use corrected PCA for clustering
    >>> # (modify clustering functions to accept 'use_harmony' parameter)
    
    References
    ----------
    Korsunsky et al. (2019) Nature Methods
    """
    try:
        import harmonypy as hm
    except ImportError:
        raise ImportError(
            "Harmony requires harmonypy package. "
            "Install with: pip install harmonypy"
        )
    
    print(f"\nHarmony integration (batch_key='{batch_key}', theta={theta})")
    
    # Check batch column exists
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")
    
    batch = spatioloji_obj.cell_meta[batch_key].values
    n_batches = len(np.unique(batch))
    
    print(f"  Found {n_batches} batches")
    
    # Get input data
    if use_pca:
        # Use or compute PCA
        if hasattr(spatioloji_obj, '_embeddings') and 'X_pca' in spatioloji_obj._embeddings:
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs")
        else:
            print(f"  Computing PCA first...")
            from .dimension_reduction import pca as run_pca
            run_pca(
                spatioloji_obj,
                layer=layer,
                use_highly_variable=use_highly_variable,
                n_comps=n_pcs,
                inplace=True
            )
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
    else:
        # Use expression data
        if layer is None:
            X = spatioloji_obj.expression.get_dense()
        else:
            X = spatioloji_obj.get_layer(layer)
            if sparse.issparse(X):
                X = X.toarray()
        
        if use_highly_variable and 'highly_variable' in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
            if gene_mask.sum() > 0:
                X = X[:, gene_mask]
                print(f"  Using {gene_mask.sum()} highly variable genes")
    
    # Prepare metadata DataFrame for Harmony
    meta_data = pd.DataFrame({batch_key: batch})
    
    # Run Harmony
    print(f"  Running Harmony (max_iter={max_iter_harmony})...")
    ho = hm.run_harmony(
        X,
        meta_data,
        batch_key,
        theta=theta,
        sigma=sigma,
        max_iter_harmony=max_iter_harmony,
        random_state=random_state,
        verbose=False
    )
    
    X_corrected = ho.Z_corr.T  # Transpose to get (cells × components)
    
    print(f"  ✓ Harmony integration complete")
    print(f"    Converged after {ho.n_iter} iterations")
    
    if inplace:
        if not hasattr(spatioloji_obj, '_embeddings'):
            spatioloji_obj._embeddings = {}
        
        spatioloji_obj._embeddings[output_key] = X_corrected
        
        # Add to cell_meta
        for i in range(X_corrected.shape[1]):
            spatioloji_obj._cell_meta[f'PC_harmony{i+1}'] = X_corrected[:, i]
        
        return None
    else:
        return X_corrected


def regress_out(
    spatioloji_obj,
    keys: Union[str, List[str]],
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = False,
    output_layer: str = 'regressed',
    inplace: bool = True
):
    """
    Regress out unwanted sources of variation.
    
    Removes the effect of specified variables (e.g., total counts, percent
    mitochondrial) using linear regression.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    keys : str or list of str
        Column name(s) in cell_meta to regress out
        (e.g., 'total_counts', 'pct_mito')
    layer : str, optional
        Which layer to use, by default 'log_normalized'
    use_highly_variable : bool, optional
        Regress only HVGs, by default False
    output_layer : str, optional
        Name for regressed layer, by default 'regressed'
    inplace : bool, optional
        Add regressed layer to spatioloji object, by default True
    
    Returns
    -------
    np.ndarray or None
        Regressed expression if inplace=False, otherwise None
    
    Examples
    --------
    >>> # Regress out total counts
    >>> sp.processing.regress_out(sp, keys='total_counts')
    >>> 
    >>> # Regress out multiple variables
    >>> sp.processing.regress_out(
    ...     sp,
    ...     keys=['total_counts', 'pct_mito']
    ... )
    
    Notes
    -----
    This can over-correct and remove biological signal. Use with caution.
    Consider using ComBat or Harmony instead for batch correction.
    """
    from sklearn.linear_model import LinearRegression
    
    print(f"\nRegressing out: {keys}")
    
    if isinstance(keys, str):
        keys = [keys]
    
    # Check all keys exist
    for key in keys:
        if key not in spatioloji_obj.cell_meta.columns:
            raise ValueError(f"Column '{key}' not found in cell_meta")
    
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()
    
    # Subset to HVGs if requested
    gene_mask = None
    if use_highly_variable:
        if 'highly_variable' in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
            if gene_mask.sum() > 0:
                X_regress = X[:, gene_mask]
                print(f"  Regressing {gene_mask.sum()} highly variable genes")
            else:
                X_regress = X
        else:
            X_regress = X
    else:
        X_regress = X
    
    # Prepare covariate matrix
    covariates = spatioloji_obj.cell_meta[keys].values
    
    # Handle categorical variables
    from sklearn.preprocessing import LabelEncoder
    covariates_processed = []
    for i, key in enumerate(keys):
        col = spatioloji_obj.cell_meta[key].values
        if pd.api.types.is_numeric_dtype(col):
            covariates_processed.append(col.reshape(-1, 1))
        else:
            # Encode categorical
            le = LabelEncoder()
            encoded = le.fit_transform(col.astype(str))
            covariates_processed.append(encoded.reshape(-1, 1))
    
    covariates = np.hstack(covariates_processed)
    
    # Regress out per gene
    print(f"  Regressing out effects...")
    X_regressed = X_regress.copy()
    
    for i in range(X_regress.shape[1]):
        lr = LinearRegression()
        lr.fit(covariates, X_regress[:, i])
        predicted = lr.predict(covariates)
        X_regressed[:, i] = X_regress[:, i] - predicted
    
    print(f"  ✓ Regression complete")
    
    # If we only regressed HVGs, merge back
    if gene_mask is not None:
        X_full = X.copy()
        X_full[:, gene_mask] = X_regressed
        X_regressed = X_full
    
    if inplace:
        spatioloji_obj.add_layer(output_layer, X_regressed, overwrite=True)
        return None
    else:
        return X_regressed


def scale_by_batch(
    spatioloji_obj,
    batch_key: str,
    layer: Optional[str] = 'log_normalized',
    method: Literal['standard', 'robust'] = 'standard',
    output_layer: str = 'batch_scaled',
    inplace: bool = True
):
    """
    Scale expression within each batch separately.
    
    Simple batch correction that scales each batch to have the same mean
    and variance. Less sophisticated than ComBat but faster and simpler.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
    layer : str, optional
        Which layer to scale, by default 'log_normalized'
    method : {'standard', 'robust'}, optional
        Scaling method, by default 'standard'
        - 'standard': z-score within batch
        - 'robust': use median and MAD
    output_layer : str, optional
        Name for scaled layer, by default 'batch_scaled'
    inplace : bool, optional
        Add scaled layer to spatioloji object, by default True
    
    Returns
    -------
    np.ndarray or None
        Scaled expression if inplace=False, otherwise None
    
    Examples
    --------
    >>> # Standard scaling within each batch
    >>> sp.processing.scale_by_batch(sp, batch_key='sample_id')
    >>> 
    >>> # Robust scaling (better for outliers)
    >>> sp.processing.scale_by_batch(
    ...     sp,
    ...     batch_key='batch',
    ...     method='robust'
    ... )
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    print(f"\nScaling by batch (batch_key='{batch_key}', method={method})")
    
    # Check batch column exists
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")
    
    batch = spatioloji_obj.cell_meta[batch_key].values
    batches = np.unique(batch)
    n_batches = len(batches)
    
    print(f"  Found {n_batches} batches")
    
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()
    
    X_scaled = np.zeros_like(X)
    
    # Scale each batch separately
    for batch_id in batches:
        batch_mask = batch == batch_id
        n_cells_batch = batch_mask.sum()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_scaled[batch_mask, :] = scaler.fit_transform(X[batch_mask, :])
        
        print(f"    Scaled batch '{batch_id}': {n_cells_batch} cells")
    
    print(f"  ✓ Batch scaling complete")
    
    if inplace:
        spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
        return None
    else:
        return X_scaled


def mnn_correct(
    spatioloji_obj,
    batch_key: str,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    k: int = 20,
    sigma: float = 1.0,
    n_pcs: int = 50,
    conda_env: Optional[str] = None,
    output_layer: str = 'mnn_corrected',
    inplace: bool = True
):
    """
    Mutual Nearest Neighbors (MNN) batch correction.
    
    Identifies mutual nearest neighbors between batches and uses them to
    learn batch-specific corrections. Can run in a separate conda environment
    if mnnpy has dependency conflicts.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
    layer : str, optional
        Which layer to correct, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only HVGs, by default True
    k : int, optional
        Number of nearest neighbors, by default 20
    sigma : float, optional
        Gaussian smoothing parameter, by default 1.0
    n_pcs : int, optional
        Number of PCs to use for finding neighbors, by default 50
    conda_env : str, optional
        Name of conda environment with mnnpy installed, by default None
        If None, tries to import mnnpy from current environment
        If specified, runs MNN in that environment via subprocess
    output_layer : str, optional
        Name for corrected layer, by default 'mnn_corrected'
    inplace : bool, optional
        Add corrected layer to spatioloji object, by default True
    
    Returns
    -------
    np.ndarray or None
        MNN-corrected expression if inplace=False, otherwise None
    
    Examples
    --------
    >>> # Try current environment first
    >>> sp.processing.mnn_correct(sp, batch_key='sample_id', k=20)
    
    >>> # Use separate conda environment (recommended if mnnpy fails to install)
    >>> # First create environment: conda create -n mnnpy_env python=3.8 mnnpy -c bioconda
    >>> sp.processing.mnn_correct(
    ...     sp, 
    ...     batch_key='sample_id',
    ...     conda_env='mnnpy_env',
    ...     k=20
    ... )
    
    References
    ----------
    Haghverdi et al. (2018) Nature Biotechnology
    
    Notes
    -----
    If using conda_env, make sure the environment has mnnpy installed:
        conda create -n mnnpy_env python=3.8
        conda activate mnnpy_env
        conda install -c bioconda mnnpy
        # or: pip install mnnpy
    """
    import tempfile
    import os
    
    print(f"\nMNN batch correction (batch_key='{batch_key}', k={k})")
    
    # Check batch column exists
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")
    
    batch = spatioloji_obj.cell_meta[batch_key].values
    batches = np.unique(batch)
    n_batches = len(batches)
    
    if n_batches < 2:
        raise ValueError("Need at least 2 batches for MNN correction")
    
    print(f"  Found {n_batches} batches")
    
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()
    
    # Subset to HVGs if requested
    if use_highly_variable and 'highly_variable' in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
        if gene_mask.sum() > 0:
            X = X[:, gene_mask]
            print(f"  Using {gene_mask.sum()} highly variable genes")
    
    # If conda_env is specified, run in separate environment
    if conda_env is not None:
        print(f"  Running MNN in conda environment: '{conda_env}'")
        X_corrected = _run_mnn_in_conda_env(
            X, batch, batches, k, sigma, n_pcs, conda_env
        )
    else:
        # Try to import mnnpy from current environment
        try:
            import mnnpy
        except ImportError:
            raise ImportError(
                "mnnpy not found in current environment. Either:\n"
                "  1. Install mnnpy: pip install mnnpy\n"
                "  2. Or specify conda_env with mnnpy installed:\n"
                "     sp.processing.mnn_correct(sp, batch_key='...', conda_env='mnnpy_env')"
            )
        
        # Split data by batch
        batch_data = []
        for batch_id in batches:
            batch_mask = batch == batch_id
            batch_data.append(X[batch_mask, :])
            print(f"    Batch '{batch_id}': {batch_mask.sum()} cells")
        
        # Run MNN
        print(f"  Running MNN correction...")
        corrected, _, _ = mnnpy.mnn_correct(
            *batch_data,
            k=k,
            sigma=sigma,
            svd_dim=n_pcs,
            var_adj=True
        )
        
        X_corrected = corrected[0]
    
    print(f"  ✓ MNN correction complete")
    
    if inplace:
        spatioloji_obj.add_layer(output_layer, X_corrected, overwrite=True)
        return None
    else:
        return X_corrected


def _run_mnn_in_conda_env(X, batch, batches, k, sigma, n_pcs, conda_env):
    """
    Helper function to run MNN correction in a separate conda environment.
    
    This creates temporary files, calls a Python script in the specified
    conda environment, and loads the results back.
    """
    import tempfile
    import subprocess
    import os
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input data
        input_file = os.path.join(tmpdir, 'mnn_input.npz')
        np.savez_compressed(
            input_file,
            X=X,
            batch=batch,
            batches=batches,
            k=k,
            sigma=sigma,
            n_pcs=n_pcs
        )
        
        output_file = os.path.join(tmpdir, 'mnn_output.npy')
        
        # Create Python script for MNN
        script_file = os.path.join(tmpdir, 'run_mnn.py')
        script_content = f"""
import numpy as np
import mnnpy

# Load input
data = np.load('{input_file}', allow_pickle=True)
X = data['X']
batch = data['batch']
batches = data['batches']
k = int(data['k'])
sigma = float(data['sigma'])
n_pcs = int(data['n_pcs'])

# Split data by batch
batch_data = []
for batch_id in batches:
    batch_mask = batch == batch_id
    batch_data.append(X[batch_mask, :])
    print(f"Batch '{{batch_id}}': {{batch_mask.sum()}} cells")

# Run MNN
print("Running MNN correction...")
corrected, _, _ = mnnpy.mnn_correct(
    *batch_data,
    k=k,
    sigma=sigma,
    svd_dim=n_pcs,
    var_adj=True
)

# Save output
np.save('{output_file}', corrected[0])
print("MNN correction complete!")
"""
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Run script in conda environment
        print(f"    Calling conda environment '{conda_env}'...")
        
        # Try different conda activation methods
        try:
            # Method 1: Using conda run (newer conda versions)
            cmd = ['conda', 'run', '-n', conda_env, 'python', script_file]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e1:
            try:
                # Method 2: Using shell script with source activate
                shell_script = os.path.join(tmpdir, 'run_mnn.sh')
                with open(shell_script, 'w') as f:
                    f.write(f"""#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}
python {script_file}
""")
                
                result = subprocess.run(
                    ['bash', shell_script],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(result.stdout)
                
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to run MNN in conda environment '{conda_env}'.\n"
                    f"Error 1 (conda run): {str(e1)}\n"
                    f"Error 2 (conda activate): {str(e2)}\n"
                    f"Make sure the environment exists and has mnnpy installed:\n"
                    f"  conda create -n {conda_env} python=3.8\n"
                    f"  conda activate {conda_env}\n"
                    f"  conda install -c bioconda mnnpy"
                )
        
        # Load results
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"MNN correction failed - output file not created.\n"
                f"Check that mnnpy is properly installed in '{conda_env}'"
            )
        
        X_corrected = np.load(output_file)
        
        return X_corrected


def evaluate_batch_correction(
    spatioloji_obj,
    batch_key: str,
    embedding_key: str = 'X_pca',
    n_neighbors: int = 30
) -> Dict[str, float]:
    """
    Evaluate batch correction quality.
    
    Calculates metrics to assess how well batch effects were removed while
    preserving biological variation.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with embeddings
    batch_key : str
        Column name in cell_meta that defines batches
    embedding_key : str, optional
        Which embedding to evaluate, by default 'X_pca'
    n_neighbors : int, optional
        Number of neighbors for metrics, by default 30
    
    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        - 'batch_mixing': Higher = better mixing (0-1)
        - 'kbet': Higher = better mixing (0-1)
        - 'silhouette_batch': Lower = better (negative is good)
    
    Examples
    --------
    >>> # Evaluate before correction
    >>> metrics_before = sp.processing.evaluate_batch_correction(
    ...     sp,
    ...     batch_key='sample_id',
    ...     embedding_key='X_pca'
    ... )
    >>> 
    >>> # Apply correction
    >>> sp.processing.harmony(sp, batch_key='sample_id')
    >>> 
    >>> # Evaluate after correction
    >>> metrics_after = sp.processing.evaluate_batch_correction(
    ...     sp,
    ...     batch_key='sample_id',
    ...     embedding_key='X_pca_harmony'
    ... )
    >>> 
    >>> print(f"Batch mixing improved: {metrics_after['batch_mixing'] > metrics_before['batch_mixing']}")
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    
    print(f"\nEvaluating batch correction...")
    
    # Get embedding
    if not hasattr(spatioloji_obj, '_embeddings') or embedding_key not in spatioloji_obj._embeddings:
        raise ValueError(f"Embedding '{embedding_key}' not found")
    
    X = spatioloji_obj._embeddings[embedding_key]
    batch = spatioloji_obj.cell_meta[batch_key].values
    
    # Batch mixing score (fraction of neighbors from different batches)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    
    mixing_scores = []
    for i in range(len(X)):
        neighbor_batches = batch[indices[i, 1:]]  # Exclude self
        different_batch = neighbor_batches != batch[i]
        mixing_scores.append(different_batch.mean())
    
    batch_mixing = np.mean(mixing_scores)
    
    # Silhouette score for batches (lower is better)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    batch_encoded = le.fit_transform(batch.astype(str))
    
    sil_batch = silhouette_score(X, batch_encoded)
    
    print(f"  Batch mixing score: {batch_mixing:.3f} (higher = better mixing)")
    print(f"  Silhouette (batch): {sil_batch:.3f} (lower = better)")
    
    return {
        'batch_mixing': batch_mixing,
        'silhouette_batch': sil_batch
    }