"""
batch_correction.py - Batch correction methods for spatial transcriptomics

Provides methods for correcting batch effects across samples, experiments,
or FOVs while preserving biological variation.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse


def combat(
    spatioloji_obj,
    batch_key: str,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = False,
    covariates: list[str] | None = None,
    output_layer: str = "combat_corrected",
    inplace: bool = True,
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
    except ImportError as err:
        raise ImportError("ComBat requires pycombat package. " "Install with: pip install pycombat") from err

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
        if "highly_variable" in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
            if gene_mask.sum() > 0:
                X = X[:, gene_mask]
                print(f"  Correcting {gene_mask.sum()} highly variable genes")
            else:
                warnings.warn("No HVGs found, using all genes", stacklevel=2)
        else:
            warnings.warn("No 'highly_variable' column found, using all genes", stacklevel=2)

    # Prepare data for pycombat (genes × cells)
    X_t = X.T

    # Create DataFrame for pycombat
    df = pd.DataFrame(X_t, index=[f"gene_{i}" for i in range(X_t.shape[0])], columns=spatioloji_obj.cell_index)

    # Prepare covariate DataFrame if provided
    covar_df = None
    if covariates is not None:
        covar_df = spatioloji_obj.cell_meta[covariates].copy()
        print(f"  Preserving covariates: {covariates}")

    # Run ComBat
    print("  Running ComBat...")
    try:
        corrected = pycombat(df, batch, mod=covar_df)
        X_corrected = corrected.T.values
    except Exception as e:
        print(f"  Error running pycombat: {e}")
        print("  Trying alternative implementation...")

        # Fallback to scanpy's combat if available
        try:
            import anndata
            import scanpy as sc

            adata_temp = anndata.AnnData(X=X)
            adata_temp.obs[batch_key] = batch

            if covariates is not None:
                for cov in covariates:
                    adata_temp.obs[cov] = spatioloji_obj.cell_meta[cov].values

            sc.pp.combat(adata_temp, key=batch_key, covariates=covariates)
            X_corrected = adata_temp.X

            if sparse.issparse(X_corrected):
                X_corrected = X_corrected.toarray()
        except Exception as err:
            raise RuntimeError("Failed to run ComBat with both pycombat and scanpy") from err

    print("  ✓ ComBat correction complete")

    # If we only corrected HVGs, need to merge back
    if gene_mask is not None:
        warnings.warn(
            "ComBat was run on HVGs only. Merging corrected HVGs "
            "with uncorrected non-HVGs creates an inconsistent matrix. "
            "Use use_highly_variable=False for consistent correction, "
            "or use the output for HVG-only downstream analysis.",
            UserWarning, stacklevel=2
        )

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_corrected, overwrite=True)
        return None
    else:
        return X_corrected


def harmony(
    spatioloji_obj,
    batch_key: str,
    n_pcs: int = 50,
    layer: str | None = "scaled",       # ✅ fixed: scaled instead of log_normalized
    use_highly_variable: bool = True,
    max_iter_harmony: int = 10,
    theta: float = 2.0,
    sigma: float = 0.1,
    random_state: int = 42,
    output_key: str = "X_pca_harmony",
    inplace: bool = True,
    # removed use_pca parameter — Harmony always operates on PCA
):
    """
    Harmony batch correction in PCA space.

    Fast integration method that corrects batch effects in reduced dimension
    space (PCA). Preserves more variance than ComBat and is scalable to
    large datasets. Requires `harmonypy` package.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
        (e.g., 'sample_id', 'batch', 'fov_id')
    n_pcs : int, optional
        Number of PCs to use for Harmony, by default 50
    layer : str, optional
        Which layer to use if PCA needs to be computed.
        Should be scaled (mean=0, std=1), by default 'scaled'
    use_highly_variable : bool, optional
        Use only HVGs if PCA needs to be computed, by default True
    max_iter_harmony : int, optional
        Maximum Harmony iterations, by default 10
    theta : float, optional
        Diversity clustering penalty (0-inf), by default 2.0
        Higher values = more aggressive batch correction
    sigma : float, optional
        Ridge regression penalty, by default 0.1
    random_state : int, optional
        Random seed for reproducibility, by default 42
    output_key : str, optional
        Key for storing corrected embeddings, by default 'X_pca_harmony'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Harmony-corrected coordinates (n_cells x n_pcs) if inplace=False,
        otherwise None

    Notes
    -----
    - Always operates on PCA embedding — run pca() first for best results.
    - If PCA not found in object, computes it automatically using scaled layer.
    - Input layer must be scaled (mean=0, std=1) before PCA computation.
    - Output is a corrected PCA embedding, NOT corrected expression values.
    - Do NOT use Harmony-corrected embedding for pseudobulk DE analysis —
      always use raw counts with batch as a covariate in the design matrix.

    Examples
    --------
    >>> # Recommended workflow
    >>> sp.processing.pca(sp, layer='scaled', n_comps=50)
    >>> sp.processing.harmony(sp, batch_key='sample_id')
    >>>
    >>> # More aggressive correction
    >>> sp.processing.harmony(sp, batch_key='batch', theta=5.0)
    >>>
    >>> # Access corrected embedding
    >>> X_harmony = sp.embeddings['X_pca_harmony']

    References
    ----------
    Korsunsky et al. (2019) Nature Methods
    """
    try:
        import harmonypy as hm
    except ImportError as err:
        raise ImportError(
            "Harmony requires harmonypy package. "
            "Install with: pip install harmonypy"
        ) from err

    print(f"\nHarmony integration (batch_key='{batch_key}', theta={theta})")

    # Check batch column exists
    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    batch = spatioloji_obj.cell_meta[batch_key].values
    n_batches = len(np.unique(batch))
    print(f"  Found {n_batches} batches")

    # --- Always operate on PCA embedding ---
    # Use existing PCA if available, otherwise compute it
    if hasattr(spatioloji_obj, "_embeddings") and \
       "X_pca" in spatioloji_obj._embeddings:
        X = spatioloji_obj._embeddings["X_pca"][:, :n_pcs]
        print(f"  Using existing PCA ({n_pcs} PCs)")
    else:
        print("  PCA not found — computing first...")
        print(f"  (using layer='{layer}', use_highly_variable={use_highly_variable})")
        from .dimension_reduction import pca as run_pca
        run_pca(
            spatioloji_obj,
            layer=layer,
            use_highly_variable=use_highly_variable,
            n_comps=n_pcs,
            inplace=True
        )
        X = spatioloji_obj._embeddings["X_pca"][:, :n_pcs]

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
        verbose=False,
    )

    X_corrected = ho.Z_corr

    print("  ✓ Harmony integration complete")
    print(f"    Output shape: {X_corrected.shape}")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}

        spatioloji_obj._embeddings[output_key] = X_corrected

        # Add to cell_meta for easy access
        for i in range(X_corrected.shape[1]):
            spatioloji_obj._cell_meta[f"PC_harmony{i+1}"] = X_corrected[:, i]

        return None
    else:
        return X_corrected


def regress_out(
    spatioloji_obj,
    keys: str | list[str],
    layer: str | None = "log_normalized",
    use_highly_variable: bool = False,
    output_layer: str = "regressed",
    inplace: bool = True,
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
        if "highly_variable" in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
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
    for _i, key in enumerate(keys):
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
    print("  Regressing out effects...")
    X_regressed = X_regress.copy()

    for i in range(X_regress.shape[1]):
        lr = LinearRegression()
        lr.fit(covariates, X_regress[:, i])
        predicted = lr.predict(covariates)
        X_regressed[:, i] = X_regress[:, i] - predicted

    print("  ✓ Regression complete")

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
    layer: str | None = "log_normalized",
    method: Literal["standard", "robust"] = "standard",
    output_layer: str = "batch_scaled",
    inplace: bool = True,
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
    from sklearn.preprocessing import RobustScaler, StandardScaler

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

        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        X_scaled[batch_mask, :] = scaler.fit_transform(X[batch_mask, :])

        print(f"    Scaled batch '{batch_id}': {n_cells_batch} cells")

    print("  ✓ Batch scaling complete")

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_scaled, overwrite=True)
        return None
    else:
        return X_scaled


def scvi_integrate(
    spatioloji_obj,
    batch_key: str,
    layer: str | None = "log_normalized",
    n_latent: int = 30,
    max_epochs: int | None = None,
    conda_env: str | None = None,
    use_highly_variable: bool = True,
    gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
    random_state: int = 42,
    output_key: str = "X_scvi",
    inplace: bool = True,
):
    """
    Batch correction and integration using scVI.

    Learns a batch-corrected latent representation with a variational
    autoencoder. Requires `scvi-tools` and `anndata`.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
    layer : str, optional
        Which layer to use for model training, by default 'log_normalized'
    n_latent : int, optional
        Number of latent dimensions, by default 30
    max_epochs : int, optional
        Maximum number of training epochs, by default None (auto)
    conda_env : str, optional
        Name of conda environment with scvi-tools installed, by default None
        If None, runs in current environment
        If specified, runs scVI in that environment via subprocess
    use_highly_variable : bool, optional
        Train on highly variable genes only when available, by default True
    gene_likelihood : {'zinb', 'nb', 'poisson', 'normal'}, optional
        Observation model for expression counts, by default 'zinb'
    random_state : int, optional
        Random seed, by default 42
    output_key : str, optional
        Key for storing latent embedding, by default 'X_scvi'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        scVI latent representation if inplace=False, otherwise None
    """
    print(f"\nscVI integration (batch_key='{batch_key}', n_latent={n_latent})")

    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    gene_names = spatioloji_obj.gene_index.astype(str).values

    if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
        if gene_mask.sum() > 0:
            X = X[:, gene_mask]
            gene_names = spatioloji_obj.gene_index[gene_mask].astype(str).values
            print(f"  Using {gene_mask.sum()} highly variable genes")

    batch = spatioloji_obj.cell_meta[batch_key].astype(str).values

    if conda_env is not None:
        print(f"  Running scVI in conda environment: '{conda_env}'")
        X_scvi = _run_scvi_in_conda_env(
            X=X,
            batch=batch,
            gene_names=gene_names,
            n_latent=n_latent,
            max_epochs=max_epochs,
            gene_likelihood=gene_likelihood,
            random_state=random_state,
            conda_env=conda_env,
        )
    else:
        try:
            import anndata
            import scvi
        except ImportError as err:
            raise ImportError(
                "scvi_integrate requires scvi-tools and anndata. "
                "Install with: pip install scvi-tools anndata"
            ) from err

        obs_df = spatioloji_obj.cell_meta.copy()
        var_df = pd.DataFrame(index=gene_names)
        adata = anndata.AnnData(X=X, obs=obs_df, var=var_df)

        scvi.settings.seed = random_state
        scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

        print("  Training scVI model...")
        model = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
        model.train(max_epochs=max_epochs)

        X_scvi = model.get_latent_representation()

    print("  ✓ scVI integration complete")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}

        spatioloji_obj._embeddings[output_key] = X_scvi

        for i in range(X_scvi.shape[1]):
            spatioloji_obj._cell_meta[f"SCVI{i+1}"] = X_scvi[:, i]

        return None
    else:
        return X_scvi


def _run_scvi_in_conda_env(X, batch, gene_names, n_latent, max_epochs, gene_likelihood, random_state, conda_env):
    """
    Helper function to run scVI integration in a separate conda environment.

    This creates temporary files, calls a Python script in the specified
    conda environment, and loads the latent embedding back.
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "scvi_input.npz")
        np.savez_compressed(
            input_file,
            X=X,
            batch=batch,
            gene_names=np.asarray(gene_names, dtype=str),
            n_latent=n_latent,
            max_epochs=-1 if max_epochs is None else int(max_epochs),
            gene_likelihood=str(gene_likelihood),
            random_state=int(random_state),
        )

        output_file = os.path.join(tmpdir, "scvi_output.npy")

        script_file = os.path.join(tmpdir, "run_scvi.py")
        script_content = f"""
import numpy as np
import pandas as pd
import anndata
import scvi

data = np.load('{input_file}', allow_pickle=True)
X = data['X']
batch = data['batch'].astype(str)
gene_names = data['gene_names'].astype(str)
n_latent = int(data['n_latent'])
max_epochs_raw = int(data['max_epochs'])
gene_likelihood = str(data['gene_likelihood'])
random_state = int(data['random_state'])

max_epochs = None if max_epochs_raw < 0 else max_epochs_raw

obs = pd.DataFrame({{'batch': batch}})
var = pd.DataFrame(index=gene_names)
adata = anndata.AnnData(X=X, obs=obs, var=var)

scvi.settings.seed = random_state
scvi.model.SCVI.setup_anndata(adata, batch_key='batch')

model = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
model.train(max_epochs=max_epochs)
latent = model.get_latent_representation()

np.save('{output_file}', latent)
print('scVI integration complete!')
"""

        with open(script_file, "w") as f:
            f.write(script_content)

        print(f"    Calling conda environment '{conda_env}'...")

        try:
            cmd = ["conda", "run", "-n", conda_env, "python", script_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e1:
            try:
                shell_script = os.path.join(tmpdir, "run_scvi.sh")
                with open(shell_script, "w") as f:
                    f.write(f"""#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}
python {script_file}
""")

                result = subprocess.run(["bash", shell_script], capture_output=True, text=True, check=True)
                print(result.stdout)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to run scVI in conda environment '{conda_env}'.\n"
                    f"Error 1 (conda run): {str(e1)}\n"
                    f"Error 2 (conda activate): {str(e2)}\n"
                    "Make sure the environment exists and has scvi-tools installed:\n"
                    f"  conda create -n {conda_env} python=3.10\n"
                    f"  conda activate {conda_env}\n"
                    "  pip install scvi-tools anndata"
                ) from e2

        if not os.path.exists(output_file):
            raise RuntimeError(
                "scVI integration failed - output file not created.\n"
                f"Check that scvi-tools is properly installed in '{conda_env}'"
            )

        return np.load(output_file)

def _run_seurat_integration_in_conda(
    X,
    batch,
    batches,
    gene_names,
    cell_names,
    method,
    n_dims,
    n_features,
    conda_env,
    n_cores=1,
):
    """
    Helper: run Seurat CCA or RPCA integration via R in a conda environment.

    Writes input data to temp files, executes an R script using Rscript
    inside the specified conda environment, and loads results back.

    Parameters
    ----------
    X : np.ndarray
        Log-normalized expression matrix (cells x genes)
    batch : np.ndarray
        Batch labels per cell
    batches : np.ndarray
        Unique batch identifiers
    gene_names : np.ndarray
        Gene names
    cell_names : np.ndarray
        Cell barcodes/names
    method : str
        'cca' or 'rpca'
    n_dims : int
        Number of dimensions for integration
    n_features : int
        Number of variable features per sample
    conda_env : str or None
        Conda environment name with R + Seurat installed

    Returns
    -------
    np.ndarray
        Corrected embedding (cells x n_dims)
    """
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:

        # --- Save input data as CSV per batch ---
        input_files = []
        for batch_id in batches:
            mask = batch == batch_id
            batch_X = X[mask, :]
            batch_cells = cell_names[mask]

            df = pd.DataFrame(
                batch_X,
                index=batch_cells,
                columns=gene_names
            )
            fpath = os.path.join(tmpdir, f"batch_{batch_id}.csv")
            df.to_csv(fpath)
            input_files.append((batch_id, fpath))

        output_file = os.path.join(tmpdir, "seurat_output.csv")

        # --- Build R script ---
        # Dynamically build the batch loading section
        r_load_lines = []
        r_obj_names = []
        for batch_id, fpath in input_files:
            safe_name = f"seurat_{str(batch_id).replace('-', '_').replace(' ', '_')}"
            r_obj_names.append(safe_name)
            r_load_lines.append(f"""
mat_{safe_name} <- read.csv('{fpath}', row.names=1, check.names=FALSE)
mat_{safe_name} <- t(as.matrix(mat_{safe_name}))  # genes x cells
{safe_name} <- CreateSeuratObject(counts=mat_{safe_name})
{safe_name} <- NormalizeData({safe_name}, verbose=FALSE)
{safe_name} <- FindVariableFeatures({safe_name}, nfeatures={n_features}, verbose=FALSE)
""")

        r_obj_list = "list(" + ", ".join(r_obj_names) + ")"
        reduction_method = '"cca"' if method == 'cca' else '"rpca"'

        # For RPCA, need to run PCA on each object first
        rpca_pca_lines = ""
        if method == "rpca":
            rpca_pca_lines = f"""
# RPCA requires PCA on each object first
features <- SelectIntegrationFeatures(seurat_list, nfeatures={n_features})
seurat_list <- lapply(seurat_list, function(x) {{
  x <- ScaleData(x, features=features, verbose=FALSE)
  x <- RunPCA(x, features=features, verbose=FALSE)
  x
}})
"""

        r_script = f"""
suppressPackageStartupMessages({{
  library(Seurat)
  library(dplyr)
  library(future)
}})

# Enable multi-threading
plan("multicore", workers={n_cores})
options(future.globals.maxSize = 8 * 1024^3)  # 8 GB per worker

cat("Using {n_cores} core(s) for integration\\n")

cat("Loading batch data...\\n")

# --- Load each batch as a Seurat object ---
{"".join(r_load_lines)}

# --- Create list of Seurat objects ---
seurat_list <- {r_obj_list}

cat("Finding integration features...\\n")
features <- SelectIntegrationFeatures(seurat_list, nfeatures={n_features})

{rpca_pca_lines}

cat("Finding integration anchors ({method.upper()})...\\n")
anchors <- FindIntegrationAnchors(
  object.list = seurat_list,
  anchor.features = features,
  reduction = {reduction_method},
  dims = 1:{n_dims},
  verbose = FALSE
)

cat("Integrating data...\\n")
integrated <- IntegrateData(
  anchorset = anchors,
  dims = 1:{n_dims},
  verbose = FALSE
)

cat("Running PCA on integrated assay...\\n")
DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose=FALSE)
integrated <- RunPCA(integrated, npcs={n_dims}, verbose=FALSE)

# --- Extract corrected PCA embedding ---
pca_embed <- Embeddings(integrated, "pca")

cat("Saving output...\\n")
write.csv(pca_embed, "{output_file}", quote=FALSE)
cat("Done!\\n")
"""

        script_file = os.path.join(tmpdir, "run_seurat.R")
        with open(script_file, "w") as f:
            f.write(r_script)

        # --- Execute R script ---
        def run_cmd(cmd):
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

        print(f"    Executing R script ({method.upper()})...")

        if conda_env is not None:
            try:
                # Method 1: conda run
                result = run_cmd(
                    ["conda", "run", "-n", conda_env, "Rscript", script_file]
                )
                print(result.stdout)
            except (subprocess.CalledProcessError, FileNotFoundError) as e1:
                try:
                    # Method 2: shell + conda activate
                    shell_script = os.path.join(tmpdir, "run_seurat.sh")
                    with open(shell_script, "w") as f:
                        f.write(f"""#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {conda_env}
Rscript {script_file}
""")
                    result = run_cmd(["bash", shell_script])
                    print(result.stdout)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to run Seurat in conda environment '{conda_env}'.\n"
                        f"Error 1 (conda run): {str(e1)}\n"
                        f"Error 2 (conda activate): {str(e2)}\n"
                        f"Make sure R + Seurat are installed in '{conda_env}':\n"
                        f"  conda create -n {conda_env} -c conda-forge r-base r-seurat"
                    ) from e2
        else:
            try:
                # Run Rscript directly from current environment
                result = run_cmd(["Rscript", script_file])
                print(result.stdout)
            except FileNotFoundError as err:
                raise RuntimeError(
                    "Rscript not found. Install R or specify conda_env with R + Seurat:\n"
                    "  conda create -n seurat_env -c conda-forge r-base r-seurat"
                ) from err

        # --- Load output ---
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"Seurat {method.upper()} failed — output file not created.\n"
                "Check that Seurat is properly installed and R script ran without errors."
            )

        result_df = pd.read_csv(output_file, index_col=0)
        return result_df.values


def cca_integrate(
    spatioloji_obj,
    batch_key: str,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = True,
    n_dims: int = 30,
    n_features: int = 2000,
    conda_env: str | None = None,
    n_cores: int = 1,
    output_key: str = "X_cca",
    inplace: bool = True,
):
    """
    Seurat CCA (Canonical Correlation Analysis) batch integration via R.

    Identifies shared correlation structure across batches using CCA,
    finds MNN anchors in this shared space, and integrates data.
    This is Seurat's classic integration method (v3+).

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
        (e.g., 'sample_id', 'batch')
    layer : str, optional
        Which layer to use, by default 'log_normalized'
        Must be log-normalized — CCA assumes additive batch model in log space
    use_highly_variable : bool, optional
        Use only HVGs for integration, by default True
    n_dims : int, optional
        Number of CCA dimensions and PCs to compute, by default 30
    n_features : int, optional
        Number of variable features per batch, by default 2000
    conda_env : str, optional
        Conda environment with R + Seurat installed, by default None
        If None, uses Rscript from current PATH
        If specified, runs R in that environment via subprocess
    output_key : str, optional
        Key for storing corrected embedding, by default 'X_cca'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        CCA-corrected PCA embedding (n_cells x n_dims) if inplace=False,
        otherwise None

    Notes
    -----
    - Input layer must be log-normalized (NOT scaled, NOT raw counts)
    - CCA is most powerful for strong batch effects across diverse datasets
    - Output is a corrected PCA embedding — NOT corrected expression values
    - Do NOT use for pseudobulk DE — always use raw counts with batch covariate
    - For large datasets (>200k cells), consider RPCA or Harmony instead
    - Requires R + Seurat installed (either in PATH or in conda_env)

    Examples
    --------
    >>> # Basic CCA integration
    >>> sp.processing.cca_integrate(sp, batch_key='sample_id')
    >>>
    >>> # Use specific conda environment
    >>> sp.processing.cca_integrate(
    ...     sp,
    ...     batch_key='sample_id',
    ...     conda_env='seurat_env',
    ...     n_dims=30
    ... )
    >>>
    >>> # Then use for clustering
    >>> sp.processing.umap(sp, use_rep='X_cca')

    References
    ----------
    Stuart et al. (2019) Cell — Seurat v3
    """
    print(f"\nSeurat CCA integration (batch_key='{batch_key}')")

    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    batch = spatioloji_obj.cell_meta[batch_key].astype(str).values
    batches = np.unique(batch)
    n_batches = len(batches)

    if n_batches < 2:
        raise ValueError("Need at least 2 batches for CCA integration")

    print(f"  Found {n_batches} batches")

    # Get expression data — must be log-normalized
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    gene_names = spatioloji_obj.gene_index.astype(str).values
    cell_names = spatioloji_obj.cell_index.astype(str).values

    # Subset to HVGs if available
    if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
        if gene_mask.sum() > 0:
            X = X[:, gene_mask]
            gene_names = gene_names[gene_mask]
            print(f"  Using {gene_mask.sum()} highly variable genes")

    # Run CCA via R/Seurat
    X_corrected = _run_seurat_integration_in_conda(
        X=X,
        batch=batch,
        batches=batches,
        gene_names=gene_names,
        cell_names=cell_names,
        method="cca",
        n_dims=n_dims,
        n_features=n_features,
        conda_env=conda_env,
        n_cores=n_cores,
    )

    print("  ✓ CCA integration complete")
    print(f"    Output shape: {X_corrected.shape}")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}

        spatioloji_obj._embeddings[output_key] = X_corrected

        for i in range(X_corrected.shape[1]):
            spatioloji_obj._cell_meta[f"CCA{i+1}"] = X_corrected[:, i]

        return None
    else:
        return X_corrected


def rpca_integrate(
    spatioloji_obj,
    batch_key: str,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = True,
    n_dims: int = 30,
    n_features: int = 2000,
    conda_env: str | None = None,
    n_cores=1,
    output_key: str = "X_rpca",
    inplace: bool = True,
):
    """
    Seurat RPCA (Reciprocal PCA) batch integration via R.

    Faster alternative to CCA integration. Projects each dataset into
    the PCA space of the other, finds MNN anchors in this reciprocal
    space, and integrates. Better suited for large datasets.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    batch_key : str
        Column name in cell_meta that defines batches
        (e.g., 'sample_id', 'batch')
    layer : str, optional
        Which layer to use, by default 'log_normalized'
        Must be log-normalized — RPCA assumes additive batch model in log space
    use_highly_variable : bool, optional
        Use only HVGs for integration, by default True
    n_dims : int, optional
        Number of PCA dimensions to compute per batch, by default 30
    n_features : int, optional
        Number of variable features per batch, by default 2000
    conda_env : str, optional
        Conda environment with R + Seurat installed, by default None
        If None, uses Rscript from current PATH
        If specified, runs R in that environment via subprocess
    output_key : str, optional
        Key for storing corrected embedding, by default 'X_rpca'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        RPCA-corrected PCA embedding (n_cells x n_dims) if inplace=False,
        otherwise None

    Notes
    -----
    - Input layer must be log-normalized (NOT scaled, NOT raw counts)
    - RPCA is faster and less prone to overcorrection than CCA
    - RPCA runs PCA on each batch separately before anchor finding
    - Output is a corrected PCA embedding — NOT corrected expression values
    - Do NOT use for pseudobulk DE — always use raw counts with batch covariate
    - Preferred over CCA for large datasets (>100k cells) or many batches
    - Requires R + Seurat installed (either in PATH or in conda_env)

    Examples
    --------
    >>> # Basic RPCA integration
    >>> sp.processing.rpca_integrate(sp, batch_key='sample_id')
    >>>
    >>> # Use specific conda environment
    >>> sp.processing.rpca_integrate(
    ...     sp,
    ...     batch_key='sample_id',
    ...     conda_env='seurat_env',
    ...     n_dims=30,
    ...     n_features=3000
    ... )
    >>>
    >>> # Then use for clustering
    >>> sp.processing.umap(sp, use_rep='X_rpca')

    References
    ----------
    Hao et al. (2021) Cell — Seurat v4
    """
    print(f"\nSeurat RPCA integration (batch_key='{batch_key}')")

    if batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    batch = spatioloji_obj.cell_meta[batch_key].astype(str).values
    batches = np.unique(batch)
    n_batches = len(batches)

    if n_batches < 2:
        raise ValueError("Need at least 2 batches for RPCA integration")

    print(f"  Found {n_batches} batches")

    # Get expression data — must be log-normalized
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    gene_names = spatioloji_obj.gene_index.astype(str).values
    cell_names = spatioloji_obj.cell_index.astype(str).values

    # Subset to HVGs if available
    if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
        if gene_mask.sum() > 0:
            X = X[:, gene_mask]
            gene_names = gene_names[gene_mask]
            print(f"  Using {gene_mask.sum()} highly variable genes")

    # Run RPCA via R/Seurat
    X_corrected = _run_seurat_integration_in_conda(
        X=X,
        batch=batch,
        batches=batches,
        gene_names=gene_names,
        cell_names=cell_names,
        method="rpca",
        n_dims=n_dims,
        n_features=n_features,
        conda_env=conda_env,
        n_cores=n_cores,
    )

    print("✓ RPCA integration complete")
    print(f"Output shape: {X_corrected.shape}")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}

        spatioloji_obj._embeddings[output_key] = X_corrected

        for i in range(X_corrected.shape[1]):
            spatioloji_obj._cell_meta[f"RPCA{i+1}"] = X_corrected[:, i]

        return None
    else:
        return X_corrected


def evaluate_batch_correction(
    spatioloji_obj, batch_key: str, embedding_key: str = "X_pca", n_neighbors: int = 30
) -> dict[str, float]:
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
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    print("\nEvaluating batch correction...")

    # Get embedding
    if not hasattr(spatioloji_obj, "_embeddings") or embedding_key not in spatioloji_obj._embeddings:
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

    return {"batch_mixing": batch_mixing, "silhouette_batch": sil_batch}
