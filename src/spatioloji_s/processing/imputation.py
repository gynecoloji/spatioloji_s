"""
imputation.py - Imputation methods for spatial transcriptomics

Provides methods for imputing missing values and recovering dropout events
in sparse single-cell expression data.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def magic_impute(
    spatioloji_obj,
    layer: str | None = 'log_normalized',
    use_highly_variable: bool = False,
    n_pca_components: int = 100,
    knn: int = 5,
    t: int = 3,
    genes: str | list[str] | None = None,
    conda_env: str | None = None,
    output_layer: str = 'magic_imputed',
    inplace: bool = True
):
    """
    MAGIC (Markov Affinity-based Graph Imputation of Cells) imputation.

    Imputes missing values by diffusing information across similar cells
    in a k-NN graph. Particularly good at denoising and revealing continuous
    relationships. Can run in a separate conda environment if magic-impute
    has dependency conflicts.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to impute, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only HVGs for graph construction, by default False
    n_pca_components : int, optional
        Number of PCs for graph construction, by default 100
    knn : int, optional
        Number of nearest neighbors, by default 5
    t : int, optional
        Number of diffusion steps, by default 3
        Higher values = more smoothing
    genes : str or list of str, optional
        Specific genes to impute. If None, imputes all genes
    conda_env : str, optional
        Name of conda environment with magic-impute installed, by default None
        If None, tries to import magic from current environment
        If specified, runs MAGIC in that environment via subprocess
    output_layer : str, optional
        Name for imputed layer, by default 'magic_imputed'
    inplace : bool, optional
        Add imputed layer to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Imputed expression if inplace=False, otherwise None

    Examples
    --------
    >>> # Try current environment first
    >>> sp.processing.magic_impute(sp, knn=5, t=3)
    >>>
    >>> # Use separate conda environment (recommended if magic-impute conflicts)
    >>> # First create environment: conda create -n magic_env python=3.8 magic-impute -c bioconda
    >>> sp.processing.magic_impute(
    ...     sp,
    ...     conda_env='magic_env',
    ...     knn=5,
    ...     t=3
    ... )
    >>>
    >>> # Impute specific genes only
    >>> sp.processing.magic_impute(
    ...     sp,
    ...     genes=['CD3D', 'CD8A', 'CD4'],
    ...     conda_env='magic_env',
    ...     t=3
    ... )

    References
    ----------
    van Dijk et al. (2018) Cell

    Notes
    -----
    If using conda_env, make sure the environment has magic-impute installed:
        conda create -n magic_env python=3.8
        conda activate magic_env
        pip install magic-impute
        # or: conda install -c bioconda magic-impute
    """

    print(f"\nMAGIC imputation (knn={knn}, t={t})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Prepare gene subset
    gene_indices = None
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        gene_indices = spatioloji_obj._get_gene_indices(genes)
        print(f"  Imputing {len(genes)} specific genes")

    # If conda_env is specified, run in separate environment
    if conda_env is not None:
        print(f"  Running MAGIC in conda environment: '{conda_env}'")
        X_imputed = _run_magic_in_conda_env(
            X, n_pca_components, knn, t, gene_indices, conda_env
        )
    else:
        # Try to import magic from current environment
        try:
            import magic
        except ImportError as err:
            raise ImportError(
                "magic-impute not found in current environment. Either:\n"
                "  1. Install magic-impute: pip install magic-impute\n"
                "  2. Or specify conda_env with magic-impute installed:\n"
                "     sp.processing.magic_impute(sp, conda_env='magic_env')"
            ) from err

        # Create MAGIC operator
        print(f"  Building k-NN graph (n_pca={n_pca_components})...")
        magic_op = magic.MAGIC(
            n_pca=n_pca_components,
            knn=knn,
            t=t,
            verbose=0
        )

        # Fit and transform
        print("  Running MAGIC diffusion...")
        X_imputed = magic_op.fit_transform(X, genes=gene_indices)

        # Convert back to numpy if needed
        if isinstance(X_imputed, pd.DataFrame):
            X_imputed = X_imputed.values

    print("  ✓ MAGIC imputation complete")

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_imputed, overwrite=True)
        return None
    else:
        return X_imputed


def _run_magic_in_conda_env(X, n_pca_components, knn, t, gene_indices, conda_env):
    """
    Helper function to run MAGIC imputation in a separate conda environment.

    This creates temporary files, calls a Python script in the specified
    conda environment, and loads the results back.
    """
    import os
    import subprocess
    import tempfile

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save input data
        input_file = os.path.join(tmpdir, 'magic_input.npz')
        np.savez_compressed(
            input_file,
            X=X,
            n_pca_components=n_pca_components,
            knn=knn,
            t=t,
            gene_indices=gene_indices if gene_indices is not None else np.array([])
        )

        output_file = os.path.join(tmpdir, 'magic_output.npy')

        # Create Python script for MAGIC
        script_file = os.path.join(tmpdir, 'run_magic.py')
        script_content = f"""
import numpy as np
import magic

# Load input
data = np.load('{input_file}', allow_pickle=True)
X = data['X']
n_pca_components = int(data['n_pca_components'])
knn = int(data['knn'])
t = int(data['t'])
gene_indices = data['gene_indices']

# Handle gene_indices
if len(gene_indices) == 0:
    gene_indices = None

print(f"Input shape: {{X.shape}}")
print(f"n_pca: {{n_pca_components}}, knn: {{knn}}, t: {{t}}")

# Create MAGIC operator
print("Building k-NN graph...")
magic_op = magic.MAGIC(
    n_pca=n_pca_components,
    knn=knn,
    t=t,
    verbose=1
)

# Fit and transform
print("Running MAGIC diffusion...")
X_imputed = magic_op.fit_transform(X, genes=gene_indices)

# Convert to numpy if needed
if hasattr(X_imputed, 'values'):
    X_imputed = X_imputed.values

# Save output
np.save('{output_file}', X_imputed)
print("MAGIC imputation complete!")
print(f"Output shape: {{X_imputed.shape}}")
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
                shell_script = os.path.join(tmpdir, 'run_magic.sh')
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
                    f"Failed to run MAGIC in conda environment '{conda_env}'.\n"
                    f"Error 1 (conda run): {str(e1)}\n"
                    f"Error 2 (conda activate): {str(e2)}\n"
                    f"Make sure the environment exists and has magic-impute installed:\n"
                    f"  conda create -n {conda_env} python=3.8\n"
                    f"  conda activate {conda_env}\n"
                    f"  pip install magic-impute"
                ) from e2

        # Load results
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"MAGIC imputation failed - output file not created.\n"
                f"Check that magic-impute is properly installed in '{conda_env}'"
            )

        X_imputed = np.load(output_file)

        return X_imputed

def knn_smooth(
    spatioloji_obj,
    layer: str | None = 'log_normalized',
    use_highly_variable: bool = False,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_pca: bool = True,
    output_layer: str = 'knn_smoothed',
    inplace: bool = True
):
    """
    k-NN smoothing imputation.

    Simple and fast imputation that replaces each cell's expression with
    the weighted average of its k nearest neighbors. No special packages
    required.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to impute, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only HVGs for finding neighbors, by default False
    n_neighbors : int, optional
        Number of nearest neighbors, by default 15
    n_pcs : int, optional
        Number of PCs for finding neighbors, by default 50
    use_pca : bool, optional
        Use PCA space for finding neighbors, by default True
    output_layer : str, optional
        Name for smoothed layer, by default 'knn_smoothed'
    inplace : bool, optional
        Add smoothed layer to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Smoothed expression if inplace=False, otherwise None

    Examples
    --------
    >>> # Basic k-NN smoothing
    >>> sp.processing.knn_smooth(sp, n_neighbors=15)
    >>>
    >>> # Use more neighbors for stronger smoothing
    >>> sp.processing.knn_smooth(sp, n_neighbors=30)
    """
    print(f"\nk-NN smoothing (n_neighbors={n_neighbors})")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Get data for neighbor finding
    if use_pca:
        # Use PCA coordinates
        if hasattr(spatioloji_obj, '_embeddings') and 'X_pca' in spatioloji_obj._embeddings:
            X_neighbors = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs for neighbor finding")
        else:
            print("  Computing PCA for neighbor finding...")
            pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]))
            X_neighbors = pca.fit_transform(X)
    else:
        X_neighbors = X

        # Subset to HVGs if requested
        if use_highly_variable and 'highly_variable' in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
            if gene_mask.sum() > 0:
                X_neighbors = X[:, gene_mask]
                print(f"  Using {gene_mask.sum()} HVGs for neighbor finding")

    # Build k-NN graph
    print("  Building k-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_neighbors)
    distances, indices = nbrs.kneighbors(X_neighbors)

    # Convert distances to weights (Gaussian kernel)
    sigma = np.median(distances[:, -1])  # Adaptive bandwidth
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize

    # Smooth expression
    print("  Smoothing expression...")
    X_smoothed = np.zeros_like(X)

    for i in range(X.shape[0]):
        neighbor_idx = indices[i]
        neighbor_weights = weights[i]
        X_smoothed[i] = (X[neighbor_idx].T @ neighbor_weights).T

    print("  ✓ k-NN smoothing complete")

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_smoothed, overwrite=True)
        return None
    else:
        return X_smoothed


def alra_impute(
    spatioloji_obj,
    layer: str | None = None,
    use_highly_variable: bool = False,
    k: int | None = None,
    q: int = 10,
    quantile_prob: float = 0.001,
    output_layer: str = 'alra_imputed',
    inplace: bool = True
):
    """
    ALRA (Adaptively-thresholded Low Rank Approximation) imputation.

    Uses low-rank matrix approximation to recover dropout events. Adaptively
    chooses rank and thresholds values. Works well on log-normalized data.
    No special packages required (pure numpy/scipy).

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to impute. If None, uses main expression matrix.
        Works best on log-normalized data.
    use_highly_variable : bool, optional
        Impute only HVGs, by default False
    k : int, optional
        Rank for low-rank approximation. If None, automatically determined
    q : int, optional
        Number of additional PCs for rank selection, by default 10
    quantile_prob : float, optional
        Quantile for thresholding, by default 0.001
    output_layer : str, optional
        Name for imputed layer, by default 'alra_imputed'
    inplace : bool, optional
        Add imputed layer to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Imputed expression if inplace=False, otherwise None

    Examples
    --------
    >>> # ALRA imputation with automatic rank selection
    >>> sp.processing.alra_impute(sp, layer='log_normalized')
    >>>
    >>> # Manual rank specification
    >>> sp.processing.alra_impute(sp, k=20)

    References
    ----------
    Linderman et al. (2018) bioRxiv
    """
    from scipy.sparse.linalg import svds

    print("\nALRA imputation")

    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Subset to HVGs if requested
    gene_mask = None
    if use_highly_variable and 'highly_variable' in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
        if gene_mask.sum() > 0:
            X_impute = X[:, gene_mask]
            print(f"  Imputing {gene_mask.sum()} highly variable genes")
        else:
            X_impute = X
    else:
        X_impute = X

    # Center the data
    X_center = X_impute - X_impute.mean(axis=0)

    # Determine rank if not provided
    if k is None:
        print("  Determining optimal rank...")

        # Compute SVD for rank selection
        max_k = min(100, X_impute.shape[0] - 1, X_impute.shape[1] - 1)

        try:
            U, s, Vt = svds(X_center, k=max_k)
            # Sort by singular values (descending)
            idx = np.argsort(-s)
            s = s[idx]

            # Find elbow point
            # Use second derivative to find change in curvature
            if len(s) > 10:
                diffs = np.diff(s)
                diffs2 = np.diff(diffs)
                k = np.argmax(diffs2) + 2  # +2 for offset from double diff
                k = min(k, max_k - q)
            else:
                k = max(5, len(s) // 2)

            print(f"    Selected rank: {k}")

        except Exception:
            print("    SVD failed, using default rank: 20")
            k = 20

    k_actual = min(k, X_impute.shape[0] - 1, X_impute.shape[1] - 1)

    # Compute low-rank approximation
    print(f"  Computing low-rank approximation (k={k_actual})...")

    try:
        U, s, Vt = svds(X_center, k=k_actual)

        # Sort by singular values (descending)
        idx = np.argsort(-s)
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]

        # Reconstruct
        X_lr = U @ np.diag(s) @ Vt
        X_lr = X_lr + X_impute.mean(axis=0)  # Add back mean

    except Exception as e:
        print(f"    Error in SVD: {e}")
        print("    Falling back to PCA-based approximation...")

        pca = PCA(n_components=k_actual)
        X_pca = pca.fit_transform(X_center)
        X_lr = pca.inverse_transform(X_pca) + X_impute.mean(axis=0)

    # Adaptive thresholding
    print(f"  Applying adaptive thresholding (quantile={quantile_prob})...")

    # For each gene, threshold at quantile
    X_imputed = X_impute.copy()

    for j in range(X_impute.shape[1]):
        # Get non-zero values
        nonzero_vals = X_impute[:, j][X_impute[:, j] > 0]

        if len(nonzero_vals) > 10:
            # Calculate threshold
            threshold = np.quantile(nonzero_vals, quantile_prob)

            # Where original was zero and imputed > threshold, use imputed
            zero_mask = X_impute[:, j] == 0
            impute_mask = zero_mask & (X_lr[:, j] > threshold)

            X_imputed[impute_mask, j] = X_lr[impute_mask, j]

    # Count imputed values
    n_imputed = (X_impute == 0).sum() - (X_imputed == 0).sum()
    pct_imputed = n_imputed / (X_impute == 0).sum() * 100

    print("  ✓ ALRA imputation complete")
    print(f"    Imputed {n_imputed:,} values ({pct_imputed:.1f}% of zeros)")

    # If we only imputed HVGs, merge back
    if gene_mask is not None:
        X_full = X.copy()
        X_full[:, gene_mask] = X_imputed
        X_imputed = X_full

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_imputed, overwrite=True)
        return None
    else:
        return X_imputed


def dca_impute(
    spatioloji_obj,
    layer: str | None = None,
    use_highly_variable: bool = False,
    epochs: int = 300,
    hidden_size: tuple[int, int, int] = (64, 32, 64),
    output_layer: str = 'dca_imputed',
    inplace: bool = True
):
    """
    DCA (Deep Count Autoencoder) imputation.

    Uses a deep autoencoder to denoise and impute expression data.
    Works on raw counts. Requires `dca` package.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to impute (should be raw counts)
    use_highly_variable : bool, optional
        Use only HVGs, by default False
    epochs : int, optional
        Number of training epochs, by default 300
    hidden_size : tuple, optional
        Hidden layer dimensions, by default (64, 32, 64)
    output_layer : str, optional
        Name for imputed layer, by default 'dca_imputed'
    inplace : bool, optional
        Add imputed layer to spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Imputed expression if inplace=False, otherwise None

    Examples
    --------
    >>> # DCA imputation on raw counts
    >>> sp.processing.dca_impute(sp, layer=None, epochs=300)

    References
    ----------
    Eraslan et al. (2019) Nature Communications

    Notes
    -----
    DCA requires TensorFlow and can be slow. Consider using simpler methods
    like ALRA or k-NN smoothing for faster results.
    """
    try:
        from dca.api import dca
    except ImportError as err:
        raise ImportError(
            "DCA requires dca package. "
            "Install with: pip install dca\n"
            "Note: Also requires TensorFlow"
        ) from err

    print(f"\nDCA imputation (epochs={epochs})")
    print("  Warning: This may take several minutes...")

    # Get expression data (should be raw counts)
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    # Subset to HVGs if requested
    gene_mask = None
    if use_highly_variable and 'highly_variable' in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
        if gene_mask.sum() > 0:
            X_impute = X[:, gene_mask]
            print(f"  Imputing {gene_mask.sum()} highly variable genes")
        else:
            X_impute = X
    else:
        X_impute = X

    # Convert to AnnData for DCA
    import anndata
    adata_temp = anndata.AnnData(X=X_impute)

    # Run DCA
    print("  Running DCA...")
    dca(
        adata_temp,
        epochs=epochs,
        hidden_size=hidden_size,
        verbose=True
    )

    X_imputed = adata_temp.X
    if sparse.issparse(X_imputed):
        X_imputed = X_imputed.toarray()

    print("  ✓ DCA imputation complete")

    # If we only imputed HVGs, merge back
    if gene_mask is not None:
        X_full = X.copy()
        X_full[:, gene_mask] = X_imputed
        X_imputed = X_full

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_imputed, overwrite=True)
        return None
    else:
        return X_imputed


def compare_imputation_methods(
    spatioloji_obj,
    layer: str = 'log_normalized',
    test_genes: list[str] | None = None,
    n_test_genes: int = 10,
    methods: list[str] | None = None,
    magic_conda_env: str | None = None,  # NEW PARAMETER
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare different imputation methods.

    Randomly masks values and compares how well different methods recover them.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to test, by default 'log_normalized'
    test_genes : list of str, optional
        Specific genes to test. If None, randomly selects genes
    n_test_genes : int, optional
        Number of genes to test if test_genes is None, by default 10
    methods : list of str, optional
        Methods to compare, by default ['knn_smooth', 'alra']
        Options: 'knn_smooth', 'alra', 'magic'
    magic_conda_env : str, optional
        Conda environment for MAGIC (if 'magic' in methods), by default None
    random_state : int, optional
        Random seed, by default 42

    Returns
    -------
    pd.DataFrame
        Comparison results with correlation and MSE for each method

    Examples
    --------
    >>> # Compare k-NN and ALRA (no dependencies)
    >>> results = sp.processing.compare_imputation_methods(
    ...     sp,
    ...     methods=['knn_smooth', 'alra'],
    ...     n_test_genes=20
    ... )
    >>>
    >>> # Include MAGIC using separate environment
    >>> results = sp.processing.compare_imputation_methods(
    ...     sp,
    ...     methods=['knn_smooth', 'alra', 'magic'],
    ...     magic_conda_env='magic_env',
    ...     n_test_genes=20
    ... )
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    if methods is None:
      methods = ['knn_smooth', 'alra']

    print(f"\nComparing imputation methods: {methods}")
    print(f"  Testing on {n_test_genes} genes")

    np.random.seed(random_state)

    # Get expression data
    X = spatioloji_obj.get_layer(layer)
    if sparse.issparse(X):
        X = X.toarray()

    # Select test genes
    if test_genes is None:
        # Select genes with moderate sparsity (not too sparse, not too dense)
        sparsity = (X == 0).mean(axis=0)
        moderate_sparse = (sparsity > 0.2) & (sparsity < 0.8)

        if moderate_sparse.sum() < n_test_genes:
            test_gene_indices = np.random.choice(X.shape[1], n_test_genes, replace=False)
        else:
            candidates = np.where(moderate_sparse)[0]
            test_gene_indices = np.random.choice(candidates, n_test_genes, replace=False)

        test_genes = spatioloji_obj.gene_index[test_gene_indices].tolist()
    else:
        test_gene_indices = spatioloji_obj._get_gene_indices(test_genes)

    # For each gene, mask 10% of non-zero values
    X_masked = X.copy()
    mask_positions = []
    true_values = []

    for gene_idx in test_gene_indices:
        nonzero_idx = np.where(X[:, gene_idx] > 0)[0]

        if len(nonzero_idx) > 10:
            n_mask = max(1, len(nonzero_idx) // 10)
            mask_idx = np.random.choice(nonzero_idx, n_mask, replace=False)

            mask_positions.extend([(i, gene_idx) for i in mask_idx])
            true_values.extend(X[mask_idx, gene_idx])

            X_masked[mask_idx, gene_idx] = 0

    print(f"  Masked {len(true_values)} values")

    # Test each method
    results = []

    for method in methods:
        print(f"\n  Testing: {method}")

        # Create temporary spatioloji with masked data
        import copy
        sp_temp = copy.deepcopy(spatioloji_obj)
        sp_temp.add_layer('test_masked', X_masked, overwrite=True)

        # Run imputation
        try:
            if method == 'knn_smooth':
                knn_smooth(sp_temp, layer='test_masked', output_layer='imputed', inplace=True)
            elif method == 'alra':
                alra_impute(sp_temp, layer='test_masked', output_layer='imputed', inplace=True)
            elif method == 'magic':
                magic_impute(
                    sp_temp,
                    layer='test_masked',
                    conda_env=magic_conda_env,  # Use specified environment
                    output_layer='imputed',
                    inplace=True
                )
            else:
                print(f"    Unknown method: {method}")
                continue

            X_imputed = sp_temp.get_layer('imputed')

            # Extract imputed values at masked positions
            imputed_values = [X_imputed[i, j] for i, j in mask_positions]

            # Calculate metrics
            correlation, _ = pearsonr(true_values, imputed_values)
            mse = mean_squared_error(true_values, imputed_values)
            rmse = np.sqrt(mse)

            results.append({
                'method': method,
                'correlation': correlation,
                'mse': mse,
                'rmse': rmse
            })

            print(f"    Correlation: {correlation:.3f}")
            print(f"    RMSE: {rmse:.3f}")

        except Exception as e:
            print(f"    Error: {e}")

    results_df = pd.DataFrame(results)

    print("\n  ✓ Comparison complete")
    print("\n  Results:")
    print(results_df.to_string(index=False))

    return results_df
