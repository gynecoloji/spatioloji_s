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
    layer: str | None = "log_normalized",
    use_highly_variable: bool = False,
    n_pca_components: int = 100,
    knn: int = 5,
    t: int = 3,
    genes: str | list[str] | None = None,
    conda_env: str | None = None,
    output_layer: str = "magic_imputed",
    inplace: bool = True,
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
        X_imputed = _run_magic_in_conda_env(X, n_pca_components, knn, t, gene_indices, conda_env)
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
        magic_op = magic.MAGIC(n_pca=n_pca_components, knn=knn, t=t, verbose=0)

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
        input_file = os.path.join(tmpdir, "magic_input.npz")
        np.savez_compressed(
            input_file,
            X=X,
            n_pca_components=n_pca_components,
            knn=knn,
            t=t,
            gene_indices=gene_indices if gene_indices is not None else np.array([]),
        )

        output_file = os.path.join(tmpdir, "magic_output.npy")

        # Create Python script for MAGIC
        script_file = os.path.join(tmpdir, "run_magic.py")
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

        with open(script_file, "w") as f:
            f.write(script_content)

        # Run script in conda environment
        print(f"    Calling conda environment '{conda_env}'...")

        # Try different conda activation methods
        try:
            # Method 1: Using conda run (newer conda versions)
            cmd = ["conda", "run", "-n", conda_env, "python", script_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

        except (subprocess.CalledProcessError, FileNotFoundError) as e1:
            try:
                # Method 2: Using shell script with source activate
                shell_script = os.path.join(tmpdir, "run_magic.sh")
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


def scvi_impute(
    spatioloji_obj,
    layer: str | None = "log_normalized",
    batch_key: str | None = None,
    use_highly_variable: bool = True,   # recommended: True for speed
    n_latent: int = 30,
    max_epochs: int | None = None,      # None = auto heuristic
    early_stopping: bool = True,        # NEW: stop when loss plateaus
    early_stopping_patience: int = 20,  # NEW: epochs to wait
    batch_size: int = 512,              # NEW: increased from default 128
    accelerator: str = "auto",          # NEW: "auto", "gpu", "cpu", "mps"
    gene_likelihood: str = "zinb",
    random_state: int = 42,
    conda_env: str | None = None,
    output_layer: str = "scvi_imputed",
    inplace: bool = True,
):
    print(f"\nscVI imputation (n_latent={n_latent})")

    if batch_key is not None and batch_key not in spatioloji_obj.cell_meta.columns:
        raise ValueError(f"Column '{batch_key}' not found in cell_meta")

    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    X_base     = X
    gene_names = spatioloji_obj.gene_index.astype(str).values
    gene_mask  = None

    if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
        if gene_mask.sum() > 0:
            X          = X[:, gene_mask]
            gene_names = spatioloji_obj.gene_index[gene_mask].astype(str).values
            print(f"  Using {gene_mask.sum()} highly variable genes (faster)")
        else:
            gene_mask = None

    batch = None
    if batch_key is not None:
        batch = spatioloji_obj.cell_meta[batch_key].astype(str).values

    if conda_env is not None:
        print(f"  Running scVI in conda environment: '{conda_env}'")
        X_imputed = _run_scvi_impute_in_conda_env(
            X=X, batch=batch,
            cell_ids=spatioloji_obj.cell_index.astype(str).values,
            gene_names=gene_names,
            n_latent=n_latent,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            batch_size=batch_size,
            accelerator=accelerator,
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
                "scvi-tools not found. Install: pip install scvi-tools anndata\n"
                "Or specify conda_env: scvi_impute(sp, conda_env='scvi_env')"
            ) from err

        obs   = pd.DataFrame(index=spatioloji_obj.cell_index.astype(str))
        if batch is not None:
            obs["batch"] = batch
        var   = pd.DataFrame(index=gene_names)
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        scvi.settings.seed = random_state
        scvi.model.SCVI.setup_anndata(adata, batch_key="batch" if batch is not None else None)

        model = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)

        # --- detect and report accelerator ---
        import torch
        if accelerator == "auto":
            if torch.cuda.is_available():
                print("  Accelerator: GPU (CUDA) ✓")
            elif torch.backends.mps.is_available():
                print("  Accelerator: GPU (Apple MPS) ✓")
            else:
                print("  Accelerator: CPU (no GPU found — training will be slow)")
        else:
            print(f"  Accelerator: {accelerator}")

        print(f"  batch_size={batch_size}, early_stopping={early_stopping}")
        print("  Training scVI model...")

        model.train(
            max_epochs              = max_epochs,
            accelerator             = accelerator,
            batch_size              = batch_size,
            early_stopping          = early_stopping,
            early_stopping_patience = early_stopping_patience,
        )

        X_imputed = model.get_normalized_expression(return_numpy=True)

    if gene_mask is not None:
        X_full = X_base.copy()
        X_full[:, gene_mask] = X_imputed
        X_imputed = X_full

    print("  ✓ scVI imputation complete")

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_imputed, overwrite=True)
        return None
    else:
        return X_imputed


def _run_scvi_impute_in_conda_env(X, batch, cell_ids, gene_names, n_latent, max_epochs,
                                  early_stopping, early_stopping_patience, batch_size,
                                  accelerator, gene_likelihood, random_state, conda_env):
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file  = os.path.join(tmpdir, "scvi_input.npz")
        output_file = os.path.join(tmpdir, "scvi_output.npy")
        script_file = os.path.join(tmpdir, "run_scvi.py")

        np.savez_compressed(
            input_file,
            X                       = X,
            batch                   = np.array([]) if batch is None else batch,
            cell_ids                = np.asarray(cell_ids, dtype=str),
            gene_names              = np.asarray(gene_names, dtype=str),
            n_latent                = n_latent,
            max_epochs              = -1 if max_epochs is None else int(max_epochs),
            early_stopping          = int(early_stopping),
            early_stopping_patience = int(early_stopping_patience),
            batch_size              = int(batch_size),
            accelerator             = str(accelerator),
            gene_likelihood         = str(gene_likelihood),
            random_state            = int(random_state),
        )

        script = f"""
import numpy as np
import pandas as pd
import anndata
import scvi

data                    = np.load('{input_file}', allow_pickle=True)
X                       = data['X']
batch                   = data['batch'].astype(str)
cell_ids                = data['cell_ids'].astype(str)
gene_names              = data['gene_names'].astype(str)
n_latent                = int(data['n_latent'])
max_epochs_raw          = int(data['max_epochs'])
early_stopping          = bool(int(data['early_stopping']))
early_stopping_patience = int(data['early_stopping_patience'])
batch_size              = int(data['batch_size'])
accelerator             = str(data['accelerator'])
gene_likelihood         = str(data['gene_likelihood'])
random_state            = int(data['random_state'])

max_epochs = None if max_epochs_raw < 0 else max_epochs_raw
has_batch  = len(batch) > 0

obs = pd.DataFrame(index=cell_ids)
if has_batch:
    obs['batch'] = batch
var   = pd.DataFrame(index=gene_names)
adata = anndata.AnnData(X=X, obs=obs, var=var)

scvi.settings.seed = random_state
scvi.model.SCVI.setup_anndata(adata, batch_key='batch' if has_batch else None)
model = scvi.model.SCVI(adata, n_latent=n_latent, gene_likelihood=gene_likelihood)

model.train(
    max_epochs              = max_epochs,
    accelerator             = accelerator,
    batch_size              = batch_size,
    early_stopping          = early_stopping,
    early_stopping_patience = early_stopping_patience,
)

X_imputed = model.get_normalized_expression(return_numpy=True)
np.save('{output_file}', X_imputed)
print('scVI done!')
"""
        with open(script_file, "w") as f:
            f.write(script)

        try:
            result = subprocess.run(["conda", "run", "-n", conda_env, "python", script_file],
                                    capture_output=True, text=True, check=True)
            print(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e1:
            try:
                sh = os.path.join(tmpdir, "run_scvi.sh")
                with open(sh, "w") as f:
                    f.write(f"#!/bin/bash\nsource $(conda info --base)/etc/profile.d/conda.sh\n"
                            f"conda activate {conda_env}\npython {script_file}\n")
                result = subprocess.run(["bash", sh], capture_output=True, text=True, check=True)
                print(result.stdout)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to run scVI in '{conda_env}'.\n{e1}\n{e2}\n"
                    f"Setup: conda create -n {conda_env} python=3.10\n"
                    f"       pip install scvi-tools anndata"
                ) from e2

        if not os.path.exists(output_file):
            raise RuntimeError(f"scVI failed — output not created in '{conda_env}'")
        return np.load(output_file)


def knn_smooth(
    spatioloji_obj,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = False,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_pca: bool = True,
    output_layer: str = "knn_smoothed",
    inplace: bool = True,
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
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X_neighbors = spatioloji_obj._embeddings["X_pca"][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs for neighbor finding")
        else:
            print("  Computing PCA for neighbor finding...")
            pca = PCA(n_components=min(n_pcs, X.shape[0], X.shape[1]))
            X_neighbors = pca.fit_transform(X)
    else:
        X_neighbors = X

        # Subset to HVGs if requested
        if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
            if gene_mask.sum() > 0:
                X_neighbors = X[:, gene_mask]
                print(f"  Using {gene_mask.sum()} HVGs for neighbor finding")

    # Build k-NN graph
    print("  Building k-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_neighbors)
    distances, indices = nbrs.kneighbors(X_neighbors)

    # Convert distances to weights (Gaussian kernel)
    sigma = np.median(distances[:, -1])  # Adaptive bandwidth
    weights = np.exp(-(distances**2) / (2 * sigma**2))
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
    conda_env: str | None = None,
    output_layer: str = "alra_imputed",
    inplace: bool = True,
):
    """
    ALRA imputation using pyALRA package.

    Requires: pip install pyALRA
    Works best on log-normalized data (NOT raw counts).
    """
    print("\nALRA imputation")

    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()

    gene_mask = None
    if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
        gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
        if gene_mask.sum() > 0:
            X_impute = X[:, gene_mask]
            print(f"  Imputing {gene_mask.sum()} highly variable genes")
        else:
            X_impute = X
    else:
        X_impute = X

    if conda_env is not None:
        print(f"  Running ALRA in conda environment: '{conda_env}'")
        X_imputed = _run_alra_in_conda_env(X_impute, k, conda_env)
    else:
        X_imputed = _run_alra_core(X_impute, k)

    if gene_mask is not None:
        X_full = X.copy()
        X_full[:, gene_mask] = X_imputed
        X_imputed = X_full

    print("  ✓ ALRA imputation complete")

    if inplace:
        spatioloji_obj.add_layer(output_layer, X_imputed, overwrite=True)
        return None
    else:
        return X_imputed


def _run_alra_core(X_impute, k):
    try:
        from pyALRA import alra, choose_k
    except ImportError as err:
        raise ImportError(
            "pyALRA not found. Install with: pip install pyALRA"
        ) from err

    if k is None:
        print("  Selecting rank automatically...")
        k = choose_k(X_impute)['k']
        print(f"    Selected rank: {k}")

    print(f"  Running pyALRA (k={k})...")
    X_imputed = alra(X_impute, k)['A_norm_rank_k_cor_sc']

    n_imp = (X_impute == 0).sum() - (X_imputed == 0).sum()
    print(f"    Imputed {n_imp:,} values ({n_imp / max((X_impute == 0).sum(), 1) * 100:.1f}% of zeros)")
    return X_imputed


def _run_alra_in_conda_env(X_impute, k, conda_env):
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file  = os.path.join(tmpdir, "alra_input.npz")
        output_file = os.path.join(tmpdir, "alra_output.npy")
        script_file = os.path.join(tmpdir, "run_alra.py")

        np.savez_compressed(input_file, X=X_impute,
                            k=-1 if k is None else int(k))

        script = f"""
import numpy as np
from pyALRA import alra, choose_k

data     = np.load('{input_file}', allow_pickle=True)
X_impute = data['X']
k        = None if int(data['k']) < 0 else int(data['k'])

if k is None:
    k = choose_k(X_impute)['k']
    print(f"Selected rank: {{k}}")

X_imputed = alra(X_impute, k)['A_norm_rank_k_cor_sc']
np.save('{output_file}', X_imputed)
print("ALRA done!")
"""
        with open(script_file, "w") as f:
            f.write(script)

        try:
            result = subprocess.run(["conda", "run", "-n", conda_env, "python", script_file],
                                    capture_output=True, text=True, check=True)
            print(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e1:
            try:
                sh = os.path.join(tmpdir, "run_alra.sh")
                with open(sh, "w") as f:
                    f.write(f"#!/bin/bash\nsource $(conda info --base)/etc/profile.d/conda.sh\n"
                            f"conda activate {conda_env}\npython {script_file}\n")
                result = subprocess.run(["bash", sh], capture_output=True, text=True, check=True)
                print(result.stdout)
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to run ALRA in '{conda_env}'.\n{e1}\n{e2}\n"
                    f"Setup: conda create -n {conda_env} python=3.10 && pip install pyALRA"
                ) from e2

        if not os.path.exists(output_file):
            raise RuntimeError(f"ALRA failed — output file not created in '{conda_env}'")
        return np.load(output_file)


def _mask_and_run(spatioloji_obj, method_name, method_kwargs, X_masked):
    """
    Deep-copy spatioloji, inject masked data, run imputation,
    and return full imputed matrix.
    """
    import copy

    sp_temp = copy.deepcopy(spatioloji_obj)
    sp_temp.add_layer("_masked", X_masked, overwrite=True)

    run_kwargs = {**method_kwargs, "layer": "_masked", "output_layer": "_imputed", "inplace": True}

    method_map = {
        "knn_smooth": knn_smooth,
        "alra": alra_impute,
        "magic": magic_impute,
        "scvi": scvi_impute,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Choose from: {list(method_map.keys())}"
        )

    method_map[method_name](sp_temp, **run_kwargs)

    X_imp = sp_temp.get_layer("_imputed")
    if sparse.issparse(X_imp):
        X_imp = X_imp.toarray()
    return np.asarray(X_imp, dtype=float)


def compare_imputation_methods(
    spatioloji_obj,
    methods: dict,
    layer: str = "log_normalized",
    test_genes: list[str] | None = None,
    n_test_genes: int = 10,
    mask_fraction: float = 0.1,
    random_state: int = 42,
    convert_raw_to_log_for_metrics: bool = True,
    raw_layer_keywords: tuple[str, ...] = ("raw", "count", "counts"),
) -> pd.DataFrame:
    """
    Compare imputation methods using a masking experiment (gold standard).

    Metrics:
    - RMSE
    - Pearson correlation
    - Spearman correlation
    - Calibration of variance
    - Recovery of zero inflation

    If a method uses a raw/count-like layer, metrics are optionally computed
    in log-normalized space for fair comparison.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error

    def _is_raw_like(layer_name: str | None) -> bool:
        if layer_name is None:
            return True
        lname = str(layer_name).lower()
        return any(k in lname for k in raw_layer_keywords)

    def _get_dense_layer(layer_name: str | None) -> np.ndarray:
        if layer_name is None:
            X_layer = spatioloji_obj.expression.get_dense()
        else:
            X_layer = spatioloji_obj.get_layer(layer_name)
        if sparse.issparse(X_layer):
            X_layer = X_layer.toarray()
        return np.asarray(X_layer, dtype=float)

    def _log_normalize_matrix(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
        X_nonneg = np.clip(np.asarray(X, dtype=float), a_min=0.0, a_max=None)
        libsize = X_nonneg.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1.0
        X_norm = X_nonneg / libsize * target_sum
        return np.log1p(X_norm)

    def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(pearsonr(x, y)[0])

    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return np.nan
        return float(spearmanr(x, y)[0])

    print(f"\nMasking experiment: comparing {list(methods.keys())}")
    print(f"  Reference layer : '{layer}'")
    print(f"  Mask fraction   : {mask_fraction:.0%}")

    np.random.seed(random_state)

    # Reference layer for selecting genes/mask positions
    X_ref = _get_dense_layer(layer)

    # Select genes (moderate sparsity preferred)
    if test_genes is None:
        sparsity = (X_ref == 0).mean(axis=0)
        candidates = np.where((sparsity > 0.2) & (sparsity < 0.8))[0]
        if len(candidates) < n_test_genes:
            print(f"  Warning: only {len(candidates)} genes with moderate sparsity; sampling from all genes")
            candidates = np.arange(X_ref.shape[1])
        test_gene_indices = np.random.choice(candidates, n_test_genes, replace=False)
    else:
        test_gene_indices = spatioloji_obj._get_gene_indices(test_genes)

    print(f"  Test genes      : {len(test_gene_indices)} selected")

    # Build mask positions from reference layer
    mask_positions = []
    for gene_idx in test_gene_indices:
        nonzero_idx = np.where(X_ref[:, gene_idx] > 0)[0]
        if len(nonzero_idx) < 10:
            continue
        n_mask = max(1, int(len(nonzero_idx) * mask_fraction))
        mask_idx = np.random.choice(nonzero_idx, n_mask, replace=False)
        mask_positions.extend((int(i), int(gene_idx)) for i in mask_idx)

    if len(mask_positions) == 0:
        raise ValueError("No values were masked. Check selected genes and mask_fraction.")

    mask_rows = np.asarray([i for i, _ in mask_positions], dtype=int)
    mask_cols = np.asarray([j for _, j in mask_positions], dtype=int)
    print(f"  Masked values   : {len(mask_rows)}")

    layer_cache: dict[str | None, np.ndarray] = {layer: X_ref}
    results = []

    for method_name, method_kwargs in methods.items():
        print(f"\n  [{method_name}] running on masked data...")
        try:
            run_kwargs = dict(method_kwargs)
            method_layer = run_kwargs.pop("layer", layer)

            if method_layer not in layer_cache:
                layer_cache[method_layer] = _get_dense_layer(method_layer)

            X_method_orig = layer_cache[method_layer]
            if X_method_orig.shape != X_ref.shape:
                raise ValueError(
                    f"Layer '{method_layer}' shape {X_method_orig.shape} does not match "
                    f"reference layer '{layer}' shape {X_ref.shape}"
                )

            X_method_masked = X_method_orig.copy()
            X_method_masked[mask_rows, mask_cols] = 0

            X_imputed = _mask_and_run(
                spatioloji_obj=spatioloji_obj,
                method_name=method_name,
                method_kwargs=run_kwargs,
                X_masked=X_method_masked,
            )

            metric_space = "native"
            X_true_metric = X_method_orig
            X_imp_metric = X_imputed

            if convert_raw_to_log_for_metrics and _is_raw_like(method_layer):
                print("    Converting raw/count-like layer to log-normalized space for metrics")
                X_true_metric = _log_normalize_matrix(X_method_orig)
                X_imp_metric = _log_normalize_matrix(X_imputed)
                metric_space = "log_normalized"

            true_values = X_true_metric[mask_rows, mask_cols]
            imputed_values = X_imp_metric[mask_rows, mask_cols]

            pearson_r = _safe_pearson(true_values, imputed_values)
            spearman_r = _safe_spearman(true_values, imputed_values)
            mse = float(mean_squared_error(true_values, imputed_values))
            rmse = float(np.sqrt(mse))

            true_var = float(np.var(true_values))
            imp_var = float(np.var(imputed_values))
            variance_ratio = np.nan if true_var == 0 else imp_var / true_var

            if np.isnan(variance_ratio) or variance_ratio <= 0:
                variance_calibration = np.nan
            else:
                # 1 is ideal, decreases symmetrically for over/under-smoothing
                variance_calibration = max(0.0, 1.0 - abs(np.log2(variance_ratio)))

            true_zero_frac = float((X_true_metric[:, test_gene_indices] == 0).mean())
            imp_zero_frac = float((X_imp_metric[:, test_gene_indices] == 0).mean())
            zero_inflation_recovery = max(0.0, 1.0 - abs(imp_zero_frac - true_zero_frac))

            variance_calibration_val = (
                round(variance_calibration, 4)
                if not np.isnan(variance_calibration)
                else np.nan
            )

            results.append({
                "method": method_name,
                "input_layer": method_layer,
                "metric_space": metric_space,
                "pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else np.nan,
                "spearman_r": round(spearman_r, 4) if not np.isnan(spearman_r) else np.nan,
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "variance_ratio": round(variance_ratio, 4) if not np.isnan(variance_ratio) else np.nan,
                "variance_calibration": variance_calibration_val,
                "zero_inflation_recovery": round(zero_inflation_recovery, 4),
            })

            print(f"    Pearson r : {pearson_r:.4f}")
            print(f"    Spearman r: {spearman_r:.4f}")
            print(f"    RMSE      : {rmse:.4f}")
            print(f"    Var ratio : {variance_ratio:.4f}")
            print(f"    Var calib : {variance_calibration:.4f}")
            print(f"    Zero recov: {zero_inflation_recovery:.4f}")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results.append({
                "method": method_name,
                "input_layer": method_kwargs.get("layer", layer),
                "metric_space": np.nan,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "variance_ratio": np.nan,
                "variance_calibration": np.nan,
                "zero_inflation_recovery": np.nan,
            })

    results_df = (
        pd.DataFrame(results)
        .sort_values("pearson_r", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    print("\n  ✓ Masking experiment complete")
    print("\n" + results_df.to_string(index=False))
    return results_df

