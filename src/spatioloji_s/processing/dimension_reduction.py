"""
dimension_reduction.py - Dimensionality reduction methods for spatial transcriptomics

Provides PCA, t-SNE, UMAP, and other methods for reducing high-dimensional
expression data to lower dimensions for visualization and clustering.

GPU acceleration
----------------
``pca``, ``tsne`` and ``umap`` accept a ``device`` parameter
(``'auto' | 'cpu' | 'gpu'``). When ``'auto'`` (default) and RAPIDS
(``cuml``, ``cupy``) is importable, computation runs on GPU and results
are returned as numpy arrays. RAPIDS must be installed separately into
the same conda env (not a pip dependency); see :mod:`._gpu`.
"""

import json
import os
import subprocess
import tempfile
import warnings
from typing import Literal

import numpy as np
from scipy import sparse

from ._gpu import Device, _resolve_device, _to_cupy, _to_numpy, _warn_fallback


def _run_in_conda_env(conda_env: str, script: str, input_path: str, output_path: str) -> None:
    """Run a Python script in a different conda environment via subprocess.

    Parameters
    ----------
    conda_env : str
        Name of the target conda environment.
    script : str
        Python script source code to execute.
    input_path : str
        Path to the input .npy file (already written by caller).
    output_path : str
        Path where the script should write its output .npy file.

    Raises
    ------
    RuntimeError
        If the subprocess fails.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["conda", "run", "-n", conda_env, "--no-capture-output", "python", script_path],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise RuntimeError(
                f"conda_env='{conda_env}' subprocess failed (exit {result.returncode}):\n{result.stderr}"
            )
    finally:
        os.unlink(script_path)


def _remote_dr(conda_env: str, method: str, X: np.ndarray, params: dict) -> np.ndarray:
    """Serialize data, run DR in another conda env, return results.

    Parameters
    ----------
    conda_env : str
        Conda environment name.
    method : str
        One of 'tsne_gpu', 'tsne_opentsne', 'umap_gpu', 'umap_cpu'.
    X : np.ndarray
        Input matrix (n_cells, n_features).
    params : dict
        Parameters to pass to the DR algorithm.

    Returns
    -------
    np.ndarray
        Embedding coordinates.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "X_input.npy")
        output_path = os.path.join(tmpdir, "X_output.npy")
        params_path = os.path.join(tmpdir, "params.json")

        np.save(input_path, X)
        with open(params_path, "w") as f:
            json.dump(params, f)

        script = _build_dr_script(method, input_path, output_path, params_path)

        print(f"  Running {method} in conda env '{conda_env}'...")
        _run_in_conda_env(conda_env, script, input_path, output_path)

        if not os.path.exists(output_path):
            raise RuntimeError(f"Expected output file not found: {output_path}")

        return np.load(output_path)


def _build_dr_script(method: str, input_path: str, output_path: str, params_path: str) -> str:
    """Generate a self-contained Python script for a DR method."""
    # Escape backslashes for Windows paths in the generated script
    ip = input_path.replace("\\", "\\\\")
    op = output_path.replace("\\", "\\\\")
    pp = params_path.replace("\\", "\\\\")

    return f'''
import json
import numpy as np

X = np.load("{ip}")
with open("{pp}") as f:
    params = json.load(f)

if "{method}" == "tsne_gpu":
    from cuml.manifold import TSNE
    model = TSNE(
        n_components=params["n_components"],
        perplexity=params["perplexity"],
        early_exaggeration=params["early_exaggeration"],
        learning_rate=params["learning_rate"],
        n_iter=params["n_iter"],
        random_state=params.get("random_state"),
        verbose=0,
    )
    result = model.fit_transform(X)
    if hasattr(result, "get"):
        result = result.get()

elif "{method}" == "tsne_opentsne":
    from openTSNE import TSNE
    model = TSNE(
        n_components=params["n_components"],
        perplexity=params["perplexity"],
        early_exaggeration=params["early_exaggeration"],
        learning_rate=params["learning_rate"],
        n_iter=params["n_iter"],
        random_state=params.get("random_state"),
        n_jobs=params.get("n_jobs", 1),
        verbose=False,
    )
    result = np.asarray(model.fit(X))

elif "{method}" == "umap_gpu":
    from cuml.manifold import UMAP
    model = UMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params.get("metric", "euclidean"),
        random_state=params.get("random_state"),
        verbose=False,
    )
    result = model.fit_transform(X)
    if hasattr(result, "get"):
        result = result.get()

elif "{method}" == "umap_cpu":
    import umap
    model = umap.UMAP(
        n_components=params["n_components"],
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params.get("metric", "euclidean"),
        random_state=params.get("random_state"),
        n_jobs=params.get("n_jobs", 1),
        low_memory=X.shape[0] > 50000,
        verbose=False,
    )
    result = model.fit_transform(X)

else:
    raise ValueError(f"Unknown method: {method}")

np.save("{op}", np.asarray(result, dtype=np.float64))
print(f"  ✓ {{"{method}"}} complete ({{result.shape[0]}} cells, {{result.shape[1]}} dims)")
'''


def _pca_gpu(X, n_comps: int, random_state: int):
    """cuML PCA path. Returns (X_pca, variance, variance_ratio, components) as numpy.

    Notes
    -----
    cuML's ``PCA.__init__`` does **not** accept ``random_state`` — full and
    Jacobi solvers are deterministic, and the iterated-power solver controls
    its randomness via ``iterated_power`` (an int) rather than a seed.  We
    accept ``random_state`` to keep the public API stable with the sklearn
    path but ignore it on the GPU side; PCA results match the CPU within
    sign-flip / numerical tolerance regardless.
    """
    from cuml.decomposition import PCA as cuPCA

    print("  Running PCA (cuML GPU)...")
    X_gpu = _to_cupy(X)
    # cuML PCA dense path is well-supported; sparse path uses TruncatedSVD
    # under the hood for some versions, so fall back to dense on GPU if sparse.
    if sparse.issparse(X):
        # cupyx CSR → dense on GPU (cheap memory-wise after HVG subset)
        X_gpu = X_gpu.toarray()
    del random_state  # not a supported cuML PCA kwarg; see docstring
    model = cuPCA(n_components=n_comps)
    X_pca_gpu = model.fit_transform(X_gpu)
    return (
        _to_numpy(X_pca_gpu),
        _to_numpy(model.explained_variance_),
        _to_numpy(model.explained_variance_ratio_),
        _to_numpy(model.components_),
    )


def _pca_cpu(X, n_comps: int, random_state: int, svd_solver: str):
    """scikit-learn PCA path. Returns (X_pca, variance, variance_ratio, components).

    Notes
    -----
    ``svd_solver`` defaults to ``'arpack'`` rather than scikit-learn's own
    ``'auto'``.  ``'auto'`` resolves to the *randomized* solver for the shapes
    typical of transcriptomics (many cells, a few thousand genes, ~50
    components), and randomized SVD only approximates the trailing
    components.  On a slowly-decaying covariance spectrum — the normal case
    for log-normalized expression — that approximation puts the top-50
    subspace >10 degrees away from the exact one and away from scanpy, which
    defaults to ``'arpack'`` for the same reason.

    ``'arpack'`` requires ``n_comps < min(n_cells, n_genes)``; a request for
    the full rank falls back to the exact dense ``'full'`` solver.
    """
    from sklearn.decomposition import PCA as _SKPCA

    solver = svd_solver
    if solver == "arpack" and n_comps >= min(X.shape):
        # arpack computes at most min(shape) - 1 singular triplets
        solver = "full"

    def _fit(matrix):
        model = _SKPCA(n_components=n_comps, svd_solver=solver, random_state=random_state)
        coords = model.fit_transform(matrix)
        return coords, model.explained_variance_, model.explained_variance_ratio_, model.components_

    print(f"  Running PCA (scikit-learn, svd_solver='{solver}')...")

    if sparse.issparse(X):
        if solver in ("randomized", "full"):
            X = X.toarray()  # neither solver accepts sparse input
        else:
            try:
                return _fit(X)
            except TypeError:
                # scikit-learn < 1.4 has no sparse PCA path
                X = X.toarray()

    return _fit(X)


def pca(
    spatioloji_obj,
    layer: str | None = "scaled",
    use_highly_variable: bool = True,
    n_comps: int = 50,
    random_state: int = 42,
    output_key: str = "X_pca",
    inplace: bool = True,
    device: Device = "auto",
    svd_solver: Literal["arpack", "full", "randomized", "covariance_eigh", "auto"] = "arpack",
):
    """
    Principal Component Analysis (PCA).

    Reduces dimensionality by finding orthogonal directions of maximum variance.
    This is typically the first step before clustering or other analyses.

    The data is always zero-centered (never scaled to unit variance) before
    decomposition, matching ``scanpy.pp.pca``'s ``zero_center=True`` default.
    Scale the input beforehand — e.g. ``layer='scaled'`` — if you want that.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to use, by default 'scaled'
    use_highly_variable : bool, optional
        Use only highly variable genes if available, by default True
    n_comps : int, optional
        Number of principal components to compute, by default 50
    random_state : int, optional
        Random seed for reproducibility, by default 42
    output_key : str, optional
        Key name for storing PCA results, by default 'X_pca'
    inplace : bool, optional
        Store results in spatioloji object, by default True
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cuML on GPU when RAPIDS
        is importable, otherwise scikit-learn on CPU.  ``'gpu'`` requires
        RAPIDS in the same conda env.  Sparse inputs stay sparse on the GPU
        path (cupyx CSR), avoiding a costly densification.
    svd_solver : {'arpack', 'full', 'randomized', 'covariance_eigh', 'auto'}, optional
        scikit-learn SVD solver for the CPU path, by default ``'arpack'`` —
        the same default ``scanpy.pp.pca`` uses, and the reason results agree
        with it.  ``'randomized'`` (and ``'auto'``, which selects it for
        typical shapes) is faster but only approximates the trailing
        components; on expression data that moves the top-50 subspace >10
        degrees off the exact answer.  Ignored on the GPU path.

    Returns
    -------
    dict or None
        If inplace=False, returns dict with:
        - 'X_pca': PCA coordinates (n_cells × n_comps)
        - 'variance': Variance explained by each PC
        - 'variance_ratio': Fraction of variance explained by each PC
        - 'components': Principal components (n_comps × n_genes)
        Otherwise None

    Examples
    --------
    >>> # Standard PCA with 50 components
    >>> sp.processing.pca(sp, layer='log_normalized', n_comps=50)
    >>>
    >>> # PCA on all genes (not recommended)
    >>> sp.processing.pca(sp, use_highly_variable=False, n_comps=100)
    >>>
    >>> # Access results
    >>> pca_coords = sp.cell_meta[['PC1', 'PC2', 'PC3', ...]]
    >>> # Or if stored as embedding
    >>> pca_coords = sp.embeddings['X_pca']
    >>>
    >>> # Trade exactness for speed on a very large panel
    >>> sj.processing.pca(sp, n_comps=50, svd_solver='randomized')
    >>>
    >>> # Force GPU
    >>> sj.processing.pca(sp, n_comps=50, device='gpu')
    """
    print(f"\nComputing PCA (n_comps={n_comps})")

    # Get expression data — DO NOT densify yet; GPU path can take CSR.
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)

    # Subset to HVGs if requested (works for both sparse CSR and dense)
    if use_highly_variable:
        if "highly_variable" not in spatioloji_obj.gene_meta.columns:
            warnings.warn(
                "No 'highly_variable' column in gene_meta. "
                "Run highly_variable_genes() first or set use_highly_variable=False. "
                "Using all genes.",
                stacklevel=2,
            )
            gene_mask = np.ones(X.shape[1], dtype=bool)
        else:
            gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
            n_hvg = gene_mask.sum()

            if n_hvg == 0:
                warnings.warn("No highly variable genes found. Using all genes.", stacklevel=2)
                gene_mask = np.ones(X.shape[1], dtype=bool)
            else:
                X = X[:, gene_mask]
                print(f"  Using {n_hvg} highly variable genes")
    else:
        gene_mask = np.ones(X.shape[1], dtype=bool)

    # Determine actual number of components
    n_comps_actual = min(n_comps, X.shape[0], X.shape[1])
    if n_comps_actual < n_comps:
        print(f"  Reducing n_comps to {n_comps_actual} (limited by data dimensions)")

    backend = _resolve_device(device, "PCA")

    if backend == "gpu":
        try:
            X_pca, variance, variance_ratio, components = _pca_gpu(X, n_comps_actual, random_state)
            print("  ✓ PCA complete (cuML GPU)")
        except Exception as exc:
            if device == "gpu":
                raise
            _warn_fallback("PCA", exc)
            backend = "cpu"

    if backend == "cpu":
        X_pca, variance, variance_ratio, components = _pca_cpu(
            X, n_comps_actual, random_state, svd_solver
        )
        print("  ✓ PCA complete")
    print(f"    Variance explained by first 10 PCs: {variance_ratio[:10].sum() * 100:.1f}%")
    print(f"    Variance explained by all {n_comps_actual} PCs: {variance_ratio.sum() * 100:.1f}%")

    # Prepare results
    results = {"X_pca": X_pca, "variance": variance, "variance_ratio": variance_ratio, "components": components}

    if inplace:
        # Store embeddings
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}

        spatioloji_obj._embeddings[output_key] = X_pca
        spatioloji_obj._embeddings[f"{output_key}_variance"] = variance
        spatioloji_obj._embeddings[f"{output_key}_variance_ratio"] = variance_ratio

        # Also add first few PCs to cell_meta for easy access
        n_pcs_to_add = min(50, n_comps_actual)
        for i in range(n_pcs_to_add):
            spatioloji_obj._cell_meta[f"PC{i + 1}"] = X_pca[:, i]

        return None
    else:
        return results


def _get_pca_or_expr(spatioloji_obj, use_pca, n_pcs, layer, use_highly_variable):
    """Get input matrix: PCA embeddings or expression data."""
    if use_pca:
        if hasattr(spatioloji_obj, "_embeddings") and "X_pca" in spatioloji_obj._embeddings:
            X = np.asarray(spatioloji_obj._embeddings["X_pca"])[:, :n_pcs]
            print(f"  Using stored PCA (first {n_pcs} PCs)")
        else:
            print("  No stored PCA found, computing PCA first...")
            pca(spatioloji_obj, n_comps=n_pcs, layer=layer, use_highly_variable=use_highly_variable, inplace=True)
            X = spatioloji_obj._embeddings["X_pca"][:, :n_pcs]
    else:
        if layer is None:
            X = spatioloji_obj.expression.get_dense()
        else:
            X = spatioloji_obj.get_layer(layer)
            if sparse.issparse(X):
                X = X.toarray()
        if use_highly_variable and "highly_variable" in spatioloji_obj.gene_meta.columns:
            gene_mask = spatioloji_obj.gene_meta["highly_variable"].values
            if gene_mask.sum() > 0:
                X = X[:, gene_mask]
                print(f"  Using {gene_mask.sum()} highly variable genes")
    return X


def tsne(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = True,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float | str = "auto",
    n_iter: int = 1000,
    random_state: int = 42,
    device: Device = "auto",
    gpu: bool | None = None,
    n_jobs: int = 1,
    conda_env: str | None = None,
    output_key: str = "X_tsne",
    inplace: bool = True,
):
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding).

    Non-linear dimensionality reduction for visualization. Preserves local
    structure of data. Typically used for 2D/3D visualization.

    Three backends are tried in order:

    1. ``gpu=True`` → **cuML** (RAPIDS GPU, fastest for >100k cells)
    2. **openTSNE** (fast multi-threaded CPU, installed via
       ``pip install openTSNE``)
    3. **sklearn** (fallback, single-threaded, slowest)

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    use_pca : bool, optional
        Run on PCA coordinates instead of raw expression, by default True
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    layer : str, optional
        Which layer to use, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only highly variable genes, by default True
    n_components : int, optional
        Number of t-SNE dimensions (2 or 3), by default 2
    perplexity : float, optional
        Perplexity parameter (5-50 typical), by default 30.0
    early_exaggeration : float, optional
        Controls tightness of clusters, by default 12.0
    learning_rate : float or 'auto', optional
        Learning rate for optimization, by default 'auto'
    n_iter : int, optional
        Number of iterations, by default 1000
    random_state : int, optional
        Random seed, by default 42
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cuML on GPU when RAPIDS
        is importable, otherwise the CPU openTSNE/sklearn backend.
    gpu : bool, optional
        **Deprecated**, use ``device='gpu'`` instead.  When set, takes
        precedence over ``device``.  Will be removed in a future release.
    n_jobs : int, optional
        Number of threads for openTSNE backend, by default 1.
        Set to -1 for all CPUs.  Ignored for sklearn and cuML backends.
    conda_env : str or None, optional
        If set, run the computation in a different conda environment
        via subprocess.  Useful when GPU libraries (cuML) or openTSNE
        are installed in a separate env.  Data is serialized to disk,
        the script runs in the target env, and results are loaded back.
    output_key : str, optional
        Key name for storing results, by default 'X_tsne'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        t-SNE coordinates if inplace=False, otherwise None

    Examples
    --------
    >>> # Standard t-SNE (auto-selects fastest available backend)
    >>> sj.processing.tsne(sp, perplexity=30)
    >>>
    >>> # Force GPU (requires RAPIDS cuML in same env)
    >>> sj.processing.tsne(sp, device='gpu', perplexity=30)
    >>>
    >>> # GPU in a separate conda env (cuML installed there)
    >>> sj.processing.tsne(sp, device='gpu', conda_env='rapids')
    >>>
    >>> # Fast CPU with openTSNE (multi-threaded)
    >>> sj.processing.tsne(sp, device='cpu', n_jobs=-1)
    """
    import os

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Resolve deprecated gpu= flag
    if gpu is not None:
        warnings.warn(
            "tsne(gpu=...) is deprecated; use device='gpu' or device='cpu' instead.",
            DeprecationWarning, stacklevel=2,
        )
        device = "gpu" if gpu else "cpu"

    X = _get_pca_or_expr(spatioloji_obj, use_pca, n_pcs, layer, use_highly_variable)
    lr = 200.0 if learning_rate == "auto" else float(learning_rate)

    tsne_params = {
        "n_components": n_components, "perplexity": perplexity,
        "early_exaggeration": early_exaggeration, "learning_rate": lr,
        "n_iter": n_iter, "random_state": random_state, "n_jobs": n_jobs,
    }

    # --- Remote execution in another conda env (subprocess path) ---
    if conda_env is not None:
        method = "tsne_gpu" if device == "gpu" else "tsne_opentsne"
        X_tsne = _remote_dr(conda_env, method, X, tsne_params)

    # --- In-process dispatch ---
    else:
        backend = _resolve_device(device, "t-SNE")

        if backend == "gpu":
            try:
                from cuml.manifold import TSNE as cuTSNE

                print(f"\nt-SNE [cuML GPU] (n_components={n_components}, perplexity={perplexity})")
                print("  Running t-SNE on GPU...")
                tsne_model = cuTSNE(
                    n_components=n_components, perplexity=perplexity,
                    early_exaggeration=early_exaggeration, learning_rate=lr,
                    n_iter=n_iter, random_state=random_state, verbose=0,
                )
                X_tsne = tsne_model.fit_transform(_to_cupy(X))
                X_tsne = np.asarray(_to_numpy(X_tsne), dtype=np.float64)
                print("  ✓ t-SNE complete (GPU)")
            except Exception as exc:
                if device == "gpu":
                    raise
                _warn_fallback("t-SNE", exc)
                backend = "cpu"

        if backend == "cpu":
            try:
                from openTSNE import TSNE as openTSNE

                print(f"\nt-SNE [openTSNE] (n_components={n_components}, perplexity={perplexity}, n_jobs={n_jobs})")
                print("  Running t-SNE...")
                tsne_model = openTSNE(
                    n_components=n_components, perplexity=perplexity,
                    early_exaggeration=early_exaggeration, learning_rate=lr,
                    n_iter=n_iter, random_state=random_state, n_jobs=n_jobs, verbose=False,
                )
                X_tsne = np.asarray(tsne_model.fit(X))
                print("  ✓ t-SNE complete (openTSNE)")
            except ImportError:
                from sklearn.manifold import TSNE

                print(f"\nt-SNE [sklearn] (n_components={n_components}, perplexity={perplexity})")
                print("  Running t-SNE (install openTSNE for ~10x speedup: pip install openTSNE)...")
                tsne_model = TSNE(
                    n_components=n_components, perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,  # sklearn accepts 'auto'
                    n_iter=n_iter, random_state=random_state, verbose=0,
                )
                X_tsne = tsne_model.fit_transform(X)
                print(f"  ✓ t-SNE complete (KL divergence: {tsne_model.kl_divergence_:.3f})")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}
        spatioloji_obj._embeddings[output_key] = X_tsne
        for i in range(n_components):
            spatioloji_obj._cell_meta[f"tSNE{i + 1}"] = X_tsne[:, i]
        return None
    return X_tsne


def umap(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = True,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 42,
    n_jobs: int = 1,
    device: Device = "auto",
    gpu: bool | None = None,
    conda_env: str | None = None,
    output_key: str = "X_umap",
    inplace: bool = True,
    init: Literal["spectral", "random"] = "spectral",
    n_epochs: int | None = None,
):
    """
    UMAP (Uniform Manifold Approximation and Projection).

    Non-linear dimensionality reduction that preserves both local and global
    structure. Generally faster than t-SNE and better at preserving global
    structure.

    Two backends:

    1. ``gpu=True`` → **cuML** (RAPIDS GPU, fastest for >100k cells)
    2. ``gpu=False`` → **umap-learn** (CPU, supports ``n_jobs``)

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    use_pca : bool, optional
        Run on PCA coordinates instead of raw expression, by default True
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    layer : str, optional
        Which layer to use, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only highly variable genes, by default True
    n_components : int, optional
        Number of UMAP dimensions (2 or 3), by default 2
    n_neighbors : int, optional
        Number of neighbors (5-100 typical), by default 15
    min_dist : float, optional
        Minimum distance between points (0-1), by default 0.1
    metric : str, optional
        Distance metric, by default 'euclidean'
    random_state : int or None, optional
        Random seed, by default 42. Note: umap-learn forces ``n_jobs=1``
        when a seed is set. Pass ``random_state=None`` to enable parallel
        execution with ``n_jobs > 1``.
    n_jobs : int, optional
        Number of parallel threads (CPU backend only), by default 1.
        Set to -1 for all CPUs.  Only effective when
        ``random_state=None`` (umap-learn limitation).
    device : {'auto', 'cpu', 'gpu'}, optional
        Compute backend.  ``'auto'`` (default) uses cuML on GPU when RAPIDS
        is importable, otherwise umap-learn on CPU.
    gpu : bool, optional
        **Deprecated**, use ``device='gpu'`` instead.  When set, takes
        precedence over ``device``.  Will be removed in a future release.
    conda_env : str or None, optional
        Run computation in a different conda environment via subprocess.
        Useful when cuML or umap-learn is in a separate env.
    output_key : str, optional
        Key name for storing results, by default 'X_umap'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        UMAP coordinates if inplace=False, otherwise None

    Examples
    --------
    >>> # Standard UMAP (auto GPU/CPU)
    >>> sj.processing.umap(sp, n_neighbors=15)
    >>>
    >>> # Force GPU (requires RAPIDS cuML in same env)
    >>> sj.processing.umap(sp, device='gpu', n_neighbors=15)
    >>>
    >>> # Fast parallel CPU (non-reproducible)
    >>> sj.processing.umap(sp, device='cpu', n_jobs=-1, random_state=None)
    """
    import os

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if gpu is not None:
        warnings.warn(
            "umap(gpu=...) is deprecated; use device='gpu' or device='cpu' instead.",
            DeprecationWarning, stacklevel=2,
        )
        device = "gpu" if gpu else "cpu"

    X = _get_pca_or_expr(spatioloji_obj, use_pca, n_pcs, layer, use_highly_variable)

    umap_params = {
        "n_components": n_components, "n_neighbors": n_neighbors,
        "min_dist": min_dist, "metric": metric,
        "random_state": random_state, "n_jobs": n_jobs,
    }

    # --- Remote execution (subprocess into another conda env) ---
    if conda_env is not None:
        method = "umap_gpu" if device == "gpu" else "umap_cpu"
        if device != "gpu" and n_jobs > 1 and random_state is not None:
            umap_params["random_state"] = None
        X_umap = _remote_dr(conda_env, method, X, umap_params)

    # --- In-process dispatch ---
    else:
        backend = _resolve_device(device, "UMAP")

        if backend == "gpu":
            try:
                from cuml.manifold import UMAP as cuUMAP

                # cuML defaults to init='random' once n_obs > 50_000, which
                # produces dispersed / "swirly" UMAPs that don't resemble the
                # CPU spectral-init layout.  Honor the explicit init=
                # argument so the GPU run matches CPU appearance.
                gpu_kw = dict(
                    n_components=n_components, n_neighbors=n_neighbors,
                    min_dist=min_dist, metric=metric,
                    random_state=random_state, verbose=False,
                    init=init,
                )
                if n_epochs is not None:
                    gpu_kw["n_epochs"] = n_epochs

                print(
                    f"\nUMAP [cuML GPU] (n_components={n_components}, "
                    f"n_neighbors={n_neighbors}, min_dist={min_dist}, "
                    f"init={init}, n_epochs={n_epochs})"
                )
                print("  Running UMAP on GPU...")
                umap_model = cuUMAP(**gpu_kw)
                X_umap = umap_model.fit_transform(_to_cupy(X))
                X_umap = np.asarray(_to_numpy(X_umap), dtype=np.float64)
                print("  ✓ UMAP complete (GPU)")
            except Exception as exc:
                if device == "gpu":
                    raise
                _warn_fallback("UMAP", exc)
                backend = "cpu"

        if backend == "cpu":
            try:
                import umap as umap_package
            except ImportError as err:
                raise ImportError("UMAP requires umap-learn: pip install umap-learn") from err

            rs = random_state
            if n_jobs > 1 and rs is not None:
                warnings.warn(
                    f"random_state={rs} forces n_jobs=1 in umap-learn. "
                    "Dropping random_state for parallel execution.",
                    UserWarning, stacklevel=2,
                )
                rs = None

            cpu_kw = dict(
                n_components=n_components, n_neighbors=n_neighbors,
                min_dist=min_dist, metric=metric, random_state=rs,
                n_jobs=n_jobs, low_memory=X.shape[0] > 50_000, verbose=False,
                init=init,
            )
            if n_epochs is not None:
                cpu_kw["n_epochs"] = n_epochs

            print(
                f"\nUMAP [umap-learn] (n_components={n_components}, "
                f"n_neighbors={n_neighbors}, min_dist={min_dist}, "
                f"n_jobs={n_jobs}, init={init})"
            )
            print("  Running UMAP...")
            umap_model = umap_package.UMAP(**cpu_kw)
            X_umap = umap_model.fit_transform(X)
            print("  ✓ UMAP complete")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}
        spatioloji_obj._embeddings[output_key] = X_umap
        for i in range(n_components):
            spatioloji_obj._cell_meta[f"UMAP{i + 1}"] = X_umap[:, i]
        return None
    return X_umap


def diffusion_map(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: str | None = "log_normalized",
    use_highly_variable: bool = True,
    n_components: int = 10,
    n_neighbors: int = 30,
    n_jobs: int = -1,
    random_state: int = 42,
    output_key: str = "X_diffmap",
    inplace: bool = True,
):
    """
    Diffusion map for capturing continuous trajectories.

    Useful for trajectory inference and capturing gradual transitions.
    Preserves the diffusion distance which represents similarity.

    The KNN step uses **pynndescent** (approximate, multi-threaded)
    when available, falling back to sklearn otherwise.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    use_pca : bool, optional
        Run on PCA coordinates, by default True
    n_pcs : int, optional
        Number of PCs to use if use_pca=True, by default 50
    layer : str, optional
        Which layer to use, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only highly variable genes, by default True
    n_components : int, optional
        Number of diffusion components, by default 10
    n_neighbors : int, optional
        Number of neighbors for graph, by default 30
    n_jobs : int, optional
        Number of threads for KNN computation, by default -1 (all CPUs).
        Effective with pynndescent; sklearn uses single-thread KDTree.
    random_state : int, optional
        Random seed, by default 42
    output_key : str, optional
        Key name for storing results, by default 'X_diffmap'
    inplace : bool, optional
        Store results in spatioloji object, by default True

    Returns
    -------
    np.ndarray or None
        Diffusion map coordinates if inplace=False, otherwise None

    Examples
    --------
    >>> # Diffusion map for trajectory analysis
    >>> sj.processing.diffusion_map(sp, n_components=10, n_neighbors=30)
    >>>
    >>> # Access results
    >>> diffmap_coords = sp.embeddings['X_diffmap']
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    print(f"\nDiffusion Map (n_components={n_components}, n_neighbors={n_neighbors})")

    X = _get_pca_or_expr(spatioloji_obj, use_pca, n_pcs, layer, use_highly_variable)
    n_cells = X.shape[0]

    # Build k-NN graph (pynndescent if available, else sklearn)
    print("  Building k-NN graph...")
    try:
        from pynndescent import NNDescent

        print("    Using pynndescent (approximate, multi-threaded)...")
        index = NNDescent(X, n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=n_jobs)
        indices, distances = index.neighbor_graph
        # Drop self (index 0)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        print("    Using sklearn (install pynndescent for ~10x speedup)...")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
        distances, indices = nbrs.kneighbors(X)

    # Construct affinity matrix (Gaussian kernel)
    print("  Computing diffusion operator...")
    sigma = np.median(distances[:, -1])  # Adaptive bandwidth

    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    weights = np.exp(-(distances.flatten() ** 2) / (2 * sigma**2))

    W = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))
    W = (W + W.T) / 2  # Symmetrize

    # Normalize to get diffusion operator (use sparse diagonal)
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt_vals = 1.0 / np.sqrt(np.maximum(D, 1e-12))
    D_inv_sqrt = csr_matrix((D_inv_sqrt_vals, (np.arange(n_cells), np.arange(n_cells))), shape=(n_cells, n_cells))
    L = D_inv_sqrt @ W @ D_inv_sqrt

    # Compute eigenvectors
    print("  Computing eigenvectors...")
    n_components_actual = min(n_components + 1, n_cells - 2)
    eigenvalues, eigenvectors = eigsh(L, k=n_components_actual, which="LM")

    # Sort by eigenvalue (descending)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip first eigenvector (trivial)
    X_diffmap = eigenvectors[:, 1 : n_components + 1]

    print("  ✓ Diffusion map complete")
    print(f"    Top 5 eigenvalues: {eigenvalues[1:6]}")

    if inplace:
        if not hasattr(spatioloji_obj, "_embeddings"):
            spatioloji_obj._embeddings = {}
        spatioloji_obj._embeddings[output_key] = X_diffmap
        spatioloji_obj._embeddings[f"{output_key}_eigenvalues"] = eigenvalues
        for i in range(min(n_components, X_diffmap.shape[1])):
            spatioloji_obj._cell_meta[f"DC{i + 1}"] = X_diffmap[:, i]
        return None
    return X_diffmap


def plot_pca_variance(spatioloji_obj, n_pcs: int = 50, save_path: str | None = None, show_plot: bool = True):
    """
    Plot variance explained by principal components.

    Helps determine how many PCs to use for downstream analysis.

    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with PCA results
    n_pcs : int, optional
        Number of PCs to plot, by default 50
    save_path : str, optional
        Path to save figure
    show_plot : bool, optional
        Whether to display the plot, by default True

    Examples
    --------
    >>> sp.processing.pca(sp, n_comps=50)
    >>> sp.processing.plot_pca_variance(sp, n_pcs=50, save_path='./pca_variance.png')
    """
    import matplotlib.pyplot as plt

    if not hasattr(spatioloji_obj, "_embeddings") or "X_pca_variance_ratio" not in spatioloji_obj._embeddings:
        raise ValueError("No PCA results found. Run pca() first.")

    variance_ratio = spatioloji_obj._embeddings["X_pca_variance_ratio"]
    n_pcs = min(n_pcs, len(variance_ratio))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot variance explained per PC
    ax1.bar(range(1, n_pcs + 1), variance_ratio[:n_pcs] * 100)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Variance Explained per PC")
    ax1.grid(True, alpha=0.3)

    # Plot cumulative variance
    cumsum = np.cumsum(variance_ratio[:n_pcs]) * 100
    ax2.plot(range(1, n_pcs + 1), cumsum, "b-", linewidth=2)
    ax2.axhline(y=80, color="r", linestyle="--", label="80% variance")
    ax2.axhline(y=90, color="orange", linestyle="--", label="90% variance")
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Print summary
    pc_80 = np.where(cumsum >= 80)[0][0] + 1 if any(cumsum >= 80) else n_pcs
    pc_90 = np.where(cumsum >= 90)[0][0] + 1 if any(cumsum >= 90) else n_pcs

    print("\nPCA Variance Summary:")
    print(f"  PCs needed for 80% variance: {pc_80}")
    print(f"  PCs needed for 90% variance: {pc_90}")
    print(f"  Total variance (first {n_pcs} PCs): {cumsum[-1]:.1f}%")
