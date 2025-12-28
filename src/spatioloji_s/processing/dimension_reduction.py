"""
dimension_reduction.py - Dimensionality reduction methods for spatial transcriptomics

Provides PCA, t-SNE, UMAP, and other methods for reducing high-dimensional
expression data to lower dimensions for visualization and clustering.
"""

from typing import Optional, Literal, Tuple, Union, Dict
import numpy as np
import pandas as pd
from scipy import sparse
import warnings


def pca(
    spatioloji_obj,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    n_comps: int = 50,
    zero_center: bool = True,
    random_state: int = 42,
    output_key: str = 'X_pca',
    inplace: bool = True
):
    """
    Principal Component Analysis (PCA).
    
    Reduces dimensionality by finding orthogonal directions of maximum variance.
    This is typically the first step before clustering or other analyses.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    layer : str, optional
        Which layer to use, by default 'log_normalized'
    use_highly_variable : bool, optional
        Use only highly variable genes if available, by default True
    n_comps : int, optional
        Number of principal components to compute, by default 50
    zero_center : bool, optional
        Center data to zero mean before PCA, by default True
    random_state : int, optional
        Random seed for reproducibility, by default 42
    output_key : str, optional
        Key name for storing PCA results, by default 'X_pca'
    inplace : bool, optional
        Store results in spatioloji object, by default True
    
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
    """
    from sklearn.decomposition import PCA
    
    print(f"\nComputing PCA (n_comps={n_comps})")
    
    # Get expression data
    if layer is None:
        X = spatioloji_obj.expression.get_dense()
    else:
        X = spatioloji_obj.get_layer(layer)
        if sparse.issparse(X):
            X = X.toarray()
    
    # Subset to HVGs if requested
    if use_highly_variable:
        if 'highly_variable' not in spatioloji_obj.gene_meta.columns:
            warnings.warn(
                "No 'highly_variable' column in gene_meta. "
                "Run highly_variable_genes() first or set use_highly_variable=False. "
                "Using all genes."
            )
            gene_mask = np.ones(X.shape[1], dtype=bool)
        else:
            gene_mask = spatioloji_obj.gene_meta['highly_variable'].values
            n_hvg = gene_mask.sum()
            
            if n_hvg == 0:
                warnings.warn("No highly variable genes found. Using all genes.")
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
    
    # Run PCA
    print(f"  Running PCA...")
    pca_model = PCA(
        n_components=n_comps_actual,
        random_state=random_state
    )
    
    if zero_center:
        X_pca = pca_model.fit_transform(X)
    else:
        # Manual centering control
        X_pca = pca_model.fit_transform(X - X.mean(axis=0))
    
    # Get variance explained
    variance = pca_model.explained_variance_
    variance_ratio = pca_model.explained_variance_ratio_
    components = pca_model.components_
    
    print(f"  ✓ PCA complete")
    print(f"    Variance explained by first 10 PCs: {variance_ratio[:10].sum()*100:.1f}%")
    print(f"    Variance explained by all {n_comps_actual} PCs: {variance_ratio.sum()*100:.1f}%")
    
    # Prepare results
    results = {
        'X_pca': X_pca,
        'variance': variance,
        'variance_ratio': variance_ratio,
        'components': components
    }
    
    if inplace:
        # Store embeddings
        if not hasattr(spatioloji_obj, '_embeddings'):
            spatioloji_obj._embeddings = {}
        
        spatioloji_obj._embeddings[output_key] = X_pca
        spatioloji_obj._embeddings[f'{output_key}_variance'] = variance
        spatioloji_obj._embeddings[f'{output_key}_variance_ratio'] = variance_ratio
        
        # Also add first few PCs to cell_meta for easy access
        n_pcs_to_add = min(50, n_comps_actual)
        for i in range(n_pcs_to_add):
            spatioloji_obj._cell_meta[f'PC{i+1}'] = X_pca[:, i]
        
        return None
    else:
        return results


def tsne(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    n_components: int = 2,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: Union[float, str] = 'auto',
    n_iter: int = 1000,
    random_state: int = 42,
    output_key: str = 'X_tsne',
    inplace: bool = True
):
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding).
    
    Non-linear dimensionality reduction for visualization. Preserves local
    structure of data. Typically used for 2D/3D visualization.
    
    Parameters
    ----------
    spatioloji_obj : spatioloji
        Spatioloji object with expression data
    use_pca : bool, optional
        Run on PCA coordinates instead of raw expression, by default True
        Highly recommended for speed and noise reduction
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
        Higher values = more global structure
    early_exaggeration : float, optional
        Controls tightness of clusters, by default 12.0
    learning_rate : float or 'auto', optional
        Learning rate for optimization, by default 'auto'
    n_iter : int, optional
        Number of iterations, by default 1000
    random_state : int, optional
        Random seed, by default 42
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
    >>> # Standard t-SNE on PCA coordinates (recommended)
    >>> sp.processing.pca(sp, n_comps=50)
    >>> sp.processing.tsne(sp, use_pca=True, perplexity=30)
    >>> 
    >>> # 3D t-SNE
    >>> sp.processing.tsne(sp, n_components=3, perplexity=50)
    >>> 
    >>> # Access results
    >>> tsne_coords = sp.embeddings['X_tsne']
    """
    from sklearn.manifold import TSNE
    
    print(f"\nt-SNE (n_components={n_components}, perplexity={perplexity})")
    
    # Get input data
    if use_pca:
        # Use PCA coordinates
        if hasattr(spatioloji_obj, '_embeddings') and 'X_pca' in spatioloji_obj._embeddings:
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs as input")
        else:
            print(f"  No PCA found, computing PCA first...")
            pca(spatioloji_obj, n_comps=n_pcs, layer=layer, 
                use_highly_variable=use_highly_variable, inplace=True)
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
    else:
        # Use expression data
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
    
    # Run t-SNE
    print(f"  Running t-SNE (this may take a few minutes)...")
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        verbose=0
    )
    
    X_tsne = tsne_model.fit_transform(X)
    
    print(f"  ✓ t-SNE complete")
    print(f"    Final KL divergence: {tsne_model.kl_divergence_:.3f}")
    
    if inplace:
        # Store embeddings
        if not hasattr(spatioloji_obj, '_embeddings'):
            spatioloji_obj._embeddings = {}
        
        spatioloji_obj._embeddings[output_key] = X_tsne
        
        # Add to cell_meta for easy access
        for i in range(n_components):
            spatioloji_obj._cell_meta[f'tSNE{i+1}'] = X_tsne[:, i]
        
        return None
    else:
        return X_tsne


def umap(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42,
    output_key: str = 'X_umap',
    inplace: bool = True
):
    """
    UMAP (Uniform Manifold Approximation and Projection).
    
    Non-linear dimensionality reduction that preserves both local and global
    structure. Generally faster than t-SNE and better at preserving global
    structure. Requires `umap-learn` package.
    
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
        Higher values = more global structure
    min_dist : float, optional
        Minimum distance between points (0-1), by default 0.1
        Lower values = tighter clusters
    metric : str, optional
        Distance metric, by default 'euclidean'
    random_state : int, optional
        Random seed, by default 42
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
    >>> # Standard UMAP on PCA coordinates (recommended)
    >>> sp.processing.pca(sp, n_comps=50)
    >>> sp.processing.umap(sp, use_pca=True, n_neighbors=15)
    >>> 
    >>> # UMAP with more global structure
    >>> sp.processing.umap(sp, n_neighbors=30, min_dist=0.5)
    >>> 
    >>> # 3D UMAP
    >>> sp.processing.umap(sp, n_components=3)
    >>> 
    >>> # Access results
    >>> umap_coords = sp.embeddings['X_umap']
    """
    try:
        import umap as umap_package
    except ImportError:
        raise ImportError(
            "UMAP requires umap-learn package. "
            "Install with: pip install umap-learn"
        )
    
    print(f"\nUMAP (n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist})")
    
    # Get input data
    if use_pca:
        # Use PCA coordinates
        if hasattr(spatioloji_obj, '_embeddings') and 'X_pca' in spatioloji_obj._embeddings:
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs as input")
        else:
            print(f"  No PCA found, computing PCA first...")
            pca(spatioloji_obj, n_comps=n_pcs, layer=layer,
                use_highly_variable=use_highly_variable, inplace=True)
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
    else:
        # Use expression data
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
    
    # Run UMAP
    print(f"  Running UMAP...")
    umap_model = umap_package.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    X_umap = umap_model.fit_transform(X)
    
    print(f"  ✓ UMAP complete")
    
    if inplace:
        # Store embeddings
        if not hasattr(spatioloji_obj, '_embeddings'):
            spatioloji_obj._embeddings = {}
        
        spatioloji_obj._embeddings[output_key] = X_umap
        
        # Add to cell_meta for easy access
        for i in range(n_components):
            spatioloji_obj._cell_meta[f'UMAP{i+1}'] = X_umap[:, i]
        
        return None
    else:
        return X_umap


def diffusion_map(
    spatioloji_obj,
    use_pca: bool = True,
    n_pcs: int = 50,
    layer: Optional[str] = 'log_normalized',
    use_highly_variable: bool = True,
    n_components: int = 10,
    n_neighbors: int = 30,
    random_state: int = 42,
    output_key: str = 'X_diffmap',
    inplace: bool = True
):
    """
    Diffusion map for capturing continuous trajectories.
    
    Useful for trajectory inference and capturing gradual transitions.
    Preserves the diffusion distance which represents similarity.
    
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
    >>> sp.processing.pca(sp, n_comps=50)
    >>> sp.processing.diffusion_map(sp, n_components=10, n_neighbors=30)
    >>> 
    >>> # Access results
    >>> diffmap_coords = sp.embeddings['X_diffmap']
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    print(f"\nDiffusion Map (n_components={n_components}, n_neighbors={n_neighbors})")
    
    # Get input data
    if use_pca:
        if hasattr(spatioloji_obj, '_embeddings') and 'X_pca' in spatioloji_obj._embeddings:
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
            print(f"  Using first {n_pcs} PCs as input")
        else:
            print(f"  No PCA found, computing PCA first...")
            pca(spatioloji_obj, n_comps=n_pcs, layer=layer,
                use_highly_variable=use_highly_variable, inplace=True)
            X = spatioloji_obj._embeddings['X_pca'][:, :n_pcs]
    else:
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
    
    n_cells = X.shape[0]
    
    # Build k-NN graph
    print(f"  Building k-NN graph...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Construct affinity matrix (Gaussian kernel)
    print(f"  Computing diffusion operator...")
    sigma = np.median(distances[:, -1])  # Adaptive bandwidth
    
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    weights = np.exp(-distances.flatten()**2 / (2 * sigma**2))
    
    W = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))
    W = (W + W.T) / 2  # Symmetrize
    
    # Normalize to get diffusion operator
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(D)))
    L = D_inv_sqrt @ W @ D_inv_sqrt
    
    # Compute eigenvectors
    print(f"  Computing eigenvectors...")
    n_components_actual = min(n_components + 1, n_cells - 2)
    eigenvalues, eigenvectors = eigsh(L, k=n_components_actual, which='LM')
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Skip first eigenvector (trivial)
    X_diffmap = eigenvectors[:, 1:n_components+1]
    
    print(f"  ✓ Diffusion map complete")
    print(f"    Top 5 eigenvalues: {eigenvalues[1:6]}")
    
    if inplace:
        if not hasattr(spatioloji_obj, '_embeddings'):
            spatioloji_obj._embeddings = {}
        
        spatioloji_obj._embeddings[output_key] = X_diffmap
        spatioloji_obj._embeddings[f'{output_key}_eigenvalues'] = eigenvalues
        
        # Add to cell_meta
        for i in range(min(n_components, X_diffmap.shape[1])):
            spatioloji_obj._cell_meta[f'DC{i+1}'] = X_diffmap[:, i]
        
        return None
    else:
        return X_diffmap


def plot_pca_variance(
    spatioloji_obj,
    n_pcs: int = 50,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
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
    
    if not hasattr(spatioloji_obj, '_embeddings') or 'X_pca_variance_ratio' not in spatioloji_obj._embeddings:
        raise ValueError("No PCA results found. Run pca() first.")
    
    variance_ratio = spatioloji_obj._embeddings['X_pca_variance_ratio']
    n_pcs = min(n_pcs, len(variance_ratio))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot variance explained per PC
    ax1.bar(range(1, n_pcs + 1), variance_ratio[:n_pcs] * 100)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained per PC')
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative variance
    cumsum = np.cumsum(variance_ratio[:n_pcs]) * 100
    ax2.plot(range(1, n_pcs + 1), cumsum, 'b-', linewidth=2)
    ax2.axhline(y=80, color='r', linestyle='--', label='80% variance')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90% variance')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    pc_80 = np.where(cumsum >= 80)[0][0] + 1 if any(cumsum >= 80) else n_pcs
    pc_90 = np.where(cumsum >= 90)[0][0] + 1 if any(cumsum >= 90) else n_pcs
    
    print(f"\nPCA Variance Summary:")
    print(f"  PCs needed for 80% variance: {pc_80}")
    print(f"  PCs needed for 90% variance: {pc_90}")
    print(f"  Total variance (first {n_pcs} PCs): {cumsum[-1]:.1f}%")