from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

@dataclass
class QCConfig:
    """Configuration for QC thresholds and parameters."""
    
    # Negative probe QC
    alpha_neg_probe: float = 0.01
    pct_counts_neg_max: float = 0.1
    
    # Cell area QC
    alpha_cell_area: float = 0.01
    
    # Cell metrics QC
    pct_counts_mt_max: float = 0.25
    ratio_counts_genes_min: float = 1.0
    total_counts_min: float = 20
    
    # Gene filtering
    gene_percentile_threshold: float = 50  # Percentile of negative probes
    
    # Output settings
    output_dir: str = "./output/"
    save_plots: bool = True
    
    def __post_init__(self):
        """Create output directories."""
        self.data_dir = os.path.join(self.output_dir, "data")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        if self.save_plots:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.analysis_dir, exist_ok=True)


class spatioloji_qc:
    """
    Quality control module for spatioloji objects.
    
    This class provides modular QC functions that can be run individually
    or as a complete pipeline. Results are stored in the spatioloji object
    and a filtered spatioloji object is returned.
    """
    
    def __init__(self, sp_obj, config: Optional[QCConfig] = None):
        """
        Initialize QC module with a spatioloji object.
        
        Parameters
        ----------
        sp_obj : spatioloji
            The spatioloji object to perform QC on
        config : QCConfig, optional
            Configuration for QC thresholds. If None, uses defaults.
        """
        self.sp = sp_obj
        self.config = config or QCConfig()
        
        # Validate that spatioloji has required data
        self._validate_input()
        
        # Initialize QC tracking
        self.qc_metrics = {}
        self.qc_passed = True
        
        # Prepare AnnData if not already prepared
        self._prepare_adata()
    
    def _validate_input(self) -> None:
        """Validate that spatioloji object has required components."""
        if self.sp.adata is None:
            raise ValueError("spatioloji object must have an AnnData object")
        if self.sp.cell_meta is None:
            raise ValueError("spatioloji object must have cell_meta")
    
    def _prepare_adata(self) -> None:
        """Prepare AnnData object with gene annotations."""
        adata = self.sp.adata
        
        # Add gene annotations if not already present
        if 'mt' not in adata.var.columns:
            adata.var['mt'] = adata.var_names.str.startswith("MT-")
        if 'ribo' not in adata.var.columns:
            adata.var['ribo'] = [name.startswith(("RPS", "RPL")) 
                                 for name in adata.var_names]
        if 'NegProbe' not in adata.var.columns:
            adata.var['NegProbe'] = adata.var_names.str.startswith("Neg")
        
        # Calculate QC metrics if not already present
        if 'total_counts' not in adata.obs.columns:
            import scanpy as sc
            sc.pp.calculate_qc_metrics(
                adata, 
                qc_vars=["mt", "ribo", "NegProbe"], 
                inplace=True, 
                log1p=True
            )
    
    # ========== Statistical Helper Methods ==========
    
    @staticmethod
    def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> int:
        """
        Perform Grubbs test to detect outliers.
        
        Parameters
        ----------
        data : np.ndarray
            Data to test for outliers
        alpha : float
            Significance level
        
        Returns
        -------
        int
            Index of outlier or -1 if no outlier detected
        """
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return -1
        
        # Calculate G statistic
        deviations = np.abs(data - mean)
        max_idx = np.argmax(deviations)
        G = deviations[max_idx] / std
        
        # Calculate critical G value
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        return max_idx if G > G_crit else -1
    
    def _save_plot(self, filename: str) -> None:
        """Save plot to analysis directory."""
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.analysis_dir, filename), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # ========== Negative Probe QC ==========
    
    def qc_negative_probes(self, plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on negative probes.
        
        Parameters
        ----------
        plot : bool
            Whether to create visualization
        
        Returns
        -------
        pd.DataFrame
            QC results with outlier information
        """
        adata = self.sp.adata
        
        # Get negative probe genes
        neg_probes = adata.var[adata.var['NegProbe']].index.tolist()
        
        if not neg_probes:
            print("Warning: No negative probes found")
            return pd.DataFrame()
        
        # Get counts for negative probes
        neg_probe_locs = [adata.var_names.get_loc(g) for g in neg_probes]
        counts = adata.X[:, neg_probe_locs]
        neg_counts = np.array(counts.sum(axis=1)).flatten()
        
        # Detect outliers
        idx_neg = self.grubbs_test(
            np.log1p(neg_counts), 
            alpha=self.config.alpha_neg_probe
        )
        
        # Store results
        results = pd.DataFrame({
            'neg_probe_counts': neg_counts,
            'log1p_neg_counts': np.log1p(neg_counts),
            'is_outlier': False
        }, index=adata.obs_names)
        
        if idx_neg != -1:
            outlier_cell = adata.obs_names[idx_neg]
            results.loc[outlier_cell, 'is_outlier'] = True
            adata.obs['QC_NegProbe_outlier'] = results['is_outlier']
            print(f"✗ Detected outlier cell for negative probes: {outlier_cell}")
        else:
            adata.obs['QC_NegProbe_outlier'] = False
            print("✓ No negative probe outliers detected")
        
        # Plot if requested
        if plot:
            self._plot_distribution(
                np.log1p(neg_counts),
                title='Negative Probe Counts (log1p)',
                xlabel='log1p(Counts)',
                filename='QC_NegProbe_log1p.png',
                outlier_idx=idx_neg if idx_neg != -1 else None
            )
        
        # Store in qc_metrics
        self.qc_metrics['negative_probes'] = results
        
        return results
    
    # ========== Cell Area QC ==========
    
    def qc_cell_area(self, area_column: str = 'Area', plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on cell area.
        
        Parameters
        ----------
        area_column : str
            Column name for cell area in cell_meta
        plot : bool
            Whether to create visualization
        
        Returns
        -------
        pd.DataFrame
            QC results with outlier information
        """
        cell_meta = self.sp.cell_meta
        adata = self.sp.adata
        
        if area_column not in cell_meta.columns:
            print(f"Warning: '{area_column}' not found in cell_meta")
            return pd.DataFrame()
        
        # Get cell areas aligned with adata
        cell_id_col = self.sp.config.cell_id_col
        area_data = cell_meta.set_index(cell_id_col)[area_column]
        
        # Align with adata observations
        if cell_id_col in adata.obs.columns:
            cell_ids = adata.obs[cell_id_col]
        else:
            cell_ids = adata.obs_names
        
        areas = area_data.loc[cell_ids].values
        
        # Detect outliers
        idx_area = self.grubbs_test(
            np.log1p(areas), 
            alpha=self.config.alpha_cell_area
        )
        
        # Store results
        results = pd.DataFrame({
            'cell_area': areas,
            'log1p_area': np.log1p(areas),
            'is_outlier': False
        }, index=adata.obs_names)
        
        if idx_area != -1:
            outlier_cell = adata.obs_names[idx_area]
            results.loc[outlier_cell, 'is_outlier'] = True
            adata.obs['QC_Area_outlier'] = results['is_outlier']
            print(f"✗ Detected outlier cell area: {outlier_cell}")
        else:
            adata.obs['QC_Area_outlier'] = False
            print("✓ No cell area outliers detected")
        
        # Plot if requested
        if plot:
            self._plot_distribution(
                np.log1p(areas),
                title='Cell Area (log1p)',
                xlabel='log1p(Area)',
                filename='QC_cell_area_log1p.png',
                outlier_idx=idx_area if idx_area != -1 else None
            )
        
        # Store in qc_metrics
        self.qc_metrics['cell_area'] = results
        
        return results
    
    def _plot_distribution(self, data: np.ndarray, title: str, 
                          xlabel: str, filename: str,
                          outlier_idx: Optional[int] = None) -> None:
        """Helper method to plot distributions with optional outlier marking."""
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        
        if outlier_idx is not None:
            plt.axvline(data[outlier_idx], color='red', 
                       linestyle='--', linewidth=2, 
                       label=f'Outlier: {data[outlier_idx]:.2f}')
            plt.legend()
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        self._save_plot(filename)
    
    # ========== Cell Metrics QC ==========
    
    def qc_cell_metrics(self, plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on cell-level metrics.
        
        Calculates and visualizes:
        - Ratio of counts to genes
        - Total counts
        - Mitochondrial percentage
        - Negative probe percentage
        
        Parameters
        ----------
        plot : bool
            Whether to create visualizations
        
        Returns
        -------
        pd.DataFrame
            Summary of cell metrics
        """
        adata = self.sp.adata
        
        # Calculate ratio of counts to genes
        adata.obs['ratio_counts_genes'] = (
            adata.obs['total_counts'] / adata.obs['n_genes_by_counts']
        )
        
        # Metrics to analyze
        metrics = [
            'ratio_counts_genes', 
            'total_counts', 
            'pct_counts_mt', 
            'pct_counts_NegProbe'
        ]
        
        # Get metrics data
        results = adata.obs[metrics].copy()
        
        # Plot distributions if requested
        if plot:
            self._plot_cell_metrics(results, metrics)
        
        # Store in qc_metrics
        self.qc_metrics['cell_metrics'] = results
        
        # Print summary statistics
        print("\n📊 Cell Metrics Summary:")
        print(results.describe())
        
        return results
    
    def _plot_cell_metrics(self, data: pd.DataFrame, metrics: List[str]) -> None:
        """Plot distributions for multiple cell metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.hist(data[metric], bins=50, color='skyblue', 
                   edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution of {metric}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
            
            # Add threshold lines if applicable
            if metric == 'pct_counts_mt':
                ax.axvline(self.config.pct_counts_mt_max, 
                          color='red', linestyle='--', 
                          label=f'Threshold: {self.config.pct_counts_mt_max}')
                ax.legend()
            elif metric == 'pct_counts_NegProbe':
                ax.axvline(self.config.pct_counts_neg_max, 
                          color='red', linestyle='--', 
                          label=f'Threshold: {self.config.pct_counts_neg_max}')
                ax.legend()
            elif metric == 'ratio_counts_genes':
                ax.axvline(self.config.ratio_counts_genes_min, 
                          color='red', linestyle='--', 
                          label=f'Threshold: {self.config.ratio_counts_genes_min}')
                ax.legend()
            elif metric == 'total_counts':
                ax.axvline(self.config.total_counts_min, 
                          color='red', linestyle='--', 
                          label=f'Threshold: {self.config.total_counts_min}')
                ax.legend()
        
        plt.tight_layout()
        self._save_plot('QC_cell_metrics.png')
    
    # ========== Cell Filtering ==========
    
    def filter_cells(self, 
                    custom_filters: Optional[dict] = None,
                    return_mask: bool = False) -> pd.DataFrame:
        """
        Filter cells based on QC metrics.
        
        Parameters
        ----------
        custom_filters : dict, optional
            Custom filter thresholds. Keys should be metric names,
            values should be tuples of (comparison, threshold).
            Example: {'total_counts': ('>', 50), 'pct_counts_mt': ('<', 0.2)}
        return_mask : bool
            If True, return boolean mask instead of filtered data
        
        Returns
        -------
        pd.DataFrame or pd.Series
            Filtered cell observations or boolean mask
        """
        adata = self.sp.adata
        
        # Default filters
        mask = (
            (adata.obs['pct_counts_NegProbe'] < self.config.pct_counts_neg_max) &
            (adata.obs['pct_counts_mt'] < self.config.pct_counts_mt_max) &
            (adata.obs['ratio_counts_genes'] > self.config.ratio_counts_genes_min) &
            (adata.obs['total_counts'] > self.config.total_counts_min) &
            (adata.obs['QC_Area_outlier'] == False)
        )
        
        # Apply custom filters if provided
        if custom_filters:
            for metric, (operator, threshold) in custom_filters.items():
                if metric not in adata.obs.columns:
                    print(f"Warning: Metric '{metric}' not found, skipping")
                    continue
                
                if operator == '>':
                    mask &= (adata.obs[metric] > threshold)
                elif operator == '<':
                    mask &= (adata.obs[metric] < threshold)
                elif operator == '>=':
                    mask &= (adata.obs[metric] >= threshold)
                elif operator == '<=':
                    mask &= (adata.obs[metric] <= threshold)
                elif operator == '==':
                    mask &= (adata.obs[metric] == threshold)
                elif operator == '!=':
                    mask &= (adata.obs[metric] != threshold)
        
        # Store filter mask
        adata.obs['QC_pass'] = mask
        
        if return_mask:
            return mask
        
        # Get filtered observations
        df_filtered = adata.obs[mask]
        
        # Print filtering summary
        n_before = len(adata.obs)
        n_after = len(df_filtered)
        pct_kept = (n_after / n_before) * 100
        
        print(f"\n🔍 Cell Filtering Results:")
        print(f"  Before: {n_before} cells")
        print(f"  After:  {n_after} cells")
        print(f"  Kept:   {pct_kept:.1f}%")
        print(f"  Removed: {n_before - n_after} cells ({100-pct_kept:.1f}%)")
        
        # Store in qc_metrics
        self.qc_metrics['cell_filtering'] = {
            'n_before': n_before,
            'n_after': n_after,
            'pct_kept': pct_kept,
            'filter_mask': mask
        }
        
        return df_filtered
    
    def get_filtered_cell_ids(self) -> List[str]:
        """
        Get list of cell IDs that passed QC filters.
        
        Returns
        -------
        List[str]
            Cell IDs that passed all QC filters
        """
        if 'QC_pass' not in self.sp.adata.obs.columns:
            print("Warning: Run filter_cells() first")
            return []
        
        mask = self.sp.adata.obs['QC_pass']
        
        cell_id_col = self.sp.config.cell_id_col
        if cell_id_col in self.sp.adata.obs.columns:
            return self.sp.adata.obs.loc[mask, cell_id_col].tolist()
        else:
            return self.sp.adata.obs_names[mask].tolist()
    
    # ========== FOV-Level QC ==========
    
    def qc_fov_metrics(self, plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on FOV-level metrics.
        
        Analyzes:
        - Average transcripts per cell by FOV
        - 90th percentile of gene expression by FOV
        - 50th percentile of negative probe expression by FOV
        
        Parameters
        ----------
        plot : bool
            Whether to create visualizations
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for each FOV
        """
        adata = self.sp.adata
        cell_meta = self.sp.cell_meta
        fov_col = self.sp.config.fov_id_col
        
        # Get FOV IDs for each cell
        fov_ids = self._get_fov_ids_for_cells()
        adata.obs['fov'] = fov_ids
        
        # Calculate transcripts per cell
        tx_per_cell = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['tx_per_cell'] = tx_per_cell
        
        # Calculate FOV-level statistics
        fov_summary = self._calculate_fov_statistics(adata, fov_ids)
        
        # Plot if requested
        if plot:
            self._plot_fov_metrics(adata, fov_summary)
        
        # Store in qc_metrics
        self.qc_metrics['fov_metrics'] = fov_summary
        
        # Save to file
        if self.config.save_plots:
            fov_summary.to_csv(
                os.path.join(self.config.data_dir, 'fov_metrics.csv')
            )
        
        print(f"\n📍 FOV Metrics Summary:")
        print(fov_summary)
        
        return fov_summary
    
    def _get_fov_ids_for_cells(self) -> List[str]:
        """Extract FOV IDs for each cell in adata."""
        adata = self.sp.adata
        cell_meta = self.sp.cell_meta
        cell_id_col = self.sp.config.cell_id_col
        fov_col = self.sp.config.fov_id_col
        
        # Get cell IDs from adata
        if cell_id_col in adata.obs.columns:
            cell_ids = adata.obs[cell_id_col]
        else:
            cell_ids = adata.obs_names
        
        # Map to FOV IDs
        cell_to_fov = cell_meta.set_index(cell_id_col)[fov_col].to_dict()
        fov_ids = [cell_to_fov.get(cid, 'Unknown') for cid in cell_ids]
        
        return fov_ids
    
    def _calculate_fov_statistics(self, adata, fov_ids: List[str]) -> pd.DataFrame:
        """Calculate comprehensive FOV-level statistics."""
        # Get gene and negative probe data
        gene_mask = ~adata.var['NegProbe']
        neg_mask = adata.var['NegProbe']
        
        gene_counts = np.array(adata.X[:, gene_mask].sum(axis=1)).flatten()
        neg_counts = np.array(adata.X[:, neg_mask].sum(axis=1)).flatten()
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'fov': fov_ids,
            'tx_per_cell': adata.obs['tx_per_cell'].values,
            'gene_counts': gene_counts,
            'neg_counts': neg_counts,
            'total_counts': adata.obs['total_counts'].values,
            'n_genes': adata.obs['n_genes_by_counts'].values
        })
        
        # Calculate statistics by FOV
        fov_stats = df.groupby('fov').agg({
            'tx_per_cell': ['mean', 'median', 'std'],
            'gene_counts': lambda x: np.percentile(x, 90),
            'neg_counts': lambda x: np.percentile(x, 50),
            'total_counts': ['mean', 'median'],
            'n_genes': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        fov_stats.columns = ['_'.join(col).strip() for col in fov_stats.columns.values]
        fov_stats = fov_stats.rename(columns={
            'gene_counts_<lambda>': '90_percentile_genes',
            'neg_counts_<lambda>': '50_percentile_neg'
        })
        
        # Add cell counts
        fov_stats['n_cells'] = df.groupby('fov').size()
        
        return fov_stats
    
    def _plot_fov_metrics(self, adata, fov_summary: pd.DataFrame) -> None:
        """Create visualizations for FOV-level metrics."""
        # Plot 1: Transcripts per cell by FOV
        fig, ax = plt.subplots(figsize=(12, 5))
        
        df_plot = pd.DataFrame({
            'fov': adata.obs['fov'],
            'tx_per_cell': adata.obs['tx_per_cell']
        })
        
        sns.boxplot(data=df_plot, x='fov', y='tx_per_cell', ax=ax)
        ax.set_title('Transcripts per Cell by FOV')
        ax.set_xlabel('FOV')
        ax.set_ylabel('Transcripts per Cell')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        self._save_plot('QC_fov_transcripts_per_cell.png')
        
        # Plot 2: 90th percentile genes vs 50th percentile negative probes
        fig, ax = plt.subplots(figsize=(12, 5))
        
        df_percentiles = fov_summary[['90_percentile_genes', '50_percentile_neg']].reset_index()
        df_melt = df_percentiles.melt(
            id_vars='fov', 
            var_name='metric', 
            value_name='counts'
        )
        
        sns.barplot(data=df_melt, x='fov', y='counts', hue='metric', ax=ax)
        ax.set_title('90th Percentile Genes vs 50th Percentile Negative Probes by FOV')
        ax.set_xlabel('FOV')
        ax.set_ylabel('Counts')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        self._save_plot('QC_fov_genes_vs_negprobes.png')
        
        # Plot 3: Cell counts by FOV
        fig, ax = plt.subplots(figsize=(12, 5))
        
        sns.barplot(data=fov_summary.reset_index(), x='fov', y='n_cells', ax=ax)
        ax.set_title('Number of Cells by FOV')
        ax.set_xlabel('FOV')
        ax.set_ylabel('Number of Cells')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        self._save_plot('QC_fov_cell_counts.png')
    
    def filter_fovs(self, 
                   min_cells: Optional[int] = None,
                   min_avg_transcripts: Optional[float] = None,
                   custom_filters: Optional[dict] = None) -> List[str]:
        """
        Filter FOVs based on quality metrics.
        
        Parameters
        ----------
        min_cells : int, optional
            Minimum number of cells per FOV
        min_avg_transcripts : float, optional
            Minimum average transcripts per cell
        custom_filters : dict, optional
            Custom filter thresholds for FOV metrics
        
        Returns
        -------
        List[str]
            FOV IDs that passed filters
        """
        if 'fov_metrics' not in self.qc_metrics:
            print("Warning: Run qc_fov_metrics() first")
            return []
        
        fov_summary = self.qc_metrics['fov_metrics']
        mask = pd.Series(True, index=fov_summary.index)
        
        # Apply filters
        if min_cells is not None:
            mask &= (fov_summary['n_cells'] >= min_cells)
        
        if min_avg_transcripts is not None:
            mask &= (fov_summary['tx_per_cell_mean'] >= min_avg_transcripts)
        
        if custom_filters:
            for metric, (operator, threshold) in custom_filters.items():
                if metric not in fov_summary.columns:
                    print(f"Warning: Metric '{metric}' not found, skipping")
                    continue
                
                if operator == '>':
                    mask &= (fov_summary[metric] > threshold)
                elif operator == '<':
                    mask &= (fov_summary[metric] < threshold)
                elif operator == '>=':
                    mask &= (fov_summary[metric] >= threshold)
                elif operator == '<=':
                    mask &= (fov_summary[metric] <= threshold)
        
        passed_fovs = fov_summary[mask].index.tolist()
        
        print(f"\n🔍 FOV Filtering Results:")
        print(f"  Before: {len(fov_summary)} FOVs")
        print(f"  After:  {len(passed_fovs)} FOVs")
        print(f"  Removed: {len(fov_summary) - len(passed_fovs)} FOVs")
        
        return passed_fovs
    
    # ========== Gene Filtering ==========
    
    def filter_genes(self, 
                    method: str = 'percentile',
                    threshold: Optional[float] = None,
                    min_cells: Optional[int] = None,
                    plot: bool = True) -> pd.DataFrame:
        """
        Filter genes based on expression compared to negative probes.
        
        Parameters
        ----------
        method : str
            Filtering method:
            - 'percentile': Keep genes above percentile of negative probes
            - 'absolute': Keep genes above absolute threshold
            - 'min_cells': Keep genes expressed in minimum number of cells
        threshold : float, optional
            Threshold value (percentile or absolute count)
        min_cells : int, optional
            Minimum number of cells expressing the gene
        plot : bool
            Whether to create visualizations
        
        Returns
        -------
        pd.DataFrame
            Gene filtering summary
        """
        adata = self.sp.adata
        
        # Separate genes and negative probes
        gene_mask = ~adata.var['NegProbe']
        neg_mask = adata.var['NegProbe']
        
        gene_counts = np.array(adata.X[:, gene_mask].sum(axis=0)).flatten()
        neg_counts = np.array(adata.X[:, neg_mask].sum(axis=0)).flatten()
        
        # Calculate threshold based on method
        if method == 'percentile':
            percentile = threshold or self.config.gene_percentile_threshold
            neg_threshold = np.percentile(neg_counts.sum(), percentile)
            keep_genes = gene_counts > neg_threshold
            filter_description = f"{percentile}th percentile of negative probes"
            
        elif method == 'absolute':
            if threshold is None:
                raise ValueError("threshold must be provided for absolute method")
            keep_genes = gene_counts > threshold
            filter_description = f"absolute threshold of {threshold}"
            
        elif method == 'min_cells':
            if min_cells is None:
                raise ValueError("min_cells must be provided for min_cells method")
            n_cells_expressing = (adata.X[:, gene_mask] > 0).sum(axis=0)
            keep_genes = np.array(n_cells_expressing > min_cells).flatten()
            filter_description = f"expressed in >{min_cells} cells"
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create summary DataFrame
        gene_names = adata.var_names[gene_mask]
        summary = pd.DataFrame({
            'gene': gene_names,
            'total_counts': gene_counts,
            'n_cells_expressing': (adata.X[:, gene_mask] > 0).sum(axis=0).A1,
            'pass_filter': keep_genes
        })
        
        n_before = len(gene_names)
        n_after = keep_genes.sum()
        pct_kept = (n_after / n_before) * 100
        
        print(f"\n🧬 Gene Filtering Results ({method}):")
        print(f"  Filter: {filter_description}")
        print(f"  Before: {n_before} genes")
        print(f"  After:  {n_after} genes")
        print(f"  Kept:   {pct_kept:.1f}%")
        print(f"  Removed: {n_before - n_after} genes ({100-pct_kept:.1f}%)")
        
        # Plot if requested
        if plot:
            self._plot_gene_filtering(summary, neg_counts, method)
        
        # Store results
        adata.var['QC_gene_pass'] = False
        adata.var.loc[gene_mask, 'QC_gene_pass'] = keep_genes
        
        self.qc_metrics['gene_filtering'] = {
            'n_before': n_before,
            'n_after': n_after,
            'pct_kept': pct_kept,
            'method': method,
            'summary': summary
        }
        
        # Save to file
        if self.config.save_plots:
            summary.to_csv(
                os.path.join(self.config.data_dir, 'gene_filtering_summary.csv'),
                index=False
            )
        
        return summary
    
    def _plot_gene_filtering(self, summary: pd.DataFrame, 
                            neg_counts: np.ndarray, 
                            method: str) -> None:
        """Create visualizations for gene filtering."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Gene count distribution
        ax = axes[0]
        ax.hist(np.log1p(summary['total_counts']), bins=50, 
               color='skyblue', edgecolor='black', alpha=0.7, 
               label='All genes')
        
        kept_genes = summary[summary['pass_filter']]
        ax.hist(np.log1p(kept_genes['total_counts']), bins=50,
               color='green', edgecolor='black', alpha=0.5,
               label='Kept genes')
        
        ax.set_title('Gene Expression Distribution (log1p)')
        ax.set_xlabel('log1p(Total Counts)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Comparison with negative probes
        ax = axes[1]
        gene_sum = summary['total_counts'].sum()
        neg_sum = neg_counts.sum()
        kept_sum = kept_genes['total_counts'].sum()
        
        bars = ax.bar(
            ['Negative Probes', 'All Genes', 'Kept Genes'],
            [neg_sum, gene_sum, kept_sum],
            color=['red', 'skyblue', 'green'],
            alpha=0.7
        )
        
        ax.set_title('Total Counts: Negative Probes vs Genes')
        ax.set_ylabel('Total Counts')
        ax.grid(alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot('QC_gene_filtering.png')
    
    def get_filtered_gene_names(self) -> List[str]:
        """
        Get list of gene names that passed QC filters.
        
        Returns
        -------
        List[str]
            Gene names that passed filters
        """
        if 'QC_gene_pass' not in self.sp.adata.var.columns:
            print("Warning: Run filter_genes() first")
            return []
        
        return self.sp.adata.var_names[self.sp.adata.var['QC_gene_pass']].tolist()
    
    # ========== QC Summary ==========
    
    def summarize_qc(self, plot: bool = True) -> dict:
        """
        Create comprehensive summary of all QC results.
        
        Parameters
        ----------
        plot : bool
            Whether to create summary visualizations
        
        Returns
        -------
        dict
            Dictionary containing all QC summaries
        """
        summary = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config.__dict__,
            'qc_metrics': {}
        }
        
        # Cell filtering summary
        if 'cell_filtering' in self.qc_metrics:
            cf = self.qc_metrics['cell_filtering']
            summary['qc_metrics']['cells'] = {
                'before': cf['n_before'],
                'after': cf['n_after'],
                'pct_kept': cf['pct_kept']
            }
        
        # Gene filtering summary
        if 'gene_filtering' in self.qc_metrics:
            gf = self.qc_metrics['gene_filtering']
            summary['qc_metrics']['genes'] = {
                'before': gf['n_before'],
                'after': gf['n_after'],
                'pct_kept': gf['pct_kept'],
                'method': gf['method']
            }
        
        # FOV summary
        if 'fov_metrics' in self.qc_metrics:
            fov_stats = self.qc_metrics['fov_metrics']
            summary['qc_metrics']['fovs'] = {
                'n_fovs': len(fov_stats),
                'total_cells': int(fov_stats['n_cells'].sum()),
                'avg_cells_per_fov': float(fov_stats['n_cells'].mean()),
                'avg_transcripts_per_cell': float(fov_stats['tx_per_cell_mean'].mean())
            }
        
        # Plot summary if requested
        if plot:
            self._plot_qc_summary(summary)
        
        # Save summary to file
        if self.config.save_plots:
            import json
            with open(os.path.join(self.config.data_dir, 'qc_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Print summary
        self._print_qc_summary(summary)
        
        return summary
    
    def _plot_qc_summary(self, summary: dict) -> None:
        """Create visual summary of QC results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Cell filtering
        if 'cells' in summary['qc_metrics']:
            ax = axes[0]
            cell_data = summary['qc_metrics']['cells']
            
            bars = ax.bar(
                ['Before QC', 'After QC'],
                [cell_data['before'], cell_data['after']],
                color=['lightcoral', 'lightgreen'],
                edgecolor='black'
            )
            
            ax.set_title('Cell Filtering Summary')
            ax.set_ylabel('Number of Cells')
            ax.grid(alpha=0.3, axis='y')
            
            # Add values and percentages
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i == 1:
                    label = f'{int(height):,}\n({cell_data["pct_kept"]:.1f}%)'
                else:
                    label = f'{int(height):,}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom')
        
        # Plot 2: Gene filtering
        if 'genes' in summary['qc_metrics']:
            ax = axes[1]
            gene_data = summary['qc_metrics']['genes']
            
            bars = ax.bar(
                ['Before QC', 'After QC'],
                [gene_data['before'], gene_data['after']],
                color=['lightcoral', 'lightgreen'],
                edgecolor='black'
            )
            
            ax.set_title('Gene Filtering Summary')
            ax.set_ylabel('Number of Genes')
            ax.grid(alpha=0.3, axis='y')
            
            # Add values and percentages
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if i == 1:
                    label = f'{int(height):,}\n({gene_data["pct_kept"]:.1f}%)'
                else:
                    label = f'{int(height):,}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_plot('QC_summary.png')
    
    def _print_qc_summary(self, summary: dict) -> None:
        """Print formatted QC summary."""
        print("\n" + "="*60)
        print("📊 QUALITY CONTROL SUMMARY")
        print("="*60)
        
        if 'cells' in summary['qc_metrics']:
            print("\n🔬 CELLS:")
            cd = summary['qc_metrics']['cells']
            print(f"  Before QC: {cd['before']:,} cells")
            print(f"  After QC:  {cd['after']:,} cells ({cd['pct_kept']:.1f}% kept)")
        
        if 'genes' in summary['qc_metrics']:
            print("\n🧬 GENES:")
            gd = summary['qc_metrics']['genes']
            print(f"  Before QC: {gd['before']:,} genes")
            print(f"  After QC:  {gd['after']:,} genes ({gd['pct_kept']:.1f}% kept)")
            print(f"  Method: {gd['method']}")
        
        if 'fovs' in summary['qc_metrics']:
            print("\n📍 FOVs:")
            fd = summary['qc_metrics']['fovs']
            print(f"  Number of FOVs: {fd['n_fovs']}")
            print(f"  Total cells: {fd['total_cells']:,}")
            print(f"  Avg cells per FOV: {fd['avg_cells_per_fov']:.1f}")
            print(f"  Avg transcripts per cell: {fd['avg_transcripts_per_cell']:.1f}")
        
        print("\n" + "="*60)
    # ========== Complete QC Pipeline ==========
    
    def run_qc_pipeline(self, 
                       qc_steps: Optional[List[str]] = None,
                       plot: bool = True) -> 'spatioloji':
        """
        Run complete QC pipeline and return filtered spatioloji object.
        
        Parameters
        ----------
        qc_steps : List[str], optional
            List of QC steps to run. If None, runs all steps.
            Options: ['negative_probes', 'cell_area', 'cell_metrics', 
                     'fov_metrics', 'filter_cells', 'filter_genes']
        plot : bool
            Whether to create visualizations
        
        Returns
        -------
        spatioloji
            New spatioloji object with filtered data
        """
        print("\n" + "="*60)
        print("🚀 STARTING QC PIPELINE")
        print("="*60)
        
        # Default: run all steps
        if qc_steps is None:
            qc_steps = [
                'negative_probes', 
                'cell_area', 
                'cell_metrics',
                'fov_metrics',
                'filter_cells', 
                'filter_genes'
            ]
        
        # Run QC steps
        if 'negative_probes' in qc_steps:
            print("\n[1/6] Running negative probe QC...")
            self.qc_negative_probes(plot=plot)
        
        if 'cell_area' in qc_steps:
            print("\n[2/6] Running cell area QC...")
            self.qc_cell_area(plot=plot)
        
        if 'cell_metrics' in qc_steps:
            print("\n[3/6] Running cell metrics QC...")
            self.qc_cell_metrics(plot=plot)
        
        if 'fov_metrics' in qc_steps:
            print("\n[4/6] Running FOV metrics QC...")
            self.qc_fov_metrics(plot=plot)
        
        if 'filter_cells' in qc_steps:
            print("\n[5/6] Filtering cells...")
            self.filter_cells()
        
        if 'filter_genes' in qc_steps:
            print("\n[6/6] Filtering genes...")
            self.filter_genes(plot=plot)
        
        # Create summary
        print("\n" + "-"*60)
        summary = self.summarize_qc(plot=plot)
        
        # Apply filters to spatioloji object
        filtered_sp = self.apply_filters()
        
        print("\n" + "="*60)
        print("✅ QC PIPELINE COMPLETED")
        print("="*60)
        
        return filtered_sp
    
    def apply_filters(self, 
                     filter_cells: bool = True,
                     filter_genes: bool = True,
                     filter_fovs: bool = False,
                     fov_ids: Optional[List[str]] = None) -> 'spatioloji':
        """
        Apply QC filters to create a new filtered spatioloji object.
        
        Parameters
        ----------
        filter_cells : bool
            Whether to filter cells based on QC
        filter_genes : bool
            Whether to filter genes based on QC
        filter_fovs : bool
            Whether to filter FOVs
        fov_ids : List[str], optional
            Specific FOV IDs to keep (overrides filter_fovs)
        
        Returns
        -------
        spatioloji
            New filtered spatioloji object
        """
        print("\n🔧 Applying filters to spatioloji object...")
        
        # Get filtered cell IDs
        if filter_cells:
            cell_ids = self.get_filtered_cell_ids()
            if not cell_ids:
                print("Warning: No cells passed filters. Returning original object.")
                return self.sp
        else:
            cell_id_col = self.sp.config.cell_id_col
            if cell_id_col in self.sp.adata.obs.columns:
                cell_ids = self.sp.adata.obs[cell_id_col].tolist()
            else:
                cell_ids = self.sp.adata.obs_names.tolist()
        
        # Subset by cells
        filtered_sp = self.sp.subset_by_cells(cell_ids)
        
        # Filter genes in adata
        if filter_genes:
            gene_names = self.get_filtered_gene_names()
            if gene_names:
                filtered_sp.adata = filtered_sp.adata[:, gene_names].copy()
                print(f"  ✓ Filtered to {len(gene_names)} genes")
            else:
                print("Warning: No genes passed filters. Keeping all genes.")
        
        # Filter FOVs if specified
        if fov_ids is not None:
            filtered_sp = filtered_sp.subset_by_fovs(fov_ids)
            print(f"  ✓ Filtered to {len(fov_ids)} FOVs")
        
        # Add QC information to custom data
        filtered_sp.add_custom('qc_summary', self.qc_metrics)
        filtered_sp.add_custom('qc_config', self.config)
        
        print(f"\n📦 Filtered spatioloji object created:")
        print(f"  Cells: {len(filtered_sp.cell_meta)}")
        print(f"  Genes: {filtered_sp.adata.shape[1]}")
        print(f"  FOVs: {len(filtered_sp.fov_positions)}")
        
        return filtered_sp
    
    # ========== Export Methods ==========
    
    def export_qc_report(self, filepath: str = None) -> None:
        """
        Export comprehensive QC report to HTML.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save HTML report. If None, uses default location.
        """
        if filepath is None:
            filepath = os.path.join(self.config.analysis_dir, 'qc_report.html')
        
        # Create HTML report
        html_content = self._generate_html_report()
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"\n📄 QC report saved to: {filepath}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML content for QC report."""
        summary = self.summarize_qc(plot=False)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spatioloji QC Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2980b9; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                img {{ max-width: 800px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🔬 Spatioloji QC Report</h1>
            <p><strong>Generated:</strong> {summary['timestamp']}</p>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Before QC</th><th>After QC</th><th>% Kept</th></tr>
        """
        
        if 'cells' in summary['qc_metrics']:
            cd = summary['qc_metrics']['cells']
            html += f"""
                <tr>
                    <td class="metric">Cells</td>
                    <td>{cd['before']:,}</td>
                    <td>{cd['after']:,}</td>
                    <td>{cd['pct_kept']:.1f}%</td>
                </tr>
            """
        
        if 'genes' in summary['qc_metrics']:
            gd = summary['qc_metrics']['genes']
            html += f"""
                <tr>
                    <td class="metric">Genes</td>
                    <td>{gd['before']:,}</td>
                    <td>{gd['after']:,}</td>
                    <td>{gd['pct_kept']:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>QC Plots</h2>
        """
        
        # Add images if they exist
        plot_files = [
            'QC_NegProbe_log1p.png',
            'QC_cell_area_log1p.png',
            'QC_cell_metrics.png',
            'QC_fov_transcripts_per_cell.png',
            'QC_gene_filtering.png',
            'QC_summary.png'
        ]
        
        for plot_file in plot_files:
            plot_path = os.path.join(self.config.analysis_dir, plot_file)
            if os.path.exists(plot_path):
                html += f'<h3>{plot_file.replace("_", " ").replace(".png", "")}</h3>\n'
                html += f'<img src="{plot_file}" alt="{plot_file}"><br>\n'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_filtered_data(self, 
                          output_dir: str = None,
                          save_adata: bool = True,
                          save_metadata: bool = True) -> None:
        """
        Save filtered data to files.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save files. If None, uses config output_dir.
        save_adata : bool
            Whether to save AnnData object
        save_metadata : bool
            Whether to save metadata files
        """
        if output_dir is None:
            output_dir = self.config.data_dir
        
        print(f"\n💾 Saving filtered data to: {output_dir}")
        
        # Get filtered spatioloji
        filtered_sp = self.apply_filters()
        
        # Save AnnData
        if save_adata:
            adata_path = os.path.join(output_dir, 'filtered_adata.h5ad')
            filtered_sp.adata.write_h5ad(adata_path)
            print(f"  ✓ Saved AnnData: {adata_path}")
        
        # Save metadata
        if save_metadata:
            cell_meta_path = os.path.join(output_dir, 'filtered_cell_meta.csv')
            filtered_sp.cell_meta.to_csv(cell_meta_path, index=False)
            print(f"  ✓ Saved cell metadata: {cell_meta_path}")
            
            polygons_path = os.path.join(output_dir, 'filtered_polygons.csv')
            filtered_sp.polygons.to_csv(polygons_path, index=False)
            print(f"  ✓ Saved polygons: {polygons_path}")
            
            fov_path = os.path.join(output_dir, 'filtered_fov_positions.csv')
            filtered_sp.fov_positions.to_csv(fov_path, index=False)
            print(f"  ✓ Saved FOV positions: {fov_path}")
        
        # Save as pickle
        pickle_path = os.path.join(output_dir, 'filtered_spatioloji.pkl')
        filtered_sp.to_pickle(pickle_path)
        print(f"  ✓ Saved spatioloji object: {pickle_path}")
    # ========== Helper/Utility Methods ==========
    
    def get_qc_status(self) -> dict:
        """
        Get current QC status showing which steps have been run.
        
        Returns
        -------
        dict
            Status of each QC step
        """
        status = {
            'negative_probes': 'negative_probes' in self.qc_metrics,
            'cell_area': 'cell_area' in self.qc_metrics,
            'cell_metrics': 'cell_metrics' in self.qc_metrics,
            'fov_metrics': 'fov_metrics' in self.qc_metrics,
            'cell_filtering': 'cell_filtering' in self.qc_metrics,
            'gene_filtering': 'gene_filtering' in self.qc_metrics
        }
        
        print("\n📋 QC Status:")
        for step, completed in status.items():
            symbol = "✓" if completed else "○"
            print(f"  {symbol} {step.replace('_', ' ').title()}")
        
        return status
    
    def compare_before_after(self) -> pd.DataFrame:
        """
        Compare metrics before and after QC filtering.
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        if 'QC_pass' not in self.sp.adata.obs.columns:
            print("Warning: Run filter_cells() first")
            return pd.DataFrame()
        
        adata = self.sp.adata
        before = adata.obs
        after = adata.obs[adata.obs['QC_pass']]
        
        metrics = [
            'total_counts', 
            'n_genes_by_counts',
            'pct_counts_mt',
            'pct_counts_NegProbe'
        ]
        
        comparison = pd.DataFrame({
            'Metric': metrics,
            'Before_Mean': [before[m].mean() for m in metrics],
            'After_Mean': [after[m].mean() for m in metrics],
            'Before_Median': [before[m].median() for m in metrics],
            'After_Median': [after[m].median() for m in metrics]
        })
        
        comparison['Change_%'] = (
            (comparison['After_Mean'] - comparison['Before_Mean']) / 
            comparison['Before_Mean'] * 100
        ).round(2)
        
        print("\n📊 Before vs After Comparison:")
        print(comparison.to_string(index=False))
        
        return comparison
    
    def plot_qc_overview(self) -> None:
        """Create comprehensive overview plot of all QC metrics."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        adata = self.sp.adata
        
        # Plot 1: Total counts distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(np.log1p(adata.obs['total_counts']), bins=50, 
                color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.log1p(self.config.total_counts_min), 
                   color='red', linestyle='--', label='Threshold')
        ax1.set_title('Total Counts (log1p)')
        ax1.set_xlabel('log1p(Counts)')
        ax1.legend()
        
        # Plot 2: MT percentage
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(adata.obs['pct_counts_mt'], bins=50,
                color='salmon', edgecolor='black', alpha=0.7)
        ax2.axvline(self.config.pct_counts_mt_max,
                   color='red', linestyle='--', label='Threshold')
        ax2.set_title('Mitochondrial %')
        ax2.set_xlabel('% MT Counts')
        ax2.legend()
        
        # Plot 3: Negative probe percentage
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(adata.obs['pct_counts_NegProbe'], bins=50,
                color='lightcoral', edgecolor='black', alpha=0.7)
        ax3.axvline(self.config.pct_counts_neg_max,
                   color='red', linestyle='--', label='Threshold')
        ax3.set_title('Negative Probe %')
        ax3.set_xlabel('% Neg Counts')
        ax3.legend()
        
        # Plot 4: Ratio counts to genes
        ax4 = fig.add_subplot(gs[1, 0])
        if 'ratio_counts_genes' in adata.obs.columns:
            ax4.hist(adata.obs['ratio_counts_genes'], bins=50,
                    color='lightgreen', edgecolor='black', alpha=0.7)
            ax4.axvline(self.config.ratio_counts_genes_min,
                       color='red', linestyle='--', label='Threshold')
            ax4.set_title('Ratio: Counts/Genes')
            ax4.set_xlabel('Ratio')
            ax4.legend()
        
        # Plot 5: Cells per FOV
        ax5 = fig.add_subplot(gs[1, 1])
        if 'fov' in adata.obs.columns:
            fov_counts = adata.obs['fov'].value_counts().sort_index()
            ax5.bar(range(len(fov_counts)), fov_counts.values,
                   color='plum', edgecolor='black', alpha=0.7)
            ax5.set_title('Cells per FOV')
            ax5.set_xlabel('FOV')
            ax5.set_ylabel('Cell Count')
        
        # Plot 6: Gene expression distribution
        ax6 = fig.add_subplot(gs[1, 2])
        gene_sums = np.array(adata.X.sum(axis=0)).flatten()
        ax6.hist(np.log1p(gene_sums), bins=50,
                color='wheat', edgecolor='black', alpha=0.7)
        ax6.set_title('Gene Expression (log1p)')
        ax6.set_xlabel('log1p(Total Counts)')
        
        # Plot 7: QC pass/fail
        ax7 = fig.add_subplot(gs[2, 0])
        if 'QC_pass' in adata.obs.columns:
            qc_counts = adata.obs['QC_pass'].value_counts()
            ax7.pie(qc_counts.values, labels=['Failed', 'Passed'],
                   colors=['lightcoral', 'lightgreen'],
                   autopct='%1.1f%%', startangle=90)
            ax7.set_title('Cell QC Status')
        
        # Plot 8: Scatter: total counts vs genes
        ax8 = fig.add_subplot(gs[2, 1])
        scatter_data = adata.obs.sample(min(1000, len(adata.obs)))
        ax8.scatter(scatter_data['n_genes_by_counts'],
                   scatter_data['total_counts'],
                   alpha=0.3, s=10, c='steelblue')
        ax8.set_xlabel('N Genes')
        ax8.set_ylabel('Total Counts')
        ax8.set_title('Counts vs Genes')
        ax8.set_yscale('log')
        
        # Plot 9: Gene QC pass/fail
        ax9 = fig.add_subplot(gs[2, 2])
        if 'QC_gene_pass' in adata.var.columns:
            gene_qc = adata.var['QC_gene_pass'].value_counts()
            ax9.pie(gene_qc.values, labels=['Failed', 'Passed'],
                   colors=['lightcoral', 'lightgreen'],
                   autopct='%1.1f%%', startangle=90)
            ax9.set_title('Gene QC Status')
        
        plt.suptitle('Spatioloji QC Overview', fontsize=16, y=0.995)
        self._save_plot('QC_comprehensive_overview.png')