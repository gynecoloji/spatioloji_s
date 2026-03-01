"""
qc.py - Quality control module for spatioloji objects

Provides comprehensive QC functionality that works with the new
spatioloji data structure, including cell-level, gene-level, and
FOV-level quality control.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatioloji_s.data.core import spatioloji
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


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
    gene_filter_method: str = 'percentile'           # NEW: 'percentile', 'absolute', or 'min_cells'
    gene_percentile_threshold: float = 50            # For 'percentile' method
    gene_absolute_threshold: float | None = None  # For 'absolute' method
    gene_min_cells: int | None = None             # For 'min_cells' method

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

    Provides modular QC functions that work directly with the
    spatioloji data structure. Results are stored in the expression
    metadata (cell_meta and gene_meta).

    Key Features:
    - Works directly with spatioloji expression and metadata
    - Modular QC steps (run individually or as pipeline)
    - Results stored in spatioloji.cell_meta and spatioloji.gene_meta
    - Returns filtered spatioloji object
    """

    def __init__(self, sp, config: QCConfig | None = None):
        """
        Initialize QC module with a spatioloji object.

        Parameters
        ----------
        sp : spatioloji
            The spatioloji object to perform QC on
        config : QCConfig, optional
            Configuration for QC thresholds
        """
        from .core import spatioloji

        if not isinstance(sp, spatioloji):
            raise TypeError("Input must be a spatioloji object")

        self.sp = sp
        self.config = config or QCConfig()

        # Validate input
        self._validate_input()

        # Initialize QC tracking
        self.qc_metrics = {}

        # Add gene annotations if not present
        self._prepare_gene_annotations()

        # Calculate QC metrics
        self._calculate_qc_metrics()

    def _validate_input(self) -> None:
        """Validate spatioloji object has required components."""
        if self.sp.expression is None:
            raise ValueError("spatioloji must have expression data")

        if self.sp.n_cells == 0:
            raise ValueError("spatioloji has no cells")

        if self.sp.n_genes == 0:
            raise ValueError("spatioloji has no genes")

    def _prepare_gene_annotations(self) -> None:
        """Add gene annotations for QC."""
        gene_meta = self.sp.gene_meta

        # Mitochondrial genes
        if 'mt' not in gene_meta.columns:
            gene_meta['mt'] = self.sp.gene_index.str.startswith("MT-")

        # Ribosomal genes
        if 'ribo' not in gene_meta.columns:
            gene_meta['ribo'] = [name.startswith(("RPS", "RPL"))
                                 for name in self.sp.gene_index]

        # Negative probe genes
        if 'NegProbe' not in gene_meta.columns:
            gene_meta['NegProbe'] = self.sp.gene_index.str.startswith("Neg")

    def _calculate_qc_metrics(self) -> None:
        """Calculate basic QC metrics."""
        print("Calculating QC metrics...")

        # Get expression matrix
        expr = self.sp.expression.get_dense()

        # Total counts per cell
        total_counts = expr.sum(axis=1)
        self.sp.cell_meta['total_counts'] = total_counts

        # Number of genes detected per cell
        self.sp.cell_meta['n_genes_by_counts'] = (expr > 0).sum(axis=1)

        # Percentage counts in mitochondrial genes
        mt_genes = self.sp.gene_meta['mt'].values
        if mt_genes.any():
            mt_counts = expr[:, mt_genes].sum(axis=1)
            # FIX: Use np.divide with where parameter
            self.sp.cell_meta['pct_counts_mt'] = np.divide(
                mt_counts * 100,
                total_counts,
                out=np.zeros_like(mt_counts, dtype=float),
                where=total_counts != 0
            )
        else:
            self.sp.cell_meta['pct_counts_mt'] = 0.0

        # Percentage counts in ribosomal genes
        ribo_genes = self.sp.gene_meta['ribo'].values
        if ribo_genes.any():
            ribo_counts = expr[:, ribo_genes].sum(axis=1)
            # FIX: Safe division
            self.sp.cell_meta['pct_counts_ribo'] = np.divide(
                ribo_counts * 100,
                total_counts,
                out=np.zeros_like(ribo_counts, dtype=float),
                where=total_counts != 0
            )
        else:
            self.sp.cell_meta['pct_counts_ribo'] = 0.0

        # Percentage counts in negative probes
        neg_genes = self.sp.gene_meta['NegProbe'].values
        if neg_genes.any():
            neg_counts = expr[:, neg_genes].sum(axis=1)
            # FIX: Safe division
            self.sp.cell_meta['pct_counts_NegProbe'] = np.divide(
                neg_counts * 100,
                total_counts,
                out=np.zeros_like(neg_counts, dtype=float),
                where=total_counts != 0
            )
        else:
            self.sp.cell_meta['pct_counts_NegProbe'] = 0.0

        print(f"  ✓ Calculated metrics for {self.sp.n_cells} cells")

    # ========== Statistical Tests ==========

    @staticmethod
    def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> int:
        """
        Perform Grubbs test to detect outliers.

        Parameters
        ----------
        data : np.ndarray
            Data to test
        alpha : float
            Significance level

        Returns
        -------
        int
            Index of outlier or -1 if none detected
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
            plt.savefig(
                os.path.join(self.config.analysis_dir, filename),
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()
        else:
            plt.show()

    # ========== Negative Probe QC ==========

    def qc_negative_probes(self, plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on negative probes using Grubbs test.

        Parameters
        ----------
        plot : bool
            Whether to create visualization

        Returns
        -------
        pd.DataFrame
            QC results
        """
        print("\n[QC] Negative Probes")

        # Get negative probe genes
        neg_genes = self.sp.gene_meta['NegProbe'].values

        if not neg_genes.any():
            print("  ⚠ No negative probes found")
            return pd.DataFrame()

        # Get counts
        expr = self.sp.expression.get_dense()
        neg_counts = expr[:, neg_genes].sum(axis=1)

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
        }, index=self.sp.cell_index)

        if idx_neg != -1:
            outlier_cell = self.sp.cell_index[idx_neg]
            results.loc[outlier_cell, 'is_outlier'] = True
            self.sp.cell_meta['QC_NegProbe_outlier'] = results['is_outlier'].values
            print(f"  ✗ Detected outlier: {outlier_cell}")
        else:
            self.sp.cell_meta['QC_NegProbe_outlier'] = False
            print("  ✓ No outliers detected")

        # Plot
        if plot:
            self._plot_distribution(
                np.log1p(neg_counts),
                title='Negative Probe Counts (log1p)',
                xlabel='log1p(Counts)',
                filename='QC_NegProbe_log1p.png',
                outlier_idx=idx_neg if idx_neg != -1 else None
            )

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
            QC results
        """
        print("\n[QC] Cell Area")

        if area_column not in self.sp.cell_meta.columns:
            print(f"  ⚠ '{area_column}' not found in cell_meta")
            return pd.DataFrame()

        # Get areas (already aligned to cell_index)
        areas = self.sp.cell_meta[area_column].values

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
        }, index=self.sp.cell_index)

        if idx_area != -1:
            outlier_cell = self.sp.cell_index[idx_area]
            results.loc[outlier_cell, 'is_outlier'] = True
            self.sp.cell_meta['QC_Area_outlier'] = results['is_outlier'].values
            print(f"  ✗ Detected outlier: {outlier_cell}")
        else:
            self.sp.cell_meta['QC_Area_outlier'] = False
            print("  ✓ No outliers detected")

        # Plot
        if plot:
            self._plot_distribution(
                np.log1p(areas),
                title='Cell Area (log1p)',
                xlabel='log1p(Area)',
                filename='QC_cell_area_log1p.png',
                outlier_idx=idx_area if idx_area != -1 else None
            )

        self.qc_metrics['cell_area'] = results
        return results

    def _plot_distribution(self, data: np.ndarray, title: str,
                          xlabel: str, filename: str,
                          outlier_idx: int | None = None) -> None:
        """Plot distribution with optional outlier marking."""
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

        Parameters
        ----------
        plot : bool
            Whether to create visualizations

        Returns
        -------
        pd.DataFrame
            Cell metrics summary
        """
        print("\n[QC] Cell Metrics")

        # Calculate ratio
        self.sp.cell_meta['ratio_counts_genes'] = (
            self.sp.cell_meta['total_counts'] /
            self.sp.cell_meta['n_genes_by_counts']
        )

        # Metrics to analyze
        metrics = [
            'ratio_counts_genes',
            'total_counts',
            'pct_counts_mt',
            'pct_counts_NegProbe'
        ]

        results = self.sp.cell_meta[metrics].copy()

        # Plot
        if plot:
            self._plot_cell_metrics(results, metrics)

        self.qc_metrics['cell_metrics'] = results

        print("\n  Metrics Summary:")
        print(results.describe())

        return results

    def _plot_cell_metrics(self, data: pd.DataFrame, metrics: list[str]) -> None:
        """Plot cell metrics distributions."""
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

            # Add threshold lines
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
                    custom_filters: dict | None = None,
                    return_mask: bool = False) -> pd.Series:
        """
        Filter cells based on QC metrics.

        Parameters
        ----------
        custom_filters : dict, optional
            Custom filters: {metric: (operator, threshold)}
        return_mask : bool
            If True, return boolean mask

        Returns
        -------
        pd.Series
            Boolean mask of cells passing QC
        """
        print("\n[QC] Filtering Cells")

        # Default filters
        mask = (
            (self.sp.cell_meta['pct_counts_NegProbe'] < self.config.pct_counts_neg_max) &
            (self.sp.cell_meta['pct_counts_mt'] < self.config.pct_counts_mt_max) &
            (self.sp.cell_meta['ratio_counts_genes'] > self.config.ratio_counts_genes_min) &
            (self.sp.cell_meta['total_counts'] > self.config.total_counts_min)
        )

        # Add area filter if available
        if 'QC_Area_outlier' in self.sp.cell_meta.columns:
            mask &= ~self.sp.cell_meta['QC_Area_outlier']

        # Apply custom filters
        if custom_filters:
            for metric, (operator, threshold) in custom_filters.items():
                if metric not in self.sp.cell_meta.columns:
                    print(f"  ⚠ Metric '{metric}' not found, skipping")
                    continue

                if operator == '>':
                    mask &= (self.sp.cell_meta[metric] > threshold)
                elif operator == '<':
                    mask &= (self.sp.cell_meta[metric] < threshold)
                elif operator == '>=':
                    mask &= (self.sp.cell_meta[metric] >= threshold)
                elif operator == '<=':
                    mask &= (self.sp.cell_meta[metric] <= threshold)

        # Store mask
        self.sp.cell_meta['QC_pass'] = mask

        # Report
        n_before = len(mask)
        n_after = mask.sum()
        pct_kept = (n_after / n_before) * 100

        print(f"  Before: {n_before:,} cells")
        print(f"  After:  {n_after:,} cells")
        print(f"  Kept:   {pct_kept:.1f}%")

        self.qc_metrics['cell_filtering'] = {
            'n_before': n_before,
            'n_after': n_after,
            'pct_kept': pct_kept,
            'filter_mask': mask
        }

        return mask if return_mask else mask

    def get_filtered_cell_ids(self) -> list[str]:
        """Get list of cell IDs that passed QC."""
        if 'QC_pass' not in self.sp.cell_meta.columns:
            print("  ⚠ Run filter_cells() first")
            return []

        mask = self.sp.cell_meta['QC_pass']
        return self.sp.cell_index[mask].tolist()

    # ========== FOV Metrics ==========

    def qc_fov_metrics(self, plot: bool = True) -> pd.DataFrame:
        """
        Perform QC on FOV-level metrics.

        Parameters
        ----------
        plot : bool
            Whether to create visualizations

        Returns
        -------
        pd.DataFrame
            FOV metrics summary
        """
        print("\n[QC] FOV Metrics")

        fov_col = self.sp.config.fov_id_col

        if fov_col not in self.sp.cell_meta.columns:
            print(f"  ⚠ '{fov_col}' not in cell_meta")
            return pd.DataFrame()

        # Get FOVs
        fovs = self.sp.cell_meta[fov_col]

        # Calculate transcripts per cell
        tx_per_cell = self.sp.cell_meta['total_counts']

        # FOV statistics
        fov_stats = pd.DataFrame({
            'n_cells': fovs.value_counts(),
            'avg_transcripts': fovs.groupby(fovs).apply(
                lambda x: tx_per_cell.loc[x.index].mean()
            ),
            'median_transcripts': fovs.groupby(fovs).apply(
                lambda x: tx_per_cell.loc[x.index].median()
            )
        })

        # Plot
        if plot:
            self._plot_fov_metrics(fovs, tx_per_cell, fov_stats)

        # Save
        if self.config.save_plots:
            fov_stats.to_csv(
                os.path.join(self.config.data_dir, 'fov_metrics.csv')
            )

        self.qc_metrics['fov_metrics'] = fov_stats

        print("\n  FOV Summary:")
        print(fov_stats)

        return fov_stats

    def _plot_fov_metrics(self, fovs: pd.Series,
                         tx_per_cell: pd.Series,
                         fov_stats: pd.DataFrame) -> None:
        """Plot FOV metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Transcripts per cell by FOV
        df_plot = pd.DataFrame({
            'fov': fovs,
            'transcripts': tx_per_cell
        })

        sns.boxplot(data=df_plot, x='fov', y='transcripts', ax=axes[0])
        axes[0].set_title('Transcripts per Cell by FOV')
        axes[0].set_xlabel('FOV')
        axes[0].set_ylabel('Transcripts per Cell')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Cell counts by FOV
        axes[1].bar(fov_stats.index, fov_stats['n_cells'])
        axes[1].set_title('Number of Cells by FOV')
        axes[1].set_xlabel('FOV')
        axes[1].set_ylabel('Number of Cells')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self._save_plot('QC_fov_metrics.png')

    # ========== Gene Filtering ==========

    def filter_genes(self,
                    method: str | None = None,
                    threshold: float | None = None,
                    min_cells: int | None = None,
                    plot: bool = True) -> pd.Series:
        """
        Filter genes based on expression.

        Parameters
        ----------
        method : str, optional
            'percentile', 'absolute', or 'min_cells'
            If None, uses config.gene_filter_method
        threshold : float, optional
            Threshold value. If None, uses config values
        min_cells : int, optional
            Minimum cells expressing gene. If None, uses config.gene_min_cells
        plot : bool
            Whether to create visualizations

        Returns
        -------
        pd.Series
            Boolean mask of genes passing filter
        """
        # Use config defaults if not specified
        method = method or self.config.gene_filter_method

        print(f"\n[QC] Filtering Genes (method={method})")

        expr = self.sp.expression.get_dense()
        gene_mask = ~self.sp.gene_meta['NegProbe'].values
        neg_mask = self.sp.gene_meta['NegProbe'].values

        gene_counts = expr[:, gene_mask].sum(axis=0)
        neg_counts = expr[:, neg_mask].sum(axis=0)

        # Apply filter method
        if method == 'percentile':
            # Use provided threshold or config default
            percentile = threshold if threshold is not None else self.config.gene_percentile_threshold
            neg_threshold = np.percentile(neg_counts.sum(), percentile)
            keep_genes_bool = gene_counts > neg_threshold
            description = f"{percentile}th percentile of neg probes"

        elif method == 'absolute':
            # Use provided threshold or config default
            abs_threshold = threshold if threshold is not None else self.config.gene_absolute_threshold
            if abs_threshold is None:
                raise ValueError(
                    "threshold required for absolute method. "
                    "Provide via parameter or set config.gene_absolute_threshold"
                )
            keep_genes_bool = gene_counts > abs_threshold
            description = f"absolute threshold of {abs_threshold}"

        elif method == 'min_cells':
            # Use provided min_cells or config default
            min_cells_threshold = min_cells if min_cells is not None else self.config.gene_min_cells
            if min_cells_threshold is None:
                raise ValueError(
                    "min_cells required for min_cells method. "
                    "Provide via parameter or set config.gene_min_cells"
                )
            n_cells_expressing = (expr[:, gene_mask] > 0).sum(axis=0)
            keep_genes_bool = n_cells_expressing > min_cells_threshold
            description = f"expressed in >{min_cells_threshold} cells"
        else:
            raise ValueError(f"Unknown method: {method}. Use 'percentile', 'absolute', or 'min_cells'")

        # Create full mask (include all negative probes)
        full_mask = np.zeros(self.sp.n_genes, dtype=bool)
        full_mask[gene_mask] = keep_genes_bool
        full_mask[neg_mask] = True  # Keep all negative probes

        # Store
        self.sp.gene_meta['QC_gene_pass'] = full_mask

        # Report
        n_before = gene_mask.sum()
        n_after = keep_genes_bool.sum()
        pct_kept = (n_after / n_before) * 100

        print(f"  Filter: {description}")
        print(f"  Before: {n_before:,} genes")
        print(f"  After:  {n_after:,} genes ({pct_kept:.1f}%)")

        self.qc_metrics['gene_filtering'] = {
            'n_before': n_before,
            'n_after': n_after,
            'pct_kept': pct_kept,
            'method': method
        }

        # Plot
        if plot:
            self._plot_gene_filtering(gene_counts, neg_counts, keep_genes_bool)

        return full_mask

    def _plot_gene_filtering(self, gene_counts: np.ndarray,
                            neg_counts: np.ndarray,
                            keep_genes: np.ndarray) -> None:
        """Plot gene filtering results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Distribution
        axes[0].hist(np.log1p(gene_counts), bins=50,
                    color='skyblue', alpha=0.7,
                    label='All genes', edgecolor='black')
        axes[0].hist(np.log1p(gene_counts[keep_genes]), bins=50,
                    color='green', alpha=0.5,
                    label='Kept genes', edgecolor='black')
        axes[0].set_title('Gene Expression Distribution')
        axes[0].set_xlabel('log1p(Total Counts)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Comparison
        gene_sum = gene_counts.sum()
        neg_sum = neg_counts.sum()
        kept_sum = gene_counts[keep_genes].sum()

        bars = axes[1].bar(
            ['Negative Probes', 'All Genes', 'Kept Genes'],
            [neg_sum, gene_sum, kept_sum],
            color=['red', 'skyblue', 'green'],
            alpha=0.7
        )

        axes[1].set_title('Total Counts Comparison')
        axes[1].set_ylabel('Total Counts')
        axes[1].grid(alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom')

        plt.tight_layout()
        self._save_plot('QC_gene_filtering.png')

    def get_filtered_gene_names(self) -> list[str]:
        """Get list of gene names that passed QC."""
        if 'QC_gene_pass' not in self.sp.gene_meta.columns:
            print("  ⚠ Run filter_genes() first")
            return []

        mask = self.sp.gene_meta['QC_gene_pass']
        return self.sp.gene_index[mask].tolist()

    # ========== Pipeline & Apply ==========

    def run_qc_pipeline(self,
                       qc_steps: list[str] | None = None,
                       plot: bool = True) -> spatioloji:
        """
        Run complete QC pipeline.

        Parameters
        ----------
        qc_steps : list, optional
            QC steps to run. If None, runs all.
        plot : bool
            Whether to create visualizations

        Returns
        -------
        spatioloji
            Filtered spatioloji object
        """
        print("\n" + "="*70)
        print("QC PIPELINE")
        print("="*70)

        if qc_steps is None:
            qc_steps = [
                'negative_probes',
                'cell_area',
                'cell_metrics',
                'fov_metrics',
                'filter_cells',
                'filter_genes'
            ]

        # Run steps
        if 'negative_probes' in qc_steps:
            self.qc_negative_probes(plot=plot)

        if 'cell_area' in qc_steps:
            self.qc_cell_area(plot=plot)

        if 'cell_metrics' in qc_steps:
            self.qc_cell_metrics(plot=plot)

        if 'fov_metrics' in qc_steps:
            self.qc_fov_metrics(plot=plot)

        if 'filter_cells' in qc_steps:
            self.filter_cells()

        if 'filter_genes' in qc_steps:
            self.filter_genes(plot=plot)

        # Create summary
        self.summarize_qc(plot=plot)

        # Apply filters
        filtered_sp = self.apply_filters()

        print("="*70)
        print("✓ QC PIPELINE COMPLETED")
        print("="*70 + "\n")

        return filtered_sp

    def apply_filters(self,
                     filter_cells: bool = True,
                     filter_genes: bool = True) -> spatioloji:
        """
        Apply QC filters to create filtered spatioloji object.

        Parameters
        ----------
        filter_cells : bool
            Whether to filter cells
        filter_genes : bool
            Whether to filter genes

        Returns
        -------
        spatioloji
            New filtered spatioloji object
        """
        print("\nApplying filters to spatioloji...")

        # Get filtered IDs
        if filter_cells:
            cell_ids = self.get_filtered_cell_ids()
            if not cell_ids:
                print("  ⚠ No cells passed filters, returning original")
                return self.sp
        else:
            cell_ids = self.sp.cell_index.tolist()

        if filter_genes:
            gene_names = self.get_filtered_gene_names()
            if not gene_names:
                print("  ⚠ No genes passed filters, returning original")
                return self.sp
        else:
            gene_names = self.sp.gene_index.tolist()

        # Subset
        filtered_sp = self.sp.subset_by_cells(cell_ids)
        filtered_sp = filtered_sp.subset_by_genes(gene_names)

        # Add QC metadata
        filtered_sp.cell_meta['qc_filtered'] = True
        filtered_sp.gene_meta['qc_filtered'] = True

        print("\n✓ Filtered spatioloji:")
        print(f"  Cells: {filtered_sp.n_cells:,}")
        print(f"  Genes: {filtered_sp.n_genes:,}")
        print(f"  FOVs:  {filtered_sp.n_fovs}")

        return filtered_sp

    def summarize_qc(self, plot: bool = True) -> dict:
        """
        Create comprehensive QC summary.

        Parameters
        ----------
        plot : bool
            Whether to create summary plots

        Returns
        -------
        dict
            QC summary statistics
        """
        summary = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config.__dict__,
            'metrics': {}
        }

        # Cell filtering
        if 'cell_filtering' in self.qc_metrics:
            cf = self.qc_metrics['cell_filtering']
            summary['metrics']['cells'] = {
                'before': cf['n_before'],
                'after': cf['n_after'],
                'pct_kept': cf['pct_kept']
            }

        # Gene filtering
        if 'gene_filtering' in self.qc_metrics:
            gf = self.qc_metrics['gene_filtering']
            summary['metrics']['genes'] = {
                'before': gf['n_before'],
                'after': gf['n_after'],
                'pct_kept': gf['pct_kept'],
                'method': gf['method']
            }

        # FOV metrics
        if 'fov_metrics' in self.qc_metrics:
            fov_stats = self.qc_metrics['fov_metrics']
            summary['metrics']['fovs'] = {
                'n_fovs': len(fov_stats),
                'total_cells': int(fov_stats['n_cells'].sum()),
                'avg_cells_per_fov': float(fov_stats['n_cells'].mean()),
                'avg_transcripts': float(fov_stats['avg_transcripts'].mean())
            }

        # Plot
        if plot:
            self._plot_qc_summary(summary)

        # Save - USE CUSTOM CONVERTER
        if self.config.save_plots:
            import json
            with open(os.path.join(self.config.data_dir, 'qc_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2, default=self._json_converter)

        # Print
        self._print_qc_summary(summary)

        return summary


    def _json_converter(self, obj):
        """
        Convert numpy/pandas types to JSON-serializable types.

        Parameters
        ----------
        obj : any
            Object to convert

        Returns
        -------
        any
            JSON-serializable version of obj
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _plot_qc_summary(self, summary: dict) -> None:
        """Plot QC summary."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Cells
        if 'cells' in summary['metrics']:
            cd = summary['metrics']['cells']
            bars = axes[0].bar(
                ['Before QC', 'After QC'],
                [cd['before'], cd['after']],
                color=['lightcoral', 'lightgreen'],
                edgecolor='black'
            )
            axes[0].set_title('Cell Filtering')
            axes[0].set_ylabel('Number of Cells')
            axes[0].grid(alpha=0.3, axis='y')

            for i, bar in enumerate(bars):
                height = bar.get_height()
                label = f'{int(height):,}\n({cd["pct_kept"]:.1f}%)' if i == 1 else f'{int(height):,}'
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom')

        # Genes
        if 'genes' in summary['metrics']:
            gd = summary['metrics']['genes']
            bars = axes[1].bar(
                ['Before QC', 'After QC'],
                [gd['before'], gd['after']],
                color=['lightcoral', 'lightgreen'],
                edgecolor='black'
            )
            axes[1].set_title('Gene Filtering')
            axes[1].set_ylabel('Number of Genes')
            axes[1].grid(alpha=0.3, axis='y')

            for i, bar in enumerate(bars):
                height = bar.get_height()
                label = f'{int(height):,}\n({gd["pct_kept"]:.1f}%)' if i == 1 else f'{int(height):,}'
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom')

        plt.tight_layout()
        self._save_plot('QC_summary.png')

    def _print_qc_summary(self, summary: dict) -> None:
        """Print QC summary."""
        print("\n" + "="*70)
        print("QC SUMMARY")
        print("="*70)

        if 'cells' in summary['metrics']:
            cd = summary['metrics']['cells']
            print("\nCells:")
            print(f"  Before: {cd['before']:,}")
            print(f"  After:  {cd['after']:,} ({cd['pct_kept']:.1f}%)")

        if 'genes' in summary['metrics']:
            gd = summary['metrics']['genes']
            print("\nGenes:")
            print(f"  Before: {gd['before']:,}")
            print(f"  After:  {gd['after']:,} ({gd['pct_kept']:.1f}%)")
            print(f"  Method: {gd['method']}")

        if 'fovs' in summary['metrics']:
            fd = summary['metrics']['fovs']
            print("\nFOVs:")
            print(f"  Count: {fd['n_fovs']}")
            print(f"  Avg cells/FOV: {fd['avg_cells_per_fov']:.1f}")
            print(f"  Avg transcripts: {fd['avg_transcripts']:.1f}")

        print("="*70)
