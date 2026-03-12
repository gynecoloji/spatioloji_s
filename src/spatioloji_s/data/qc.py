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
    gene_filter_method: str = "percentile"  # NEW: 'percentile', 'absolute', or 'min_cells'
    gene_percentile_threshold: float = 50  # For 'percentile' method
    gene_absolute_threshold: float | None = None  # For 'absolute' method
    gene_min_cells: int | None = None  # For 'min_cells' method

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
        if "mt" not in gene_meta.columns:
            gene_meta["mt"] = self.sp.gene_index.str.startswith("MT-")

        # Ribosomal genes
        if "ribo" not in gene_meta.columns:
            gene_meta["ribo"] = [name.startswith(("RPS", "RPL")) for name in self.sp.gene_index]

        # Negative probe genes
        if "NegProbe" not in gene_meta.columns:
            gene_meta["NegProbe"] = self.sp.gene_index.str.startswith("Neg")

    def _calculate_qc_metrics(self) -> None:
        """Calculate basic QC metrics."""
        print("Calculating QC metrics...")

        # Get expression matrix
        expr = self.sp.expression.get_dense()

        # Total counts per cell
        total_counts = expr.sum(axis=1)
        self.sp.cell_meta["total_counts"] = total_counts

        # Number of genes detected per cell
        self.sp.cell_meta["n_genes_by_counts"] = (expr > 0).sum(axis=1)

        # Percentage counts in mitochondrial genes
        mt_genes = self.sp.gene_meta["mt"].values
        if mt_genes.any():
            mt_counts = expr[:, mt_genes].sum(axis=1)
            # FIX: Use np.divide with where parameter
            self.sp.cell_meta["pct_counts_mt"] = np.divide(
                mt_counts * 100, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts != 0
            )
        else:
            self.sp.cell_meta["pct_counts_mt"] = 0.0

        # Percentage counts in ribosomal genes
        ribo_genes = self.sp.gene_meta["ribo"].values
        if ribo_genes.any():
            ribo_counts = expr[:, ribo_genes].sum(axis=1)
            # FIX: Safe division
            self.sp.cell_meta["pct_counts_ribo"] = np.divide(
                ribo_counts * 100, total_counts, out=np.zeros_like(ribo_counts, dtype=float), where=total_counts != 0
            )
        else:
            self.sp.cell_meta["pct_counts_ribo"] = 0.0

        # Percentage counts in negative probes
        neg_genes = self.sp.gene_meta["NegProbe"].values
        if neg_genes.any():
            neg_counts = expr[:, neg_genes].sum(axis=1)
            # FIX: Safe division
            self.sp.cell_meta["pct_counts_NegProbe"] = np.divide(
                neg_counts * 100, total_counts, out=np.zeros_like(neg_counts, dtype=float), where=total_counts != 0
            )
        else:
            self.sp.cell_meta["pct_counts_NegProbe"] = 0.0

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
            plt.savefig(os.path.join(self.config.analysis_dir, filename), dpi=150, bbox_inches="tight")
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
        neg_genes = self.sp.gene_meta["NegProbe"].values

        if not neg_genes.any():
            print("  ⚠ No negative probes found")
            return pd.DataFrame()

        # Get counts
        expr = self.sp.expression.get_dense()
        neg_counts = expr[:, neg_genes].sum(axis=1)

        # Detect outliers
        idx_neg = self.grubbs_test(np.log1p(neg_counts), alpha=self.config.alpha_neg_probe)

        # Store results
        results = pd.DataFrame(
            {"neg_probe_counts": neg_counts, "log1p_neg_counts": np.log1p(neg_counts), "is_outlier": False},
            index=self.sp.cell_index,
        )

        if idx_neg != -1:
            outlier_cell = self.sp.cell_index[idx_neg]
            results.loc[outlier_cell, "is_outlier"] = True
            self.sp.cell_meta["QC_NegProbe_outlier"] = results["is_outlier"].values
            print(f"  ✗ Detected outlier: {outlier_cell}")
        else:
            self.sp.cell_meta["QC_NegProbe_outlier"] = False
            print("  ✓ No outliers detected")

        # Plot
        if plot:
            self._plot_distribution(
                np.log1p(neg_counts),
                title="Negative Probe Counts (log1p)",
                xlabel="log1p(Counts)",
                filename="QC_NegProbe_log1p.png",
                outlier_idx=idx_neg if idx_neg != -1 else None,
            )

        self.qc_metrics["negative_probes"] = results
        return results

    # ========== Cell Area QC ==========

    def qc_cell_area(self, area_column: str = "Area", plot: bool = True) -> pd.DataFrame:
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
        idx_area = self.grubbs_test(np.log1p(areas), alpha=self.config.alpha_cell_area)

        # Store results
        results = pd.DataFrame(
            {"cell_area": areas, "log1p_area": np.log1p(areas), "is_outlier": False}, index=self.sp.cell_index
        )

        if idx_area != -1:
            outlier_cell = self.sp.cell_index[idx_area]
            results.loc[outlier_cell, "is_outlier"] = True
            self.sp.cell_meta["QC_Area_outlier"] = results["is_outlier"].values
            print(f"  ✗ Detected outlier: {outlier_cell}")
        else:
            self.sp.cell_meta["QC_Area_outlier"] = False
            print("  ✓ No outliers detected")

        # Plot
        if plot:
            self._plot_distribution(
                np.log1p(areas),
                title="Cell Area (log1p)",
                xlabel="log1p(Area)",
                filename="QC_cell_area_log1p.png",
                outlier_idx=idx_area if idx_area != -1 else None,
            )

        self.qc_metrics["cell_area"] = results
        return results

    def qc_polygon_validity(self, plot: bool = True) -> pd.DataFrame:
        """
        Check validity of cell polygons using Shapely.

        Flags cells whose boundary geometry is degenerate (fewer than 3
        vertices, zero area, or self-intersecting / topologically invalid).
        Results are stored in ``sp.cell_meta["QC_polygon_valid"]``.

        Parameters
        ----------
        plot : bool
            Whether to create a summary bar chart of valid vs. invalid
            counts, broken down by failure reason.

        Returns
        -------
        pd.DataFrame
            Per-cell validity DataFrame indexed by cell ID with columns
            ``n_vertices``, ``area``, ``is_valid``, and ``shapely_reason``.
        """
        print("\n[QC] Polygon Validity")

        if self.sp.polygons is None:
            print("  ⚠ No polygon data attached — skipping")
            return pd.DataFrame()

        from shapely.geometry import Polygon
        from shapely.validation import explain_validity

        poly_df     = self.sp.polygons
        cell_id_col = self.sp.config.cell_id_col
        x_col       = self.sp.config.x_global_col
        y_col       = self.sp.config.y_global_col

        records: list[dict] = []
        for cell_id, grp in poly_df.groupby(cell_id_col):
            coords  = list(zip(grp[x_col].values, grp[y_col].values))
            n_verts = len(coords)
            if n_verts < 3:
                records.append({
                    "cell_id":       str(cell_id),
                    "n_vertices":    n_verts,
                    "area":          0.0,
                    "is_valid":      False,
                    "shapely_reason": "fewer than 3 vertices",
                })
                continue
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                reason = ""
                valid  = True
            elif not poly.is_valid:
                reason = explain_validity(poly)
                valid  = False
            elif poly.is_empty:
                reason = "empty geometry"
                valid  = False
            else:
                reason = "zero area"
                valid  = False
            records.append({
                "cell_id":        str(cell_id),
                "n_vertices":     n_verts,
                "area":           poly.area,
                "is_valid":       valid,
                "shapely_reason": reason,
            })

        results = pd.DataFrame(records).set_index("cell_id")

        # Align to the master cell index — cells missing from polygon data → False
        valid_series = results["is_valid"].reindex(self.sp.cell_index, fill_value=False)
        self.sp.cell_meta["QC_polygon_valid"] = valid_series.values

        n_total   = len(self.sp.cell_index)
        n_valid   = int(valid_series.sum())
        n_missing = n_total - len(results)
        n_invalid = n_total - n_valid
        print(f"  Total cells       : {n_total:,}")
        print(f"  Valid polygons    : {n_valid:,} ({n_valid / n_total * 100:.1f}%)")
        print(f"  Invalid polygons  : {n_invalid:,} ({n_invalid / n_total * 100:.1f}%)")
        if n_missing > 0:
            print(f"    Of which {n_missing:,} have no polygon data at all")

        if plot:
            self._plot_polygon_validity(results, n_valid, n_invalid)

        self.qc_metrics["polygon_validity"] = results
        return results

    def _plot_polygon_validity(
        self, results: pd.DataFrame, n_valid: int, n_invalid: int
    ) -> None:
        """Plot polygon validity summary (counts and failure reasons)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1 — valid vs invalid bar
        axes[0].bar(
            ["Valid", "Invalid"],
            [n_valid, n_invalid],
            color=["lightgreen", "lightcoral"],
            edgecolor="black",
        )
        axes[0].set_title("Polygon Validity")
        axes[0].set_ylabel("Number of Cells")
        axes[0].grid(alpha=0.3, axis="y")
        for i, v in enumerate([n_valid, n_invalid]):
            axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

        # Panel 2 — breakdown of reasons for invalid cells
        invalid_df = results[~results["is_valid"]]
        if len(invalid_df):
            reason_counts = invalid_df["shapely_reason"].value_counts()
            axes[1].barh(reason_counts.index, reason_counts.values,
                         color="lightcoral", edgecolor="black")
            axes[1].set_title("Invalid Polygon Failure Reasons")
            axes[1].set_xlabel("Number of Cells")
            for i, v in enumerate(reason_counts.values):
                axes[1].text(v, i, f" {v:,}", va="center")
        else:
            axes[1].text(0.5, 0.5, "No invalid polygons", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title("Invalid Polygon Failure Reasons")

        plt.tight_layout()
        self._save_plot("QC_polygon_validity.png")

    def _plot_distribution(
        self, data: np.ndarray, title: str, xlabel: str, filename: str, outlier_idx: int | None = None
    ) -> None:
        """Plot distribution with optional outlier marking."""
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

        if outlier_idx is not None:
            plt.axvline(
                data[outlier_idx], color="red", linestyle="--", linewidth=2, label=f"Outlier: {data[outlier_idx]:.2f}"
            )
            plt.legend()

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
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
        self.sp.cell_meta["ratio_counts_genes"] = (
            self.sp.cell_meta["total_counts"] / self.sp.cell_meta["n_genes_by_counts"]
        )

        # Metrics to analyze
        metrics = ["ratio_counts_genes", "total_counts", "pct_counts_mt", "pct_counts_NegProbe"]

        results = self.sp.cell_meta[metrics].copy()

        # Plot
        if plot:
            self._plot_cell_metrics(results, metrics)

        self.qc_metrics["cell_metrics"] = results

        print("\n  Metrics Summary:")
        print(results.describe())

        return results

    def _plot_cell_metrics(self, data: pd.DataFrame, metrics: list[str]) -> None:
        """Plot cell metrics distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.hist(data[metric], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
            ax.set_title(f"Distribution of {metric}")
            ax.set_xlabel(metric)
            ax.set_ylabel("Frequency")
            ax.grid(alpha=0.3)

            # Add threshold lines
            if metric == "pct_counts_mt":
                ax.axvline(
                    self.config.pct_counts_mt_max,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {self.config.pct_counts_mt_max}",
                )
                ax.legend()
            elif metric == "pct_counts_NegProbe":
                ax.axvline(
                    self.config.pct_counts_neg_max,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {self.config.pct_counts_neg_max}",
                )
                ax.legend()
            elif metric == "ratio_counts_genes":
                ax.axvline(
                    self.config.ratio_counts_genes_min,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {self.config.ratio_counts_genes_min}",
                )
                ax.legend()
            elif metric == "total_counts":
                ax.axvline(
                    self.config.total_counts_min,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {self.config.total_counts_min}",
                )
                ax.legend()

        plt.tight_layout()
        self._save_plot("QC_cell_metrics.png")

    # ========== Cell Filtering ==========

    def filter_cells(self, custom_filters: dict | None = None, return_mask: bool = False) -> pd.Series:
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
            (self.sp.cell_meta["pct_counts_NegProbe"] < self.config.pct_counts_neg_max)
            & (self.sp.cell_meta["pct_counts_mt"] < self.config.pct_counts_mt_max)
            & (self.sp.cell_meta["ratio_counts_genes"] > self.config.ratio_counts_genes_min)
            & (self.sp.cell_meta["total_counts"] > self.config.total_counts_min)
        )

        # Add area filter if available
        if "QC_Area_outlier" in self.sp.cell_meta.columns:
            mask &= ~self.sp.cell_meta["QC_Area_outlier"]

        # Add polygon validity filter if available
        if "QC_polygon_valid" in self.sp.cell_meta.columns:
            mask &= self.sp.cell_meta["QC_polygon_valid"]

        # Apply custom filters
        if custom_filters:
            for metric, (operator, threshold) in custom_filters.items():
                if metric not in self.sp.cell_meta.columns:
                    print(f"  ⚠ Metric '{metric}' not found, skipping")
                    continue

                if operator == ">":
                    mask &= self.sp.cell_meta[metric] > threshold
                elif operator == "<":
                    mask &= self.sp.cell_meta[metric] < threshold
                elif operator == ">=":
                    mask &= self.sp.cell_meta[metric] >= threshold
                elif operator == "<=":
                    mask &= self.sp.cell_meta[metric] <= threshold

        # Store mask
        self.sp.cell_meta["QC_pass"] = mask

        # Report
        n_before = len(mask)
        n_after = mask.sum()
        pct_kept = (n_after / n_before) * 100

        print(f"  Before: {n_before:,} cells")
        print(f"  After:  {n_after:,} cells")
        print(f"  Kept:   {pct_kept:.1f}%")
        if "QC_polygon_valid" in self.sp.cell_meta.columns:
            n_invalid_poly = int((~self.sp.cell_meta["QC_polygon_valid"]).sum())
            print(f"  Invalid polygon: {n_invalid_poly:,}")

        self.qc_metrics["cell_filtering"] = {
            "n_before": n_before,
            "n_after": n_after,
            "pct_kept": pct_kept,
            "filter_mask": mask,
        }

        return mask if return_mask else mask

    def get_filtered_cell_ids(self) -> list[str]:
        """Get list of cell IDs that passed QC."""
        if "QC_pass" not in self.sp.cell_meta.columns:
            print("  ⚠ Run filter_cells() first")
            return []

        mask = self.sp.cell_meta["QC_pass"]
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
        tx_per_cell = self.sp.cell_meta["total_counts"]

        # FOV statistics
        fov_stats = pd.DataFrame(
            {
                "n_cells": fovs.value_counts(),
                "avg_transcripts": fovs.groupby(fovs).apply(lambda x: tx_per_cell.loc[x.index].mean()),
                "median_transcripts": fovs.groupby(fovs).apply(lambda x: tx_per_cell.loc[x.index].median()),
            }
        )

        # Plot
        if plot:
            self._plot_fov_metrics(fovs, tx_per_cell, fov_stats)

        # Save
        if self.config.save_plots:
            fov_stats.to_csv(os.path.join(self.config.data_dir, "fov_metrics.csv"))

        self.qc_metrics["fov_metrics"] = fov_stats

        print("\n  FOV Summary:")
        print(fov_stats)

        return fov_stats

    def _plot_fov_metrics(self, fovs: pd.Series, tx_per_cell: pd.Series, fov_stats: pd.DataFrame) -> None:
        """Plot FOV metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Transcripts per cell by FOV
        df_plot = pd.DataFrame({"fov": fovs, "transcripts": tx_per_cell})

        sns.boxplot(data=df_plot, x="fov", y="transcripts", ax=axes[0])
        axes[0].set_title("Transcripts per Cell by FOV")
        axes[0].set_xlabel("FOV")
        axes[0].set_ylabel("Transcripts per Cell")
        axes[0].tick_params(axis="x", rotation=45)

        # Plot 2: Cell counts by FOV
        axes[1].bar(fov_stats.index, fov_stats["n_cells"])
        axes[1].set_title("Number of Cells by FOV")
        axes[1].set_xlabel("FOV")
        axes[1].set_ylabel("Number of Cells")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_plot("QC_fov_metrics.png")

    # ========== Gene Filtering ==========

    def filter_genes(
        self, method: str | None = None, threshold: float | None = None, min_cells: int | None = None, plot: bool = True
    ) -> pd.Series:
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
        gene_mask = ~self.sp.gene_meta["NegProbe"].values
        neg_mask = self.sp.gene_meta["NegProbe"].values

        gene_counts = expr[:, gene_mask].sum(axis=0)
        neg_counts = expr[:, neg_mask].sum(axis=0)

        # Apply filter method
        if method == "percentile":
            # Use provided threshold or config default
            percentile = threshold if threshold is not None else self.config.gene_percentile_threshold
            neg_threshold = np.percentile(neg_counts.sum(), percentile)
            keep_genes_bool = gene_counts > neg_threshold
            description = f"{percentile}th percentile of neg probes"

        elif method == "absolute":
            # Use provided threshold or config default
            abs_threshold = threshold if threshold is not None else self.config.gene_absolute_threshold
            if abs_threshold is None:
                raise ValueError(
                    "threshold required for absolute method. "
                    "Provide via parameter or set config.gene_absolute_threshold"
                )
            keep_genes_bool = gene_counts > abs_threshold
            description = f"absolute threshold of {abs_threshold}"

        elif method == "min_cells":
            # Use provided min_cells or config default
            min_cells_threshold = min_cells if min_cells is not None else self.config.gene_min_cells
            if min_cells_threshold is None:
                raise ValueError(
                    "min_cells required for min_cells method. Provide via parameter or set config.gene_min_cells"
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
        self.sp.gene_meta["QC_gene_pass"] = full_mask

        # Report
        n_before = gene_mask.sum()
        n_after = keep_genes_bool.sum()
        pct_kept = (n_after / n_before) * 100

        print(f"  Filter: {description}")
        print(f"  Before: {n_before:,} genes")
        print(f"  After:  {n_after:,} genes ({pct_kept:.1f}%)")

        self.qc_metrics["gene_filtering"] = {
            "n_before": n_before,
            "n_after": n_after,
            "pct_kept": pct_kept,
            "method": method,
        }

        # Plot
        if plot:
            self._plot_gene_filtering(gene_counts, neg_counts, keep_genes_bool)

        return full_mask

    def _plot_gene_filtering(self, gene_counts: np.ndarray, neg_counts: np.ndarray, keep_genes: np.ndarray) -> None:
        """Plot gene filtering results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Distribution
        axes[0].hist(np.log1p(gene_counts), bins=50, color="skyblue", alpha=0.7, label="All genes", edgecolor="black")
        axes[0].hist(
            np.log1p(gene_counts[keep_genes]), bins=50, color="green", alpha=0.5, label="Kept genes", edgecolor="black"
        )
        axes[0].set_title("Gene Expression Distribution")
        axes[0].set_xlabel("log1p(Total Counts)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Comparison
        gene_sum = gene_counts.sum()
        neg_sum = neg_counts.sum()
        kept_sum = gene_counts[keep_genes].sum()

        bars = axes[1].bar(
            ["Negative Probes", "All Genes", "Kept Genes"],
            [neg_sum, gene_sum, kept_sum],
            color=["red", "skyblue", "green"],
            alpha=0.7,
        )

        axes[1].set_title("Total Counts Comparison")
        axes[1].set_ylabel("Total Counts")
        axes[1].grid(alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height):,}", ha="center", va="bottom")

        plt.tight_layout()
        self._save_plot("QC_gene_filtering.png")

    def get_filtered_gene_names(self) -> list[str]:
        """Get list of gene names that passed QC."""
        if "QC_gene_pass" not in self.sp.gene_meta.columns:
            print("  ⚠ Run filter_genes() first")
            return []

        mask = self.sp.gene_meta["QC_gene_pass"]
        return self.sp.gene_index[mask].tolist()

    # ========== Pipeline & Apply ==========

    def run_qc_pipeline(self, qc_steps: list[str] | None = None, plot: bool = True) -> spatioloji:
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
        print("\n" + "=" * 70)
        print("QC PIPELINE")
        print("=" * 70)

        if qc_steps is None:
            qc_steps = [
                "negative_probes", "cell_area", "polygon_validity",
                "cell_metrics", "fov_metrics", "filter_cells", "filter_genes",
            ]

        # Run steps
        if "negative_probes" in qc_steps:
            self.qc_negative_probes(plot=plot)

        if "cell_area" in qc_steps:
            self.qc_cell_area(plot=plot)

        if "polygon_validity" in qc_steps:
            self.qc_polygon_validity(plot=plot)

        if "cell_metrics" in qc_steps:
            self.qc_cell_metrics(plot=plot)

        if "fov_metrics" in qc_steps:
            self.qc_fov_metrics(plot=plot)

        if "filter_cells" in qc_steps:
            self.filter_cells()

        if "filter_genes" in qc_steps:
            self.filter_genes(plot=plot)

        # Create summary
        self.summarize_qc(plot=plot)

        # Apply filters
        filtered_sp = self.apply_filters()

        print("=" * 70)
        print("✓ QC PIPELINE COMPLETED")
        print("=" * 70 + "\n")

        return filtered_sp

    def apply_filters(self, filter_cells: bool = True, filter_genes: bool = True) -> spatioloji:
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
        filtered_sp.cell_meta["qc_filtered"] = True
        filtered_sp.gene_meta["qc_filtered"] = True

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
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.__dict__,
            "metrics": {},
        }

        # Cell filtering
        if "cell_filtering" in self.qc_metrics:
            cf = self.qc_metrics["cell_filtering"]
            summary["metrics"]["cells"] = {"before": cf["n_before"], "after": cf["n_after"], "pct_kept": cf["pct_kept"]}

        # Gene filtering
        if "gene_filtering" in self.qc_metrics:
            gf = self.qc_metrics["gene_filtering"]
            summary["metrics"]["genes"] = {
                "before": gf["n_before"],
                "after": gf["n_after"],
                "pct_kept": gf["pct_kept"],
                "method": gf["method"],
            }

        # FOV metrics
        if "fov_metrics" in self.qc_metrics:
            fov_stats = self.qc_metrics["fov_metrics"]
            summary["metrics"]["fovs"] = {
                "n_fovs": len(fov_stats),
                "total_cells": int(fov_stats["n_cells"].sum()),
                "avg_cells_per_fov": float(fov_stats["n_cells"].mean()),
                "avg_transcripts": float(fov_stats["avg_transcripts"].mean()),
            }

        # Plot
        if plot:
            self._plot_qc_summary(summary)

        # Save - USE CUSTOM CONVERTER
        if self.config.save_plots:
            import json

            with open(os.path.join(self.config.data_dir, "qc_summary.json"), "w") as f:
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
        if isinstance(obj, np.integer | np.int64 | np.int32):
            return int(obj)
        elif isinstance(obj, np.floating | np.float64 | np.float32):
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
        if "cells" in summary["metrics"]:
            cd = summary["metrics"]["cells"]
            bars = axes[0].bar(
                ["Before QC", "After QC"],
                [cd["before"], cd["after"]],
                color=["lightcoral", "lightgreen"],
                edgecolor="black",
            )
            axes[0].set_title("Cell Filtering")
            axes[0].set_ylabel("Number of Cells")
            axes[0].grid(alpha=0.3, axis="y")

            for i, bar in enumerate(bars):
                height = bar.get_height()
                label = f"{int(height):,}\n({cd['pct_kept']:.1f}%)" if i == 1 else f"{int(height):,}"
                axes[0].text(bar.get_x() + bar.get_width() / 2.0, height, label, ha="center", va="bottom")

        # Genes
        if "genes" in summary["metrics"]:
            gd = summary["metrics"]["genes"]
            bars = axes[1].bar(
                ["Before QC", "After QC"],
                [gd["before"], gd["after"]],
                color=["lightcoral", "lightgreen"],
                edgecolor="black",
            )
            axes[1].set_title("Gene Filtering")
            axes[1].set_ylabel("Number of Genes")
            axes[1].grid(alpha=0.3, axis="y")

            for i, bar in enumerate(bars):
                height = bar.get_height()
                label = f"{int(height):,}\n({gd['pct_kept']:.1f}%)" if i == 1 else f"{int(height):,}"
                axes[1].text(bar.get_x() + bar.get_width() / 2.0, height, label, ha="center", va="bottom")

        plt.tight_layout()
        self._save_plot("QC_summary.png")

    def _print_qc_summary(self, summary: dict) -> None:
        """Print QC summary."""
        print("\n" + "=" * 70)
        print("QC SUMMARY")
        print("=" * 70)

        if "cells" in summary["metrics"]:
            cd = summary["metrics"]["cells"]
            print("\nCells:")
            print(f"  Before: {cd['before']:,}")
            print(f"  After:  {cd['after']:,} ({cd['pct_kept']:.1f}%)")

        if "genes" in summary["metrics"]:
            gd = summary["metrics"]["genes"]
            print("\nGenes:")
            print(f"  Before: {gd['before']:,}")
            print(f"  After:  {gd['after']:,} ({gd['pct_kept']:.1f}%)")
            print(f"  Method: {gd['method']}")

        if "fovs" in summary["metrics"]:
            fd = summary["metrics"]["fovs"]
            print("\nFOVs:")
            print(f"  Count: {fd['n_fovs']}")
            print(f"  Avg cells/FOV: {fd['avg_cells_per_fov']:.1f}")
            print(f"  Avg transcripts: {fd['avg_transcripts']:.1f}")

        print("=" * 70)


# ============================================================================
# Xenium-specific QC
# ============================================================================

_XENIUM_NEG_PREFIXES: tuple[str, ...] = (
    "NegControlProbe",
    "NegControlCodeword",
    "DeprecatedCodeword",
    "UnassignedCodeword",
    "Intergenic",
)


@dataclass
class XeniumQCConfig:
    """Configuration for Xenium-specific QC thresholds and parameters.

    Attributes
    ----------
    neg_prefixes : tuple[str, ...]
        Gene name prefixes that identify non-biological control features.
        Default covers all five Xenium control types.
    transcript_counts_min : int
        Minimum number of panel-gene transcripts per cell.
    n_features_min : int
        Minimum number of unique panel genes detected per cell.
    pct_counts_neg_max : float
        Maximum fraction (0–1) of counts from any control type.
        XOA alert thresholds: warning >= 0.04, error >= 0.10.
    pct_counts_mt_max : float
        Maximum fraction (0–1) of counts from mitochondrial genes.
    cell_area_min : float
        Minimum cell area (um2). Cells below this are likely debris.
    cell_area_max : float
        Maximum cell area (um2). Cells above this are likely doublets.
    nucleus_cell_ratio_max : float
        Maximum nucleus_area / cell_area ratio.
        Values > 1.0 indicate segmentation errors.
    gene_filter_method : str
        Gene-level filter: ``'percentile'``, ``'absolute'``, or ``'min_cells'``.
    gene_percentile_threshold : float
        Percentile of control-gene baseline used as expression cut-off
        when ``gene_filter_method='percentile'``.
    gene_min_cells : int or None
        Minimum number of cells expressing a gene
        when ``gene_filter_method='min_cells'``.
    output_dir : str
        Root directory for QC plots and data files.
    save_plots : bool
        Whether to save figures to disk (True) or display interactively.
    """

    # Negative-control gene prefixes (all five Xenium types)
    neg_prefixes: tuple[str, ...] = _XENIUM_NEG_PREFIXES

    # Cell-level thresholds
    transcript_counts_min: int = 10
    n_features_min: int = 5
    pct_counts_neg_max: float = 0.05        # XOA warning >= 0.04, error >= 0.10
    pct_counts_mt_max: float = 0.25
    cell_area_min: float = 20.0             # um2
    cell_area_max: float = 500.0            # um2
    nucleus_cell_ratio_max: float = 1.0     # nucleus / cell area

    # Gene-level filtering
    gene_filter_method: str = "percentile"
    gene_percentile_threshold: float = 50
    gene_min_cells: int | None = None

    # Output
    output_dir: str = "./xenium_qc_output/"
    save_plots: bool = True

    def __post_init__(self) -> None:
        """Create output sub-directories."""
        self.data_dir = os.path.join(self.output_dir, "data")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        if self.save_plots:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.analysis_dir, exist_ok=True)


class xenium_qc:
    """Quality control module for 10x Genomics Xenium spatioloji objects.

    Tailored to Xenium's coordinate system (um, no FOVs), its five
    negative-control gene prefixes, and its per-cell morphology columns
    (``cell_area``, ``nucleus_area``) from ``cells.csv.gz``.

    Differences from ``spatioloji_qc`` (CosMx / MERFISH):

    * No FOV-level QC -- Xenium is a single continuous coordinate space.
    * Five control-gene prefixes instead of one ``NegProbe`` prefix.
    * Cell-area thresholds are in um2 (not px2).
    * Nucleus : cell area ratio is checked as a segmentation quality metric.
    * Transcript density (transcripts / um2) is computed and visualised.

    Parameters
    ----------
    sp : spatioloji
        Object loaded with :meth:`spatioloji.from_xenium`.
    config : XeniumQCConfig, optional
        QC thresholds and settings.  Defaults created if not provided.

    Examples
    --------
    >>> sp = sj.spatioloji.from_xenium("path/to/xenium_outs")
    >>> qc = sj.xenium_qc(sp)
    >>> qc.qc_cell_area()
    >>> qc.qc_transcript_metrics()
    >>> qc.qc_control_metrics()
    >>> qc.qc_nucleus_cell_ratio()
    >>> qc.filter_cells()
    >>> qc.filter_genes()
    >>> sp_filtered = qc.apply_filters()
    """

    def __init__(self, sp: "spatioloji", config: XeniumQCConfig | None = None) -> None:
        """
        Initialise the Xenium QC module.

        Parameters
        ----------
        sp : spatioloji
            Xenium spatioloji object.
        config : XeniumQCConfig, optional
            QC configuration.  Defaults are applied if None.

        Raises
        ------
        TypeError
            If ``sp`` is not a :class:`spatioloji` instance.
        ValueError
            If ``sp`` has no cells or genes.
        """
        from .core import spatioloji as _spatioloji

        if not isinstance(sp, _spatioloji):
            raise TypeError("Input must be a spatioloji object")

        self.sp = sp
        self.config = config or XeniumQCConfig()
        self.qc_metrics: dict = {}

        if sp.n_cells == 0:
            raise ValueError("spatioloji has no cells")
        if sp.n_genes == 0:
            raise ValueError("spatioloji has no genes")

        self._prepare_gene_annotations()
        self._calculate_qc_metrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_gene_annotations(self) -> None:
        """Annotate genes with control-type and mitochondrial flags.

        Adds boolean columns to ``sp.gene_meta``:

        * ``is_neg_ctrl`` -- True for any of the five Xenium control prefixes.
        * ``neg_ctrl_type`` -- the matched prefix string (or ``''`` for panel genes).
        * ``is_mt`` -- True for mitochondrial genes (``MT-`` prefix).
        * ``is_<Prefix>`` -- per-prefix boolean for each control type.
        """
        gene_meta = self.sp.gene_meta
        gene_names = self.sp.gene_index.tolist()

        neg_type: list[str] = []
        for name in gene_names:
            matched = next((p for p in self.config.neg_prefixes if name.startswith(p)), "")
            neg_type.append(matched)

        gene_meta["neg_ctrl_type"] = neg_type
        gene_meta["is_neg_ctrl"] = gene_meta["neg_ctrl_type"] != ""

        for prefix in self.config.neg_prefixes:
            gene_meta[f"is_{prefix}"] = [n.startswith(prefix) for n in gene_names]

        if "is_mt" not in gene_meta.columns:
            gene_meta["is_mt"] = [n.startswith("MT-") for n in gene_names]

        n_ctrl  = gene_meta["is_neg_ctrl"].sum()
        n_panel = (~gene_meta["is_neg_ctrl"]).sum()
        print(f"  Gene annotations: {n_panel:,} panel genes, {n_ctrl:,} control features")
        for prefix in self.config.neg_prefixes:
            n = gene_meta[f"is_{prefix}"].sum()
            if n:
                print(f"    {prefix}: {n}")

    def _calculate_qc_metrics(self) -> None:
        """Compute all per-cell QC metrics from the expression matrix.

        Populates ``sp.cell_meta`` with:

        * ``total_counts_expr`` -- row sums of the full expression matrix.
        * ``transcript_counts_expr`` -- counts from panel genes only.
        * ``n_features_by_counts`` -- unique panel genes with >= 1 count.
        * ``pct_counts_neg`` -- fraction of counts from all control genes.
        * ``pct_counts_mt`` -- fraction of counts from mitochondrial genes.
        * ``pct_counts_<Prefix>`` -- per-prefix fractions.
        * ``nucleus_cell_ratio`` -- nucleus_area / cell_area (if columns available).
        * ``transcript_density`` -- transcript_counts / cell_area (transcripts / um2).
        """
        print("Calculating Xenium QC metrics...")

        import scipy.sparse as _sp_sparse

        # Use the sparse matrix throughout — never materialise a dense copy.
        # get_sparse() returns a scipy sparse matrix (CSR or equivalent).
        X = self.sp.expression.get_sparse()
        if not _sp_sparse.issparse(X):
            X = _sp_sparse.csr_matrix(X)
        elif X.format != "csr":
            X = X.tocsr()
        X.eliminate_zeros()  # drop any explicitly-stored zeros before nnz counts

        gm       = self.sp.gene_meta
        is_ctrl  = gm["is_neg_ctrl"].values
        is_panel = ~is_ctrl
        is_mt    = gm["is_mt"].values

        # Integer column indices — avoids repeated boolean fancy-indexing overhead
        panel_idx = np.where(is_panel)[0]
        ctrl_idx  = np.where(is_ctrl)[0]
        mt_idx    = np.where(is_mt)[0]

        # --- Row sums (O(nnz), no dense copy) ---
        total_counts_expr = np.asarray(X.sum(axis=1)).ravel()
        self.sp.cell_meta["total_counts_expr"] = total_counts_expr

        X_panel = X[:, panel_idx]  # sparse column slice; still sparse
        panel_counts = np.asarray(X_panel.sum(axis=1)).ravel()
        self.sp.cell_meta["transcript_counts_expr"] = panel_counts

        # n_features: nnz per row in the panel slice (after eliminate_zeros above)
        X_panel_csr = X_panel.tocsr()
        self.sp.cell_meta["n_features_by_counts"] = np.diff(X_panel_csr.indptr)

        safe_total = np.where(total_counts_expr > 0, total_counts_expr, 1.0)

        neg_counts = np.asarray(X[:, ctrl_idx].sum(axis=1)).ravel() if ctrl_idx.size else np.zeros(self.sp.n_cells)
        self.sp.cell_meta["pct_counts_neg"] = neg_counts / safe_total

        mt_counts = np.asarray(X[:, mt_idx].sum(axis=1)).ravel() if mt_idx.size else np.zeros(self.sp.n_cells)
        self.sp.cell_meta["pct_counts_mt"] = mt_counts / safe_total

        for prefix in self.config.neg_prefixes:
            col_flag = f"is_{prefix}"
            if col_flag in gm.columns and gm[col_flag].any():
                pfx_idx = np.where(gm[col_flag].values)[0]
                prefix_counts = np.asarray(X[:, pfx_idx].sum(axis=1)).ravel()
                self.sp.cell_meta[f"pct_counts_{prefix}"] = prefix_counts / safe_total

        if "cell_area" in self.sp.cell_meta.columns and "nucleus_area" in self.sp.cell_meta.columns:
            cell_area    = self.sp.cell_meta["cell_area"].values.astype(float)
            nucleus_area = self.sp.cell_meta["nucleus_area"].values.astype(float)
            safe_area    = np.where(cell_area > 0, cell_area, np.nan)
            self.sp.cell_meta["nucleus_cell_ratio"] = nucleus_area / safe_area
            tc_col = "transcript_counts" if "transcript_counts" in self.sp.cell_meta.columns \
                     else "transcript_counts_expr"
            self.sp.cell_meta["transcript_density"] = (
                self.sp.cell_meta[tc_col].values / safe_area
            )

        print(f"  Calculated QC metrics for {self.sp.n_cells:,} cells")

    def _save_plot(self, filename: str) -> None:
        """Save the current figure or display it interactively.

        Parameters
        ----------
        filename : str
            Output file name placed in ``config.analysis_dir``.
        """
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.analysis_dir, filename),
                        dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------
    # QC steps
    # ------------------------------------------------------------------

    def qc_cell_area(self, plot: bool = True) -> pd.DataFrame:
        """Assess cell and nucleus size distributions.

        Plots histograms of ``cell_area`` and ``nucleus_area`` with
        threshold lines, and a scatter of nucleus vs. cell area coloured
        by nucleus : cell ratio.

        Parameters
        ----------
        plot : bool
            Whether to create figures.

        Returns
        -------
        pd.DataFrame
            Per-cell area metrics with ``area_pass`` flag.

        Raises
        ------
        ValueError
            If ``cell_area`` is absent from ``sp.cell_meta``.
        """
        print("\n[Xenium QC] Cell & Nucleus Area")

        if "cell_area" not in self.sp.cell_meta.columns:
            raise ValueError(
                "'cell_area' not found in cell_meta. "
                "Was the object loaded with from_xenium()?"
            )

        cell_area    = self.sp.cell_meta["cell_area"].values.astype(float)
        nucleus_area = self.sp.cell_meta["nucleus_area"].values.astype(float) \
                       if "nucleus_area" in self.sp.cell_meta.columns \
                       else np.full(self.sp.n_cells, np.nan)
        ratio        = self.sp.cell_meta["nucleus_cell_ratio"].values.astype(float) \
                       if "nucleus_cell_ratio" in self.sp.cell_meta.columns \
                       else np.full(self.sp.n_cells, np.nan)

        area_pass = (
            (cell_area >= self.config.cell_area_min) &
            (cell_area <= self.config.cell_area_max)
        )
        self.sp.cell_meta["QC_area_pass"] = area_pass

        results = pd.DataFrame({
            "cell_area":          cell_area,
            "nucleus_area":       nucleus_area,
            "nucleus_cell_ratio": ratio,
            "area_pass":          area_pass,
        }, index=self.sp.cell_index)

        pct_pass = area_pass.mean() * 100
        print(f"  cell_area range  : {cell_area.min():.1f} -- {cell_area.max():.1f} um2")
        print(f"  Median cell_area : {np.median(cell_area):.1f} um2")
        print(f"  Cells passing    : {area_pass.sum():,} / {len(area_pass):,} ({pct_pass:.1f}%)")

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            axes[0].hist(cell_area, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
            axes[0].axvline(self.config.cell_area_min, color="red",    linestyle="--",
                            label=f"min {self.config.cell_area_min} um2")
            axes[0].axvline(self.config.cell_area_max, color="orange", linestyle="--",
                            label=f"max {self.config.cell_area_max} um2")
            axes[0].set_xlabel("Cell area (um2)")
            axes[0].set_ylabel("Cells")
            axes[0].set_title("Cell Area Distribution")
            axes[0].legend(fontsize=8)

            valid_nuc = nucleus_area[~np.isnan(nucleus_area)]
            if len(valid_nuc):
                axes[1].hist(valid_nuc, bins=80, color="coral", alpha=0.75, edgecolor="none")
            axes[1].set_xlabel("Nucleus area (um2)")
            axes[1].set_ylabel("Cells")
            axes[1].set_title("Nucleus Area Distribution")

            valid = ~np.isnan(ratio)
            if valid.any():
                sc = axes[2].scatter(cell_area[valid], nucleus_area[valid],
                                     c=ratio[valid], cmap="RdYlGn_r",
                                     vmin=0, vmax=1.2, s=2, alpha=0.5, linewidths=0)
                x_line = np.linspace(0, cell_area[valid].max(), 100)
                axes[2].plot(x_line, x_line * self.config.nucleus_cell_ratio_max,
                             color="red", linestyle="--",
                             label=f"ratio = {self.config.nucleus_cell_ratio_max}")
                axes[2].set_xlabel("Cell area (um2)")
                axes[2].set_ylabel("Nucleus area (um2)")
                axes[2].set_title("Nucleus vs Cell Area")
                axes[2].legend(fontsize=8)
                plt.colorbar(sc, ax=axes[2], label="Nucleus : cell ratio")

            plt.suptitle("Xenium QC -- Cell & Nucleus Area", fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_cell_area.png")

        self.qc_metrics["cell_area"] = results
        return results

    def qc_transcript_metrics(self, plot: bool = True) -> pd.DataFrame:
        """Assess per-cell transcript count and feature diversity.

        Plots distributions of ``transcript_counts``, ``n_features_by_counts``,
        and a joint scatter coloured by ``pct_counts_neg``.

        Parameters
        ----------
        plot : bool
            Whether to create figures.

        Returns
        -------
        pd.DataFrame
            Per-cell transcript metrics.
        """
        print("\n[Xenium QC] Transcript Metrics")

        tc_col = "transcript_counts" if "transcript_counts" in self.sp.cell_meta.columns \
                 else "transcript_counts_expr"
        tc    = self.sp.cell_meta[tc_col].values.astype(float)
        nf    = self.sp.cell_meta["n_features_by_counts"].values.astype(float)
        p_neg = self.sp.cell_meta["pct_counts_neg"].values.astype(float)

        print(f"  Median {tc_col}     : {np.median(tc):.0f}")
        print(f"  Median n_features          : {np.median(nf):.0f}")
        print(f"  Median pct_counts_neg      : {np.median(p_neg)*100:.2f}%")
        print(f"  Cells with < {self.config.transcript_counts_min} transcripts: "
              f"{(tc < self.config.transcript_counts_min).sum():,}")
        print(f"  Cells with < {self.config.n_features_min} features    : "
              f"{(nf < self.config.n_features_min).sum():,}")

        results = pd.DataFrame({
            tc_col:                 tc,
            "n_features_by_counts": nf,
            "pct_counts_neg":       p_neg,
        }, index=self.sp.cell_index)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            axes[0].hist(tc, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
            axes[0].axvline(self.config.transcript_counts_min, color="red", linestyle="--",
                            label=f"min = {self.config.transcript_counts_min}")
            axes[0].set_xlabel("Transcript counts (panel genes)")
            axes[0].set_ylabel("Cells")
            axes[0].set_title("Transcript Counts per Cell")
            axes[0].set_yscale("log")
            axes[0].legend(fontsize=8)

            axes[1].hist(nf, bins=80, color="seagreen", alpha=0.75, edgecolor="none")
            axes[1].axvline(self.config.n_features_min, color="red", linestyle="--",
                            label=f"min = {self.config.n_features_min}")
            axes[1].set_xlabel("Unique genes detected")
            axes[1].set_ylabel("Cells")
            axes[1].set_title("Features per Cell")
            axes[1].set_yscale("log")
            axes[1].legend(fontsize=8)

            sc = axes[2].scatter(np.log1p(tc), nf,
                                 c=p_neg * 100, cmap="Reds",
                                 vmin=0, vmax=self.config.pct_counts_neg_max * 100 * 2,
                                 s=2, alpha=0.4, linewidths=0)
            axes[2].axvline(np.log1p(self.config.transcript_counts_min),
                            color="red",    linestyle="--", linewidth=1)
            axes[2].axhline(self.config.n_features_min,
                            color="orange", linestyle="--", linewidth=1)
            axes[2].set_xlabel("log1p(transcript counts)")
            axes[2].set_ylabel("Unique genes detected")
            axes[2].set_title("Transcript Counts vs Features")
            plt.colorbar(sc, ax=axes[2], label="% neg-ctrl counts")

            plt.suptitle("Xenium QC -- Transcript Metrics", fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_transcript_metrics.png")

        self.qc_metrics["transcript_metrics"] = results
        return results

    def qc_control_metrics(self, plot: bool = True) -> pd.DataFrame:
        """Assess per-cell counts from each of the five Xenium control types.

        Produces a grouped bar chart of per-type mean fractions and
        violin plots of each control-type fraction per cell.

        Parameters
        ----------
        plot : bool
            Whether to create figures.

        Returns
        -------
        pd.DataFrame
            Per-cell fractions for all five control-gene prefixes plus
            the aggregate ``pct_counts_neg``.
        """
        print("\n[Xenium QC] Control Feature Metrics")

        pct_cols = [f"pct_counts_{p}" for p in self.config.neg_prefixes
                    if f"pct_counts_{p}" in self.sp.cell_meta.columns]

        results = self.sp.cell_meta[["pct_counts_neg"] + pct_cols].copy()

        print(f"  {'Type':<35}  {'Mean %':>8}  {'Median %':>9}  {'Max %':>8}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*9}  {'-'*8}")
        for col in ["pct_counts_neg"] + pct_cols:
            vals = self.sp.cell_meta[col].values * 100
            print(f"  {col:<35}  {vals.mean():8.3f}  {np.median(vals):9.3f}  {vals.max():8.3f}")

        pct_neg_pct = self.sp.cell_meta["pct_counts_neg"].values * 100
        n_warn  = (pct_neg_pct >= 4.0).sum()
        n_error = (pct_neg_pct >= 10.0).sum()
        print(f"\n  XOA alert -- cells >= 4%  (warning threshold): {n_warn:,}")
        print(f"  XOA alert -- cells >= 10% (error threshold)  : {n_error:,}")

        if plot and pct_cols:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            labels    = [c.replace("pct_counts_", "") for c in pct_cols]
            means_pct = [self.sp.cell_meta[c].mean() * 100 for c in pct_cols]
            bar_colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
            bars = axes[0].bar(labels, means_pct,
                               color=bar_colors[:len(labels)],
                               alpha=0.85, edgecolor="black")
            axes[0].axhline(self.config.pct_counts_neg_max * 100,
                            color="red", linestyle="--",
                            label=f"threshold {self.config.pct_counts_neg_max*100:.0f}%")
            axes[0].set_xlabel("Control type")
            axes[0].set_ylabel("Mean % of total counts")
            axes[0].set_title("Mean Control-Type Fractions per Cell")
            axes[0].tick_params(axis="x", rotation=20)
            axes[0].legend(fontsize=8)
            for bar, v in zip(bars, means_pct):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{v:.3f}%", ha="center", va="bottom", fontsize=8)

            data_violin = [self.sp.cell_meta[c].values * 100 for c in pct_cols]
            axes[1].violinplot(data_violin, showmedians=True)
            axes[1].set_xticks(range(1, len(pct_cols) + 1))
            axes[1].set_xticklabels(labels, rotation=20, ha="right")
            axes[1].axhline(self.config.pct_counts_neg_max * 100,
                            color="red",    linestyle="--",
                            label=f"threshold {self.config.pct_counts_neg_max*100:.0f}%")
            axes[1].axhline(4.0,  color="orange", linestyle=":", label="XOA warning 4%")
            axes[1].axhline(10.0, color="red",    linestyle=":", label="XOA error 10%",
                            linewidth=1.5)
            axes[1].set_ylabel("% of total counts")
            axes[1].set_title("Per-Cell Control-Type Distribution")
            axes[1].legend(fontsize=7)

            plt.suptitle("Xenium QC -- Negative Control Metrics", fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_control_metrics.png")

        self.qc_metrics["control_metrics"] = results
        return results

    def qc_nucleus_cell_ratio(self, plot: bool = True) -> pd.DataFrame:
        """Assess nucleus : cell area ratio as a segmentation quality metric.

        Cells with a ratio > 1.0 have a nucleus larger than their segmented
        cell boundary, indicating a segmentation error.

        Parameters
        ----------
        plot : bool
            Whether to create figures.

        Returns
        -------
        pd.DataFrame
            Per-cell ratio values with ``ratio_pass`` flag.

        Raises
        ------
        ValueError
            If nucleus_cell_ratio has not been computed
            (missing ``cell_area`` or ``nucleus_area`` in ``cell_meta``).
        """
        print("\n[Xenium QC] Nucleus : Cell Area Ratio")

        if "nucleus_cell_ratio" not in self.sp.cell_meta.columns:
            raise ValueError(
                "'nucleus_cell_ratio' not in cell_meta. "
                "Ensure cell_area and nucleus_area are present "
                "(loaded via from_xenium)."
            )

        ratio = self.sp.cell_meta["nucleus_cell_ratio"].values.astype(float)
        valid = ~np.isnan(ratio)

        ratio_pass             = np.ones(len(ratio), dtype=bool)
        ratio_pass[valid]      = ratio[valid] <= self.config.nucleus_cell_ratio_max
        self.sp.cell_meta["QC_ratio_pass"] = ratio_pass

        n_err  = (~ratio_pass & valid).sum()
        pct_ok = ratio_pass[valid].mean() * 100
        print(f"  Median ratio                   : {np.nanmedian(ratio):.3f}")
        print(f"  Cells with ratio > {self.config.nucleus_cell_ratio_max} (seg. error): "
              f"{n_err:,}  ({100 - pct_ok:.1f}%)")

        results = pd.DataFrame({
            "nucleus_cell_ratio": ratio,
            "ratio_pass":         ratio_pass,
        }, index=self.sp.cell_index)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            r_valid = ratio[valid]
            axes[0].hist(r_valid[r_valid <= 2], bins=80,
                         color="steelblue", alpha=0.75, edgecolor="none")
            axes[0].axvline(self.config.nucleus_cell_ratio_max,
                            color="red", linestyle="--",
                            label=f"max = {self.config.nucleus_cell_ratio_max}")
            axes[0].set_xlabel("Nucleus / Cell area ratio")
            axes[0].set_ylabel("Cells")
            axes[0].set_title("Nucleus : Cell Ratio Distribution")
            axes[0].legend(fontsize=8)

            sorted_r = np.sort(r_valid)
            cdf      = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
            axes[1].plot(sorted_r, cdf, color="steelblue")
            axes[1].axvline(self.config.nucleus_cell_ratio_max,
                            color="red", linestyle="--",
                            label=f"threshold = {self.config.nucleus_cell_ratio_max}")
            axes[1].set_xlabel("Nucleus / Cell area ratio")
            axes[1].set_ylabel("Cumulative fraction")
            axes[1].set_title("CDF -- Nucleus : Cell Ratio")
            axes[1].set_xlim(0, min(sorted_r.max(), 2))
            axes[1].legend(fontsize=8)

            plt.suptitle("Xenium QC -- Segmentation Quality (Nucleus : Cell Ratio)",
                         fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_nucleus_cell_ratio.png")

        self.qc_metrics["nucleus_cell_ratio"] = results
        return results

    def qc_transcript_density(self, plot: bool = True) -> pd.DataFrame:
        """Assess spatial uniformity via transcript density (transcripts / um2).

        A spatially low-density region may indicate a tissue fold, wash
        failure, or other technical artefact.

        Parameters
        ----------
        plot : bool
            Whether to create figures.

        Returns
        -------
        pd.DataFrame
            Per-cell transcript density values.

        Raises
        ------
        ValueError
            If ``transcript_density`` has not been computed
            (missing ``cell_area`` in ``cell_meta``).
        """
        print("\n[Xenium QC] Transcript Density")

        if "transcript_density" not in self.sp.cell_meta.columns:
            raise ValueError(
                "'transcript_density' not in cell_meta. "
                "Ensure cell_area is present (loaded via from_xenium)."
            )

        density = self.sp.cell_meta["transcript_density"].values.astype(float)
        valid   = ~np.isnan(density)

        print(f"  Median transcript density : {np.nanmedian(density):.4f} tx/um2")
        print(f"  Mean   transcript density : {np.nanmean(density):.4f} tx/um2")

        results = pd.DataFrame({"transcript_density": density}, index=self.sp.cell_index)

        if plot:
            x = self.sp.cell_meta.get("x_centroid",
                                      pd.Series(np.full(self.sp.n_cells, np.nan))).values
            y = self.sp.cell_meta.get("y_centroid",
                                      pd.Series(np.full(self.sp.n_cells, np.nan))).values
            has_coords = not np.all(np.isnan(x))

            ncols = 2 if has_coords else 1
            fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
            if ncols == 1:
                axes = [axes]

            axes[0].hist(density[valid], bins=80, color="steelblue",
                         alpha=0.75, edgecolor="none")
            axes[0].set_xlabel("Transcript density (tx / um2)")
            axes[0].set_ylabel("Cells")
            axes[0].set_title("Transcript Density Distribution")

            if has_coords:
                sc = axes[1].scatter(x[valid], y[valid],
                                     c=density[valid], cmap="viridis",
                                     s=1, alpha=0.6, linewidths=0)
                axes[1].invert_yaxis()
                axes[1].set_aspect("equal")
                axes[1].set_xlabel("X (um)")
                axes[1].set_ylabel("Y (um)")
                axes[1].set_title("Spatial Map -- Transcript Density")
                plt.colorbar(sc, ax=axes[1], label="tx / um2")

            plt.suptitle("Xenium QC -- Transcript Density", fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_transcript_density.png")

        self.qc_metrics["transcript_density"] = results
        return results

    def qc_polygon_validity(self, plot: bool = True) -> pd.DataFrame:
        """Check validity of cell polygons using Shapely.

        Flags cells whose boundary geometry is degenerate (fewer than 3
        vertices, zero area, or self-intersecting / topologically invalid).
        Results are stored in ``sp.cell_meta["QC_polygon_valid"]``.

        Parameters
        ----------
        plot : bool
            Whether to create a summary bar chart of valid vs. invalid
            counts (with failure-reason breakdown) and a spatial scatter
            map of invalid cells (requires ``x_centroid`` / ``y_centroid``).

        Returns
        -------
        pd.DataFrame
            Per-cell validity DataFrame indexed by cell ID with columns
            ``n_vertices``, ``area``, ``is_valid``, and ``shapely_reason``.
        """
        print("\n[Xenium QC] Polygon Validity")

        if self.sp.polygons is None:
            print("  ⚠ No polygon data attached — skipping")
            return pd.DataFrame()

        from shapely.geometry import Polygon
        from shapely.validation import explain_validity

        poly_df     = self.sp.polygons
        cell_id_col = self.sp.config.cell_id_col
        x_col       = self.sp.config.x_global_col
        y_col       = self.sp.config.y_global_col

        records: list[dict] = []
        for cell_id, grp in poly_df.groupby(cell_id_col):
            coords  = list(zip(grp[x_col].values, grp[y_col].values))
            n_verts = len(coords)
            if n_verts < 3:
                records.append({
                    "cell_id":        str(cell_id),
                    "n_vertices":     n_verts,
                    "area":           0.0,
                    "is_valid":       False,
                    "shapely_reason": "fewer than 3 vertices",
                })
                continue
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                reason = ""
                valid  = True
            elif not poly.is_valid:
                reason = explain_validity(poly)
                valid  = False
            elif poly.is_empty:
                reason = "empty geometry"
                valid  = False
            else:
                reason = "zero area"
                valid  = False
            records.append({
                "cell_id":        str(cell_id),
                "n_vertices":     n_verts,
                "area":           poly.area,
                "is_valid":       valid,
                "shapely_reason": reason,
            })

        results = pd.DataFrame(records).set_index("cell_id")

        # Align to master cell index — cells with no polygon data → False
        valid_series = results["is_valid"].reindex(self.sp.cell_index, fill_value=False)
        self.sp.cell_meta["QC_polygon_valid"] = valid_series.values

        n_total   = len(self.sp.cell_index)
        n_valid   = int(valid_series.sum())
        n_missing = n_total - len(results)
        n_invalid = n_total - n_valid
        print(f"  Total cells       : {n_total:,}")
        print(f"  Valid polygons    : {n_valid:,} ({n_valid / n_total * 100:.1f}%)")
        print(f"  Invalid polygons  : {n_invalid:,} ({n_invalid / n_total * 100:.1f}%)")
        if n_missing > 0:
            print(f"    Of which {n_missing:,} have no polygon data at all")

        if plot:
            self._plot_polygon_validity_xenium(results, valid_series, n_valid, n_invalid)

        self.qc_metrics["polygon_validity"] = results
        return results

    def _plot_polygon_validity_xenium(
        self,
        results: pd.DataFrame,
        valid_series: pd.Series,
        n_valid: int,
        n_invalid: int,
    ) -> None:
        """Plot polygon validity summary with optional spatial map."""
        x = self.sp.cell_meta.get("x_centroid",
                                  pd.Series(np.full(self.sp.n_cells, np.nan))).values
        y = self.sp.cell_meta.get("y_centroid",
                                  pd.Series(np.full(self.sp.n_cells, np.nan))).values
        has_coords = not np.all(np.isnan(x))

        ncols = 3 if has_coords else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

        # Panel 1 — valid vs invalid counts
        axes[0].bar(
            ["Valid", "Invalid"],
            [n_valid, n_invalid],
            color=["lightgreen", "lightcoral"],
            edgecolor="black",
        )
        axes[0].set_title("Polygon Validity")
        axes[0].set_ylabel("Number of Cells")
        axes[0].grid(alpha=0.3, axis="y")
        for i, v in enumerate([n_valid, n_invalid]):
            axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

        # Panel 2 — failure reason breakdown
        invalid_df = results[~results["is_valid"]]
        if len(invalid_df):
            reason_counts = invalid_df["shapely_reason"].value_counts()
            axes[1].barh(reason_counts.index, reason_counts.values,
                         color="lightcoral", edgecolor="black")
            axes[1].set_title("Failure Reasons")
            axes[1].set_xlabel("Number of Cells")
            for i, v in enumerate(reason_counts.values):
                axes[1].text(v, i, f" {v:,}", va="center")
        else:
            axes[1].text(0.5, 0.5, "No invalid polygons", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title("Failure Reasons")

        # Panel 3 — spatial scatter (Xenium has continuous coordinates)
        if has_coords:
            valid_mask   = valid_series.reindex(self.sp.cell_index).fillna(False).values
            invalid_mask = ~valid_mask
            axes[2].scatter(x[valid_mask],   y[valid_mask],
                            c="lightgreen", s=1, alpha=0.3, linewidths=0, label="Valid")
            if invalid_mask.any():
                axes[2].scatter(x[invalid_mask], y[invalid_mask],
                                c="red", s=4, alpha=0.8, linewidths=0, label="Invalid")
            axes[2].invert_yaxis()
            axes[2].set_aspect("equal")
            axes[2].set_xlabel("X (um)")
            axes[2].set_ylabel("Y (um)")
            axes[2].set_title("Spatial Map — Invalid Polygons")
            axes[2].legend(fontsize=8, markerscale=4)

        plt.suptitle("Xenium QC -- Polygon Validity", fontsize=13)
        plt.tight_layout()
        self._save_plot("QC_xenium_polygon_validity.png")

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_cells(self, custom_filters: dict | None = None) -> pd.Series:
        """Apply all cell-level QC thresholds.

        Stores boolean mask in ``sp.cell_meta["QC_pass"]``.

        Default filters:

        * ``transcript_counts`` >= ``transcript_counts_min``
        * ``n_features_by_counts`` >= ``n_features_min``
        * ``pct_counts_neg`` <= ``pct_counts_neg_max``
        * ``pct_counts_mt`` <= ``pct_counts_mt_max``
        * ``cell_area`` in [``cell_area_min``, ``cell_area_max``] (if available)
        * ``nucleus_cell_ratio`` <= ``nucleus_cell_ratio_max`` (if available)

        Parameters
        ----------
        custom_filters : dict, optional
            Extra filters as ``{column: (operator, threshold)}``
            where *operator* is one of ``">"``, ``"<"``, ``">="`` or ``"<="``.

        Returns
        -------
        pd.Series
            Boolean mask; True = cell passes QC.
        """
        print("\n[Xenium QC] Filtering Cells")

        tc_col = "transcript_counts" if "transcript_counts" in self.sp.cell_meta.columns \
                 else "transcript_counts_expr"

        mask = (
            (self.sp.cell_meta[tc_col] >= self.config.transcript_counts_min) &
            (self.sp.cell_meta["n_features_by_counts"] >= self.config.n_features_min) &
            (self.sp.cell_meta["pct_counts_neg"] <= self.config.pct_counts_neg_max) &
            (self.sp.cell_meta["pct_counts_mt"]  <= self.config.pct_counts_mt_max)
        )

        if "cell_area" in self.sp.cell_meta.columns:
            mask &= (
                (self.sp.cell_meta["cell_area"] >= self.config.cell_area_min) &
                (self.sp.cell_meta["cell_area"] <= self.config.cell_area_max)
            )

        if "nucleus_cell_ratio" in self.sp.cell_meta.columns:
            mask &= (
                self.sp.cell_meta["nucleus_cell_ratio"].isna() |
                (self.sp.cell_meta["nucleus_cell_ratio"] <= self.config.nucleus_cell_ratio_max)
            )

        # Polygon validity filter (run qc_polygon_validity first to populate)
        if "QC_polygon_valid" in self.sp.cell_meta.columns:
            mask &= self.sp.cell_meta["QC_polygon_valid"]

        if custom_filters:
            ops = {">": pd.Series.gt, "<": pd.Series.lt,
                   ">=": pd.Series.ge, "<=": pd.Series.le}
            for col, (op, thr) in custom_filters.items():
                if col not in self.sp.cell_meta.columns:
                    print(f"  Warning: custom filter column '{col}' not found -- skipped")
                    continue
                if op not in ops:
                    raise ValueError(f"Unknown operator '{op}'. Use: {list(ops)}")
                mask &= ops[op](self.sp.cell_meta[col], thr)

        self.sp.cell_meta["QC_pass"] = mask

        n_before = len(mask)
        n_after  = mask.sum()
        print(f"  Before : {n_before:,} cells")
        print(f"  After  : {n_after:,} cells ({n_after / n_before * 100:.1f}% kept)")

        reasons: dict[str, int] = {
            f"{tc_col} < {self.config.transcript_counts_min}":
                int((self.sp.cell_meta[tc_col] < self.config.transcript_counts_min).sum()),
            f"n_features < {self.config.n_features_min}":
                int((self.sp.cell_meta["n_features_by_counts"] < self.config.n_features_min).sum()),
            f"pct_neg > {self.config.pct_counts_neg_max:.0%}":
                int((self.sp.cell_meta["pct_counts_neg"] > self.config.pct_counts_neg_max).sum()),
            f"pct_mt > {self.config.pct_counts_mt_max:.0%}":
                int((self.sp.cell_meta["pct_counts_mt"] > self.config.pct_counts_mt_max).sum()),
        }
        if "cell_area" in self.sp.cell_meta.columns:
            reasons[f"cell_area < {self.config.cell_area_min} um2"] = int(
                (self.sp.cell_meta["cell_area"] < self.config.cell_area_min).sum())
            reasons[f"cell_area > {self.config.cell_area_max} um2"] = int(
                (self.sp.cell_meta["cell_area"] > self.config.cell_area_max).sum())
        if "nucleus_cell_ratio" in self.sp.cell_meta.columns:
            reasons[f"nucleus_cell_ratio > {self.config.nucleus_cell_ratio_max}"] = int(
                (self.sp.cell_meta["nucleus_cell_ratio"].fillna(0)
                 > self.config.nucleus_cell_ratio_max).sum())
        if "QC_polygon_valid" in self.sp.cell_meta.columns:
            reasons["invalid polygon geometry"] = int(
                (~self.sp.cell_meta["QC_polygon_valid"]).sum())

        print("\n  Cells failing each individual filter (counts may overlap):")
        for reason, n in reasons.items():
            print(f"    {reason:<50}: {n:,}")

        self.qc_metrics["cell_filtering"] = {
            "n_before": n_before,
            "n_after":  n_after,
            "pct_kept": n_after / n_before * 100,
            "reasons":  reasons,
        }
        return mask

    def filter_genes(self, plot: bool = True) -> pd.Series:
        """Filter panel genes by expression relative to the negative-control baseline.

        Control genes (all five Xenium prefixes) are always excluded from
        the output mask.  Panel genes are retained if their total expression
        exceeds the baseline defined by ``gene_filter_method``.

        Parameters
        ----------
        plot : bool
            Whether to create a figure.

        Returns
        -------
        pd.Series
            Boolean mask indexed by gene name; stored as
            ``sp.gene_meta["QC_gene_pass"]``.

        Raises
        ------
        ValueError
            If an unknown ``gene_filter_method`` is specified, or a
            required config value is missing.
        """
        print(f"\n[Xenium QC] Filtering Genes (method={self.config.gene_filter_method})")

        import scipy.sparse as _sp_sparse

        # Work on the sparse matrix — no dense materialisation.
        X = self.sp.expression.get_sparse()
        if not _sp_sparse.issparse(X):
            X = _sp_sparse.csc_matrix(X)
        else:
            X = X.tocsc()          # CSC is optimal for column-wise sums / nnz
        X.eliminate_zeros()

        is_ctrl  = self.sp.gene_meta["is_neg_ctrl"].values
        is_panel = ~is_ctrl

        panel_idx = np.where(is_panel)[0]
        ctrl_idx  = np.where(is_ctrl)[0]

        # Column sums (O(nnz), no dense copy)
        gene_totals = np.asarray(X[:, panel_idx].sum(axis=0)).ravel()
        ctrl_totals = np.asarray(X[:, ctrl_idx].sum(axis=0)).ravel() if ctrl_idx.size else np.zeros(0)

        method = self.config.gene_filter_method
        baseline = 0.0

        if method == "percentile":
            baseline   = float(np.percentile(ctrl_totals, self.config.gene_percentile_threshold)) \
                         if ctrl_totals.size else 0.0
            keep_panel = gene_totals > baseline
            desc       = (f"{self.config.gene_percentile_threshold}th pct of ctrl baseline "
                          f"({baseline:.1f} counts)")

        elif method == "absolute":
            if self.config.gene_min_cells is None:
                raise ValueError("Set config.gene_min_cells for 'absolute' method.")
            keep_panel = gene_totals > self.config.gene_min_cells
            baseline   = float(self.config.gene_min_cells)
            desc       = f"absolute threshold > {self.config.gene_min_cells}"

        elif method == "min_cells":
            if self.config.gene_min_cells is None:
                raise ValueError("Set config.gene_min_cells for 'min_cells' method.")
            # nnz per column in panel slice = cells expressing each gene
            X_panel_csc = X[:, panel_idx]  # already CSC
            n_expressing = np.diff(X_panel_csc.indptr)
            keep_panel   = n_expressing >= self.config.gene_min_cells
            desc         = f"expressed in >= {self.config.gene_min_cells} cells"

        else:
            raise ValueError(
                f"Unknown gene_filter_method '{method}'. "
                "Use 'percentile', 'absolute', or 'min_cells'."
            )

        full_mask = np.zeros(self.sp.n_genes, dtype=bool)
        full_mask[is_panel] = keep_panel

        self.sp.gene_meta["QC_gene_pass"] = full_mask

        n_panel  = int(is_panel.sum())
        n_kept   = int(keep_panel.sum())
        n_ctrl   = int(is_ctrl.sum())
        print(f"  Filter criterion   : {desc}")
        print(f"  Panel genes before : {n_panel:,}")
        print(f"  Panel genes kept   : {n_kept:,} ({n_kept / n_panel * 100:.1f}%)")
        print(f"  Control genes      : {n_ctrl:,}  (excluded from output)")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].hist(np.log1p(gene_totals), bins=60,
                         color="steelblue", alpha=0.75, edgecolor="none", label="All panel genes")
            axes[0].hist(np.log1p(gene_totals[keep_panel]), bins=60,
                         color="seagreen",  alpha=0.65, edgecolor="none", label="Kept genes")
            if len(ctrl_totals) and method == "percentile":
                axes[0].axvline(np.log1p(baseline), color="red", linestyle="--",
                                label=f"Ctrl baseline (log1p={np.log1p(baseline):.2f})")
            axes[0].set_xlabel("log1p(Total counts)")
            axes[0].set_ylabel("Genes")
            axes[0].set_title("Gene Expression Distribution")
            axes[0].legend(fontsize=8)

            labels = ["Ctrl genes\n(excluded)", "Panel kept", "Panel removed"]
            values = [n_ctrl, n_kept, n_panel - n_kept]
            colors = ["#D65F5F", "#6ACC65", "#B0B0B0"]
            axes[1].bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
            for i, v in enumerate(values):
                axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
            axes[1].set_ylabel("Number of genes")
            axes[1].set_title("Gene Filtering Summary")

            plt.suptitle("Xenium QC -- Gene Filtering", fontsize=13)
            plt.tight_layout()
            self._save_plot("QC_xenium_gene_filtering.png")

        self.qc_metrics["gene_filtering"] = {
            "n_panel_before":  n_panel,
            "n_panel_kept":    n_kept,
            "n_ctrl_excluded": n_ctrl,
            "pct_kept":        n_kept / n_panel * 100,
            "method":          method,
        }
        return pd.Series(full_mask, index=self.sp.gene_index)

    # ------------------------------------------------------------------
    # Apply & summarize
    # ------------------------------------------------------------------

    def apply_filters(
        self, filter_cells: bool = True, filter_genes: bool = True
    ) -> "spatioloji":
        """Apply QC masks to return a filtered spatioloji object.

        Parameters
        ----------
        filter_cells : bool
            Apply the cell mask from :meth:`filter_cells`.
        filter_genes : bool
            Apply the gene mask from :meth:`filter_genes`.

        Returns
        -------
        spatioloji
            New filtered object with Xenium-specific attributes preserved.

        Raises
        ------
        RuntimeError
            If the requested filter step has not been run.
        """
        print("\nApplying Xenium QC filters...")

        if filter_cells:
            if "QC_pass" not in self.sp.cell_meta.columns:
                raise RuntimeError("Run filter_cells() before apply_filters()")
            cell_ids = self.sp.cell_index[self.sp.cell_meta["QC_pass"]].tolist()
        else:
            cell_ids = self.sp.cell_index.tolist()

        if filter_genes:
            if "QC_gene_pass" not in self.sp.gene_meta.columns:
                raise RuntimeError("Run filter_genes() before apply_filters()")
            gene_names = self.sp.gene_index[self.sp.gene_meta["QC_gene_pass"]].tolist()
        else:
            gene_names = self.sp.gene_index.tolist()

        filtered = self.sp.subset_by_cells(cell_ids)
        filtered = filtered.subset_by_genes(gene_names)

        for attr in ("_xenium_dir", "_xenium_image_path", "_xenium_pixel_size"):
            if hasattr(self.sp, attr):
                setattr(filtered, attr, getattr(self.sp, attr))

        print(f"  Filtered: {filtered.n_cells:,} cells x {filtered.n_genes:,} genes")
        return filtered

    def summarize_qc(self, plot: bool = True) -> dict:
        """Print and optionally plot a full QC summary.

        Parameters
        ----------
        plot : bool
            Whether to produce a summary bar chart.

        Returns
        -------
        dict
            All collected QC metrics.
        """
        import json

        print("\n" + "=" * 70)
        print("XENIUM QC SUMMARY")
        print("=" * 70)

        summary: dict = {
            "timestamp":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_cells_input":  self.sp.n_cells,
            "n_genes_input":  self.sp.n_genes,
            "metrics":        self.qc_metrics,
        }

        if "cell_filtering" in self.qc_metrics:
            cf = self.qc_metrics["cell_filtering"]
            print(f"\nCells  : {cf['n_before']:,} -> {cf['n_after']:,} "
                  f"({cf['pct_kept']:.1f}% kept)")
            for reason, n in cf.get("reasons", {}).items():
                print(f"  Fail {reason}: {n:,}")

        if "gene_filtering" in self.qc_metrics:
            gf = self.qc_metrics["gene_filtering"]
            print(f"\nGenes  : {gf['n_panel_before']:,} panel -> {gf['n_panel_kept']:,} kept "
                  f"({gf['pct_kept']:.1f}%); {gf['n_ctrl_excluded']:,} ctrl excluded")

        print("=" * 70)

        if plot and "cell_filtering" in self.qc_metrics and "gene_filtering" in self.qc_metrics:
            cf = self.qc_metrics["cell_filtering"]
            gf = self.qc_metrics["gene_filtering"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, title, before, after in [
                (axes[0], "Cell Filtering",  cf["n_before"],        cf["n_after"]),
                (axes[1], "Gene Filtering",  gf["n_panel_before"],  gf["n_panel_kept"]),
            ]:
                bars = ax.bar(["Before QC", "After QC"], [before, after],
                              color=["lightcoral", "lightgreen"], edgecolor="black")
                for i, bar in enumerate(bars):
                    h   = bar.get_height()
                    lbl = f"{h:,}\n({after/before*100:.1f}%)" if i == 1 else f"{h:,}"
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            lbl, ha="center", va="bottom", fontsize=9)
                ax.set_title(f"Xenium QC -- {title}")
                ax.set_ylabel("Count")
            plt.tight_layout()
            self._save_plot("QC_xenium_summary.png")

        if self.config.save_plots:
            def _serial(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    return obj.to_dict()
                raise TypeError(type(obj))

            with open(os.path.join(self.config.data_dir, "xenium_qc_summary.json"), "w") as f:
                json.dump(summary, f, indent=2, default=_serial)

        return summary

    def run_qc_pipeline(self, plot: bool = True) -> "spatioloji":
        """Run the full Xenium QC pipeline in one call.

        Steps:

        1. :meth:`qc_cell_area`
        2. :meth:`qc_transcript_metrics`
        3. :meth:`qc_control_metrics`
        4. :meth:`qc_nucleus_cell_ratio` (if nucleus_cell_ratio available)
        5. :meth:`qc_transcript_density` (if transcript_density available)
        6. :meth:`qc_polygon_validity` (if polygons attached)
        7. :meth:`filter_cells`
        8. :meth:`filter_genes`
        9. :meth:`summarize_qc`
        10. :meth:`apply_filters`

        Parameters
        ----------
        plot : bool
            Whether to produce figures at each step.

        Returns
        -------
        spatioloji
            Filtered spatioloji object ready for downstream analysis.
        """
        print("\n" + "=" * 70)
        print("XENIUM QC PIPELINE")
        print("=" * 70)

        self.qc_cell_area(plot=plot)
        self.qc_transcript_metrics(plot=plot)
        self.qc_control_metrics(plot=plot)

        if "nucleus_cell_ratio" in self.sp.cell_meta.columns:
            self.qc_nucleus_cell_ratio(plot=plot)

        if "transcript_density" in self.sp.cell_meta.columns:
            self.qc_transcript_density(plot=plot)

        if self.sp.polygons is not None:
            self.qc_polygon_validity(plot=plot)

        self.filter_cells()
        self.filter_genes(plot=plot)
        self.summarize_qc(plot=plot)
        filtered = self.apply_filters()

        print("\n" + "=" * 70)
        print("XENIUM QC PIPELINE COMPLETED")
        print("=" * 70 + "\n")

        return filtered
