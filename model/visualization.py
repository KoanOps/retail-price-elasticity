"""
Visualization module for Bayesian elasticity models.

This module provides visualization tools for exploring and communicating
results from Bayesian elasticity models.
"""
from typing import Dict, List, Optional, Any, Tuple, Union, cast
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from utils.logging_utils import logger, log_step
from model.exceptions import VisualizationError


class BayesianVisualizer:
    """
    Visualization tools for Bayesian elasticity models.
    
    Responsibilities:
    - Creating elasticity distribution plots
    - Generating comparative visualizations
    - Creating business-focused visualizations
    - Exporting visualization-ready data
    """
    
    def __init__(
        self,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save visualization outputs
        """
        self.results_dir = results_dir
        
        if self.results_dir is not None:
            self.viz_dir = self.results_dir / "visualizations"
            self.viz_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.viz_dir = None
            
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 100
            
    def plot_elasticity_distribution(
        self,
        elasticity_df: pd.DataFrame,
        filename: str = "elasticity_distribution.png"
    ) -> Optional[Path]:
        """
        Plot the distribution of elasticity estimates across SKUs.
        
        Args:
            elasticity_df: DataFrame with elasticity estimates
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.viz_dir is None:
            logger.warning("No visualization directory specified")
            return None
            
        try:
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the distribution
            sns.histplot(
                elasticity_df["elasticity_mean"],
                kde=True,
                bins=30,
                color="steelblue",
                ax=ax
            )
            
            # Add a vertical line at elasticity = -1 (unit elasticity)
            ax.axvline(x=-1, color="red", linestyle="--", alpha=0.7, 
                      label="Unit Elasticity")
            
            # Add the mean elasticity
            mean_elasticity = elasticity_df["elasticity_mean"].mean()
            ax.axvline(x=mean_elasticity, color="green", linestyle="-.", alpha=0.7,
                      label=f"Mean: {mean_elasticity:.2f}")
            
            # Add labels and title
            ax.set_xlabel("Price Elasticity")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Price Elasticity Estimates Across SKUs")
            ax.legend()
            
            # Add annotations for elasticity ranges
            ymin, ymax = ax.get_ylim()
            text_y = ymax * 0.9
            
            ax.text(-2.5, text_y, "Elastic\n(>1)", color="darkblue", ha="center", 
                   bbox=dict(facecolor="white", alpha=0.5))
            ax.text(-0.5, text_y, "Inelastic\n(<1)", color="darkred", ha="center", 
                   bbox=dict(facecolor="white", alpha=0.5))
            
            # Save the plot
            output_path = self.viz_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved elasticity distribution plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Distribution plotting failed: {str(e)}")
            
    def plot_elasticity_by_class(
        self,
        elasticity_df: pd.DataFrame,
        filename: str = "elasticity_by_class.png"
    ) -> Optional[Path]:
        """
        Plot elasticity estimates grouped by product class.
        
        Args:
            elasticity_df: DataFrame with elasticity estimates and product class
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.viz_dir is None:
            logger.warning("No visualization directory specified")
            return None
            
        if "product_class" not in elasticity_df.columns:
            logger.warning("Product class information missing from elasticity data")
            return None
            
        try:
            # Compute class-level statistics
            class_stats = elasticity_df.groupby("product_class").agg({
                "elasticity_mean": ["mean", "std", "count"]
            }).reset_index()
            
            # Flatten column names
            class_stats.columns = ["product_class", "mean", "std", "count"]
            
            # Sort by mean elasticity
            class_stats = class_stats.sort_values("mean")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, len(class_stats) * 0.4 + 2))
            
            # Plot error bars
            ax.errorbar(
                x=class_stats["mean"],
                y=class_stats["product_class"],
                xerr=class_stats["std"],
                fmt="o",
                capsize=5,
                markersize=8,
                markeredgewidth=1,
                markerfacecolor="steelblue",
                markeredgecolor="black",
                ecolor="black",
                elinewidth=1,
                capthick=1
            )
            
            # Add count labels
            for i, row in class_stats.iterrows():
                ax.text(
                    row["mean"],
                    i,
                    f" n={row['count']}",
                    va="center"
                )
            
            # Add vertical line at elasticity = -1
            ax.axvline(x=-1, color="red", linestyle="--", alpha=0.7,
                      label="Unit Elasticity")
            
            # Add labels and title
            ax.set_xlabel("Mean Price Elasticity (with Std Dev)")
            ax.set_title("Price Elasticity by Product Class")
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            ax.legend()
            
            # Save the plot
            output_path = self.viz_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved elasticity by class plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Class elasticity plotting failed: {str(e)}")
            
    def plot_price_optimization(
        self,
        elasticity_df: pd.DataFrame,
        n_skus: int = 10,
        filename: str = "price_optimization.png"
    ) -> Optional[Path]:
        """
        Plot revenue impact of price changes for top elastic SKUs.
        
        Args:
            elasticity_df: DataFrame with elasticity estimates
            n_skus: Number of SKUs to include
            filename: Name of the file to save the plot to
            
        Returns:
            Path to the saved plot or None if saving failed
            
        Raises:
            VisualizationError: If plotting fails
        """
        if self.viz_dir is None:
            logger.warning("No visualization directory specified")
            return None
            
        try:
            # Select the most elastic SKUs
            top_skus = elasticity_df.sort_values("elasticity_mean").head(n_skus)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define price change range
            price_changes = np.linspace(-0.20, 0.20, 41)  # -20% to +20%
            
            # Compute revenue impact for each SKU across price changes
            for i, (_, sku) in enumerate(top_skus.iterrows()):
                elasticity = sku["elasticity_mean"]
                revenue_impact = (1 + price_changes) * (
                    1 + elasticity * price_changes
                ) - 1
                
                # Plot revenue impact curve
                ax.plot(
                    price_changes * 100,  # Convert to percentage
                    revenue_impact * 100,  # Convert to percentage
                    marker="o",
                    markersize=4,
                    label=f"{sku['sku']} (e={elasticity:.2f})"
                )
                
                # Find and mark the optimal price point
                optimal_idx = np.argmax(revenue_impact)
                optimal_price_change = price_changes[optimal_idx]
                optimal_revenue_impact = revenue_impact[optimal_idx]
                
                ax.scatter(
                    optimal_price_change * 100,
                    optimal_revenue_impact * 100,
                    s=100,
                    color="red",
                    zorder=10
                )
                
                # Add annotation for optimal price
                ax.annotate(
                    f"{optimal_price_change*100:.1f}%",
                    (optimal_price_change * 100, optimal_revenue_impact * 100),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )
            
            # Add reference line at y=0 (no revenue impact)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            
            # Add reference line at x=0 (no price change)
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
            
            # Set labels and title
            ax.set_xlabel("Price Change (%)")
            ax.set_ylabel("Revenue Impact (%)")
            ax.set_title("Estimated Revenue Impact of Price Changes")
            
            # Add grid
            ax.grid(linestyle="--", alpha=0.7)
            
            # Add legend with smaller font
            ax.legend(fontsize=9, loc="best")
            
            # Save the plot
            output_path = self.viz_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Saved price optimization plot to {output_path}")
            return output_path
            
        except Exception as e:
            raise VisualizationError(f"Price optimization plotting failed: {str(e)}")
            
    def export_elasticity_excel(
        self,
        elasticity_df: pd.DataFrame,
        filename: str = "elasticity_results.xlsx"
    ) -> Optional[Path]:
        """
        Export elasticity results to Excel for business users.
        
        Args:
            elasticity_df: DataFrame with elasticity estimates
            filename: Name of the Excel file
            
        Returns:
            Path to the saved file or None if saving failed
        """
        if self.viz_dir is None:
            logger.warning("No visualization directory specified")
            return None
            
        try:
            # Create a copy to avoid modifying the original
            df_export = elasticity_df.copy()
            
            # Add column with optimal price change recommendation
            def get_optimal_price_change(elasticity):
                """Calculate optimal price change based on elasticity."""
                if elasticity >= 0:  # Non-negative elasticity
                    return 0.10  # Increase price by 10% for inelastic goods
                
                optimal = -1 / (2 * elasticity)
                
                # Clamp to reasonable range
                return max(-0.25, min(0.25, optimal))
            
            # Add optimal price recommendation
            df_export["optimal_price_change"] = df_export["elasticity_mean"].apply(
                get_optimal_price_change
            )
            
            # Add revenue impact of optimal price change
            df_export["estimated_revenue_impact"] = (
                (1 + df_export["optimal_price_change"]) *
                (1 + df_export["elasticity_mean"] * df_export["optimal_price_change"]) - 1
            )
            
            # Format percentages
            df_export["optimal_price_change"] = df_export["optimal_price_change"] * 100
            df_export["estimated_revenue_impact"] = df_export["estimated_revenue_impact"] * 100
            
            # Add elasticity category
            df_export["elasticity_category"] = pd.cut(
                df_export["elasticity_mean"],
                bins=[-np.inf, -2, -1, -0.5, 0, np.inf],
                labels=["Highly Elastic", "Elastic", "Moderately Inelastic", 
                       "Highly Inelastic", "Atypical"]
            )
            
            # Create writer
            output_path = self.viz_dir / filename
            
            # Export to Excel with formatting
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                # Write to Excel
                df_export.to_excel(writer, sheet_name="Elasticity Results", index=False)
                
                # Access workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets["Elasticity Results"]
                
                # Add formats
                header_format = workbook.add_format({
                    "bold": True, "text_wrap": True, "valign": "top", "border": 1
                })
                percent_format = workbook.add_format({"num_format": "0.00%"})
                
                # Apply formats
                for col_num, value in enumerate(df_export.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Auto-adjust columns
                for col_num, column in enumerate(df_export.columns):
                    column_width = max(
                        df_export[column].astype(str).map(len).max(),
                        len(column)
                    )
                    worksheet.set_column(col_num, col_num, column_width + 1)
            
            logger.info(f"Exported elasticity results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export elasticity results: {str(e)}")
            return None 