"""
Visualization utilities for NER evaluation results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List
from config import config

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Handles visualization generation for evaluation results."""
    
    def __init__(self, figsize: tuple = None, dpi: int = None):
        """
        Initialize plot generator.
        
        Args:
            figsize: Figure size tuple (width, height)
            dpi: Plot resolution
        """
        self.figsize = figsize or config.PLOT_FIGSIZE
        self.dpi = dpi or config.PLOT_DPI
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comparison_plot(self, results: Dict[str, Dict[str, float]], 
                             metric: str, save_path: str = None) -> None:
        """
        Create a bar plot comparing models based on the specified metric.
        
        Args:
            results: Dictionary mapping model names to their metrics
            metric: Metric to plot ('precision', 'recall', 'f1')
            save_path: Path to save plot (uses config default if None)
        """
        if metric not in config.METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {config.METRICS}")
        
        # Extract data
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        if not models:
            logger.warning("No data to plot")
            return
        
        # Create plot
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create bar plot
        bars = plt.bar(range(len(models)), values, alpha=0.8)
        
        # Customize plot
        plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        
        # Set x-axis labels
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = getattr(config, f"{metric.upper()}_PLOT")
        
        try:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving plot to {save_path}: {e}")
        
        plt.close()
    
    def create_all_comparison_plots(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Create comparison plots for all metrics.
        
        Args:
            results: Dictionary mapping model names to their metrics
        """
        logger.info("Creating comparison plots...")
        
        for metric in config.METRICS:
            try:
                self.create_comparison_plot(results, metric)
            except Exception as e:
                logger.error(f"Error creating {metric} plot: {e}")
    
    def create_summary_table_plot(self, results: Dict[str, Dict[str, float]], 
                                save_path: str = None) -> None:
        """
        Create a summary table visualization.
        
        Args:
            results: Dictionary mapping model names to their metrics
            save_path: Path to save plot
        """
        import pandas as pd
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results).T
        df_results = df_results.round(3)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(df_results) * 0.5)), dpi=self.dpi)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df_results.values,
                        rowLabels=df_results.index,
                        colLabels=[col.capitalize() for col in df_results.columns],
                        cellLoc='center',
                        loc='center')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(df_results.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color row labels
        for i in range(1, len(df_results) + 1):
            table[(i, -1)].set_facecolor('#E8F5E8')
            table[(i, -1)].set_text_props(weight='bold')
        
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Save plot
        if save_path is None:
            save_path = config.OUTPUT_DIR / "summary_table.png"
        
        try:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Summary table saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving summary table to {save_path}: {e}")
        
        plt.close()
    
    def create_heatmap(self, results: Dict[str, Dict[str, float]], 
                      save_path: str = None) -> None:
        """
        Create a heatmap of model performance across metrics.
        
        Args:
            results: Dictionary mapping model names to their metrics
            save_path: Path to save plot
        """
        import pandas as pd
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results).T
        
        # Create heatmap
        plt.figure(figsize=(10, max(6, len(df_results) * 0.4)), dpi=self.dpi)
        
        sns.heatmap(df_results, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
                   fmt='.3f', cbar_kws={'label': 'Score'})
        
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = config.OUTPUT_DIR / "performance_heatmap.png"
        
        try:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving heatmap to {save_path}: {e}")
        
        plt.close()
    
    def print_summary_table(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Print a formatted summary table to console.
        
        Args:
            results: Dictionary mapping model names to their metrics
        """
        if not results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-" * 70)
        
        for model, metrics in results.items():
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            print(f"{model:<25} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
        
        print("="*70)