"""Data visualization for financial data analysis.

This module provides tools for visualizing financial data distributions, 
time series, and statistical properties to validate data processing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Logger
from utils.logger import Logger
logger = Logger.get_logger()

# Import technical indicator groups from config
from config.data import (
    PRICE_COLUMNS, 
    VOLUME_COLUMNS, 
    MARKET_INDICATORS, 
    TECHNICAL_INDICATOR_GROUPS, 
    EXCLUDE_COLUMNS
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")


class DataVisualization:
    """Visualize financial data for analysis and validation.
    
    This class provides methods to visualize distributions, time series,
    correlations, and other properties of financial data to ensure
    data quality and proper processing.
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        """Initialize the visualization tool.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Default figure size
        self.figsize = (12, 8)
        logger.info(f"Visualization tool initialized with output directory: {save_dir}")
        
    def plot_distributions(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        by_ticker: bool = False,
        bins: int = 50,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot histograms of the selected columns to show data distributions.
        
        Args:
            data: DataFrame containing the data
            columns: List of columns to visualize. If None, uses all numeric columns
            by_ticker: Whether to group by ticker
            bins: Number of bins for histograms
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Select columns to plot
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
        
        if not columns:
            logger.warning("No numeric columns to plot distributions for.")
            return
            
        # Handle plotting by ticker
        if by_ticker and 'ticker' in data.columns:
            tickers = data['ticker'].unique()
            
            for ticker in tickers:
                ticker_data = data[data['ticker'] == ticker]
                
                # Create plot filename
                if save_filename:
                    plot_filename = f"{save_filename}_{ticker}.png"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"distribution_{ticker}_{timestamp}.png"
                
                # Create a figure with multiple subplots
                n_cols = min(3, len(columns))
                n_rows = (len(columns) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
                fig.suptitle(f'Data Distributions for {ticker}', fontsize=16)
                
                # Flatten axes array for easier iteration
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                # Plot each column
                for i, col in enumerate(columns):
                    if i < len(axes):
                        if col in ticker_data.columns:
                            sns.histplot(ticker_data[col].dropna(), bins=bins, kde=True, ax=axes[i])
                            axes[i].set_title(f'{col}')
                            axes[i].set_xlabel('')
                
                # Hide any unused subplots
                for j in range(len(columns), len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(os.path.join(self.save_dir, plot_filename))
                plt.close()
                
        else:
            # Create plot filename
            if save_filename:
                plot_filename = f"{save_filename}.png"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"distribution_{timestamp}.png"
            
            # Create a figure with multiple subplots
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
            fig.suptitle('Data Distributions', fontsize=16)
            
            # Flatten axes array for easier iteration
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Plot each column
            for i, col in enumerate(columns):
                if i < len(axes) and col in data.columns:
                    sns.histplot(data[col].dropna(), bins=bins, kde=True, ax=axes[i])
                    axes[i].set_title(f'{col}')
                    axes[i].set_xlabel('')
            
            # Hide any unused subplots
            for j in range(len(columns), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            
        logger.info(f"Distribution plots saved to {self.save_dir}")
        
    def plot_time_series(
        self,
        data: pd.DataFrame,
        columns: List[str],
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot time series of selected columns.
        
        Args:
            data: DataFrame containing the data
            columns: Columns to visualize
            tickers: List of tickers to include. If None, includes all
            start_date: Start date for time series (format: 'YYYY-MM-DD')
            end_date: End date for time series (format: 'YYYY-MM-DD')
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        if 'date' not in data.columns:
            logger.error("DataFrame must have a 'date' column for time series plots.")
            return
            
        # Ensure date column is datetime
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        # Filter by date range if provided
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
            
        # Filter by tickers if provided
        if tickers and 'ticker' in data.columns:
            data = data[data['ticker'].isin(tickers)]
            
        # Create plot filename and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            plot_filename = f"time_series_{timestamp}.png"
            
        # If we have multiple tickers, plot each ticker separately
        if 'ticker' in data.columns and data['ticker'].nunique() > 1:
            for ticker in data['ticker'].unique():
                ticker_data = data[data['ticker'] == ticker]
                
                # Set up the figure
                fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4*len(columns)))
                fig.suptitle(f'Time Series for {ticker}', fontsize=16)
                
                if len(columns) == 1:
                    axes = [axes]
                    
                # Plot each column
                for i, col in enumerate(columns):
                    if col in ticker_data.columns:
                        ticker_data.plot(x='date', y=col, ax=axes[i], legend=False)
                        axes[i].set_title(f'{col}')
                        axes[i].set_xlabel('')
                        
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save with ticker in filename
                if save_filename:
                    ticker_filename = f"{save_filename}_{ticker}.png"
                else:
                    ticker_filename = f"time_series_{ticker}_{timestamp}.png"
                    
                plt.savefig(os.path.join(self.save_dir, ticker_filename))
                plt.close()
        else:
            # Set up the figure
            fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4*len(columns)))
            
            if len(columns) == 1:
                axes = [axes]
                
            # Plot each column
            for i, col in enumerate(columns):
                if col in data.columns:
                    data.plot(x='date', y=col, ax=axes[i], legend=False)
                    axes[i].set_title(f'{col}')
                    axes[i].set_xlabel('')
                    
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            
        logger.info(f"Time series plots saved to {self.save_dir}")
        
    def compare_raw_vs_normalized(
        self,
        raw_data: pd.DataFrame,
        normalized_data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        save_filename: Optional[str] = None
    ) -> None:
        """Compare raw data with normalized data.
        
        Args:
            raw_data: DataFrame with raw data
            normalized_data: DataFrame with normalized data
            columns: Columns to compare. If None, uses all shared numeric columns
            tickers: List of tickers to include. If None, includes all
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Find common columns if not specified
        if columns is None:
            raw_numeric = raw_data.select_dtypes(include=np.number).columns
            norm_numeric = normalized_data.select_dtypes(include=np.number).columns
            columns = [col for col in raw_numeric if col in norm_numeric]
            
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
                    
        if not columns:
            logger.warning("No common numeric columns found between raw and normalized data.")
            return
            
        # Filter by tickers if provided
        if tickers and 'ticker' in raw_data.columns and 'ticker' in normalized_data.columns:
            raw_data = raw_data[raw_data['ticker'].isin(tickers)]
            normalized_data = normalized_data[normalized_data['ticker'].isin(tickers)]
            
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"raw_vs_normalized_{timestamp}.png"
            
        # Plot comparisons
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
        fig.suptitle('Raw vs. Normalized Data Comparison', fontsize=16)
        
        # Flatten axes for easier iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes) and col in raw_data.columns and col in normalized_data.columns:
                ax = axes[i]
                
                # Plot histograms side by side
                sns.histplot(raw_data[col].dropna(), color='blue', alpha=0.5, 
                            label='Raw', bins=50, kde=True, ax=ax)
                sns.histplot(normalized_data[col].dropna(), color='red', alpha=0.5, 
                            label='Normalized', bins=50, kde=True, ax=ax)
                
                ax.set_title(f'{col}')
                ax.legend()
                
        # Hide any unused subplots
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.save_dir, plot_filename))
        plt.close()
        
        logger.info(f"Raw vs. normalized comparison saved to {self.save_dir}")
        
    def plot_correlations(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson',
        ticker: Optional[str] = None,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot correlation matrix for selected columns.
        
        Args:
            data: DataFrame with the data
            columns: Columns to include in correlation analysis
            method: Correlation method to use ('pearson', 'kendall', 'spearman')
            ticker: If provided, only use data for this ticker
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Filter by ticker if provided
        if ticker and 'ticker' in data.columns:
            data = data[data['ticker'] == ticker]
            
        # Select columns to correlate
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
                    
        if not columns:
            logger.warning("No numeric columns found for correlation analysis.")
            return
            
        # Subset data and compute correlations
        subset_data = data[columns].copy()
        corr_matrix = subset_data.corr(method=method)
        
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if ticker:
                plot_filename = f"correlation_{ticker}_{timestamp}.png"
            else:
                plot_filename = f"correlation_{timestamp}.png"
                
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=.5,
            annot=False,
            fmt='.2f'
        )
        
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, plot_filename))
        plt.close()
        
        logger.info(f"Correlation matrix saved to {self.save_dir}")
        
    def plot_statistics(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        by_ticker: bool = False,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot basic statistics for the data.
        
        Args:
            data: DataFrame with the data
            columns: Columns to analyze
            by_ticker: Whether to group by ticker
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Select columns for statistics
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
                    
        if not columns:
            logger.warning("No numeric columns for statistical analysis.")
            return
            
        # Create timestamp for consistent filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            plot_filename = f"statistics_{timestamp}.png"
            
        # Compute and display statistics
        if by_ticker and 'ticker' in data.columns:
            tickers = data['ticker'].unique()
            
            for ticker in tickers:
                ticker_data = data[data['ticker'] == ticker]
                stats_df = ticker_data[columns].describe().T
                
                # Add more statistics
                stats_df['skew'] = ticker_data[columns].skew()
                stats_df['kurtosis'] = ticker_data[columns].kurtosis()
                
                # Plot as a table
                fig, ax = plt.subplots(figsize=(14, len(columns)*0.4 + 2))
                ax.axis('off')
                
                plt.title(f'Statistics for {ticker}', fontsize=16)
                
                # Format the table data - convert to string with rounding for numeric values
                table_data = stats_df.reset_index()
                
                # Convert to string format with proper rounding
                formatted_data = []
                for row in table_data.values:
                    formatted_row = []
                    for val in row:
                        if isinstance(val, (int, float)):
                            formatted_row.append(f"{val:.4f}")
                        else:
                            formatted_row.append(str(val))
                    formatted_data.append(formatted_row)
                
                # Add table
                table = ax.table(
                    cellText=formatted_data,
                    colLabels=['column'] + list(stats_df.columns),
                    cellLoc='center',
                    loc='center'
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                plt.tight_layout()
                
                # Create ticker-specific filename
                if save_filename:
                    ticker_filename = f"{save_filename}_{ticker}.png"
                else:
                    ticker_filename = f"statistics_{ticker}_{timestamp}.png"
                
                plt.savefig(os.path.join(self.save_dir, ticker_filename))
                plt.close()
                
        else:
            stats_df = data[columns].describe().T
            
            # Add more statistics
            stats_df['skew'] = data[columns].skew()
            stats_df['kurtosis'] = data[columns].kurtosis()
            
            # Plot as a table
            fig, ax = plt.subplots(figsize=(14, len(columns)*0.4 + 2))
            ax.axis('off')
            
            plt.title('Data Statistics', fontsize=16)
            
            # Format the table data - convert to string with rounding for numeric values
            table_data = stats_df.reset_index()
            
            # Convert to string format with proper rounding
            formatted_data = []
            for row in table_data.values:
                formatted_row = []
                for val in row:
                    if isinstance(val, (int, float)):
                        formatted_row.append(f"{val:.4f}")
                    else:
                        formatted_row.append(str(val))
                formatted_data.append(formatted_row)
            
            # Add table
            table = ax.table(
                cellText=formatted_data,
                colLabels=['column'] + list(stats_df.columns),
                cellLoc='center',
                loc='center'
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            
        logger.info(f"Statistical summary saved to {self.save_dir}")
    
    def plot_anomaly_detection(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        z_score_threshold: float = 3.0,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot z-score ranges to identify outliers and anomalies across all stocks.
        
        Args:
            data: DataFrame with the data
            columns: Columns to analyze for anomalies. If None, uses all numeric columns
            z_score_threshold: Threshold for considering a value as an anomaly
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Select columns to analyze
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
                    
        if not columns:
            logger.warning("No numeric columns found for anomaly detection.")
            return
            
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"anomaly_detection_{timestamp}.png"
            
        # Calculate z-scores for each column
        z_scores = pd.DataFrame()
        for col in columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:  # Avoid division by zero
                    z_scores[col] = (data[col] - mean) / std
        
        # Count anomalies by column
        anomaly_counts = (z_scores.abs() > z_score_threshold).sum()
        anomaly_percentages = (anomaly_counts / len(data)) * 100
        
        # Calculate z-score statistics
        z_min = z_scores.min()
        z_max = z_scores.max()
        z_mean = z_scores.mean()
        z_std = z_scores.std()
        
        # Create a figure for z-score ranges
        plt.figure(figsize=(14, 8))
        
        # Plot z-score ranges
        x = np.arange(len(columns))
        plt.errorbar(x, z_mean, yerr=z_std, fmt='o', capsize=5, label='Mean ± Std')
        plt.scatter(x, z_min, marker='v', color='blue', label='Min')
        plt.scatter(x, z_max, marker='^', color='red', label='Max')
        
        # Add threshold lines
        plt.axhline(y=z_score_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold (+{z_score_threshold})')
        plt.axhline(y=-z_score_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold (-{z_score_threshold})')
        
        # Add percentage labels - using iloc to fix FutureWarning
        for i, (col, pct) in enumerate(anomaly_percentages.items()):
            plt.text(i, z_max.iloc[i], f"{pct:.1f}%", ha='center', va='bottom')
        
        plt.xticks(x, columns, rotation=90)
        plt.ylabel('Z-Score')
        plt.title('Anomaly Detection - Z-Score Ranges Across All Stocks')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.save_dir, plot_filename))
        plt.close()
        
        logger.info(f"Anomaly detection plot saved to {self.save_dir}")
        
    def plot_aggregate_distributions(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        bins: int = 50,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot aggregate distributions of data across all stocks.
        
        Args:
            data: DataFrame containing the data
            columns: List of columns to visualize. If None, uses all numeric columns
            bins: Number of bins for histograms
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Select columns to plot
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
        
        if not columns:
            logger.warning("No numeric columns to plot aggregate distributions for.")
            return
            
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"aggregate_distribution_{timestamp}.png"
        
        # Create a figure with multiple subplots
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
        fig.suptitle('Aggregate Data Distributions', fontsize=16)
        
        # Flatten axes array for easier iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(columns):
            if i < len(axes) and col in data.columns:
                sns.histplot(data[col].dropna(), bins=bins, kde=True, ax=axes[i])
                
                # Calculate and display statistics
                mean_val = data[col].mean()
                median_val = data[col].median()
                std_val = data[col].std()
                
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
                
                axes[i].set_title(f'{col} (σ={std_val:.2f})')
                axes[i].set_xlabel('')
                axes[i].legend(fontsize='small')
        
        # Hide any unused subplots
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.save_dir, plot_filename))
        plt.close()
            
        logger.info(f"Aggregate distribution plots saved to {self.save_dir}")

    def plot_time_series_bands(
        self,
        data: pd.DataFrame,
        columns: List[str],
        max_features_per_plot: int = 4,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot time series with confidence bands showing mean, median, and quantiles across stocks.
        
        Args:
            data: DataFrame containing the data
            columns: Columns to visualize
            max_features_per_plot: Maximum number of features to show in a single plot
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        if 'date' not in data.columns:
            logger.error("DataFrame must have a 'date' column for time series bands plots.")
            return
            
        # Ensure date column is datetime
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        # Create plot filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = save_filename if save_filename else f"time_series_bands_{timestamp}"
        
        # Plot in groups of max_features_per_plot
        for i in range(0, len(columns), max_features_per_plot):
            subset_columns = columns[i:i+max_features_per_plot]
            subset_suffix = f"_part{i//max_features_per_plot+1}" if len(columns) > max_features_per_plot else ""
            
            # Set up the figure
            fig, axes = plt.subplots(len(subset_columns), 1, figsize=(12, 5*len(subset_columns)))
            
            if len(subset_columns) == 1:
                axes = [axes]
                
            # Calculate aggregates by date
            date_grouped = data.groupby('date')
            
            for j, col in enumerate(subset_columns):
                if col in data.columns:
                    # Calculate statistics for each date
                    stats = date_grouped[col].agg(['mean', 'median', 'std', 'min', 'max', 
                                                lambda x: np.percentile(x, 25), 
                                                lambda x: np.percentile(x, 75)])
                    stats.columns = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']
                    
                    # Plot the median as the main line
                    axes[j].plot(stats.index, stats['median'], color='blue', label='Median')
                    
                    # Plot the mean as a separate line
                    axes[j].plot(stats.index, stats['mean'], color='red', linestyle='--', label='Mean')
                    
                    # Plot the quantile bands
                    axes[j].fill_between(stats.index, stats['q25'], stats['q75'], 
                                        alpha=0.3, color='blue', label='25-75 Percentile')
                    
                    # Plot min/max as light bands
                    axes[j].fill_between(stats.index, stats['min'], stats['max'], 
                                        alpha=0.1, color='blue', label='Min-Max Range')
                    
                    axes[j].set_title(f'{col}')
                    axes[j].set_ylabel('Value')
                    
                    # Only show legend for the first subplot to save space
                    if j == 0:
                        axes[j].legend(loc='best')
                        
                    # Format x-axis
                    axes[j].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.autofmt_xdate()  # Rotate date labels
            
            # Save the figure
            plot_filename = f"{base_filename}{subset_suffix}.png"
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            
        logger.info(f"Time series bands plots saved to {self.save_dir}")
        
    def plot_feature_evolution(
        self,
        data: pd.DataFrame,
        columns: List[str],
        max_features_per_plot: int = 4,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot how feature distributions evolve over time with box plots.
        
        Args:
            data: DataFrame containing the data
            columns: Columns to visualize
            max_features_per_plot: Maximum number of features to show in a single plot
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        if 'date' not in data.columns:
            logger.error("DataFrame must have a 'date' column for feature evolution plots.")
            return
            
        # Ensure date column is datetime
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        # Create a month period column for grouping
        data['month'] = data['date'].dt.to_period('M')
        
        # Create plot filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = save_filename if save_filename else f"feature_evolution_{timestamp}"
        
        # Plot in groups of max_features_per_plot
        for i in range(0, len(columns), max_features_per_plot):
            subset_columns = columns[i:i+max_features_per_plot]
            subset_suffix = f"_part{i//max_features_per_plot+1}" if len(columns) > max_features_per_plot else ""
            
            # Set up the figure
            fig, axes = plt.subplots(len(subset_columns), 1, figsize=(14, 5*len(subset_columns)))
            
            if len(subset_columns) == 1:
                axes = [axes]
                
            for j, col in enumerate(subset_columns):
                if col in data.columns:
                    # Create monthly box plots
                    sns.boxplot(x='month', y=col, data=data, ax=axes[j])
                    
                    # Get the current tick positions and labels
                    ticks = axes[j].get_xticks()
                    labels = [item.get_text() for item in axes[j].get_xticklabels()]
                    
                    # Set ticks and rotate labels
                    axes[j].set_xticks(ticks)
                    axes[j].set_xticklabels(labels, rotation=90)
                    
                    axes[j].set_title(f'Monthly Distribution of {col}')
                    axes[j].set_xlabel('Month')
                    axes[j].set_ylabel(col)
                    axes[j].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            plot_filename = f"{base_filename}{subset_suffix}.png"
            plt.savefig(os.path.join(self.save_dir, plot_filename))
            plt.close()
            
        logger.info(f"Feature evolution plots saved to {self.save_dir}")
        
    def plot_aggregate_correlation(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson',
        save_filename: Optional[str] = None
    ) -> None:
        """Plot aggregate correlation matrix across all stocks.
        
        Args:
            data: DataFrame with the data
            columns: Columns to include in correlation analysis. If None, uses all numeric columns
            method: Correlation method to use ('pearson', 'kendall', 'spearman')
            save_filename: Filename to save the plot. If None, uses a timestamp
        """
        # Select columns to correlate
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
            
            # Remove non-feature columns
            for col in ['day', 'timestamp']:
                if col in columns:
                    columns.remove(col)
                    
        if not columns:
            logger.warning("No numeric columns found for correlation analysis.")
            return
            
        # Create plot filename
        if save_filename:
            plot_filename = f"{save_filename}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"aggregate_correlation_{timestamp}.png"
            
        # Subset data and compute correlations
        subset_data = data[columns].copy()
        corr_matrix = subset_data.corr(method=method)
        
        # Plot correlation matrix
        plt.figure(figsize=(max(12, len(columns)//2), max(10, len(columns)//2)))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Plot the correlation matrix
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=.5,
            annot=True if len(columns) < 20 else False,  # Only show annotations if not too many columns
            fmt='.2f' if len(columns) < 20 else None
        )
        
        plt.title(f'Aggregate Correlation Matrix ({method.capitalize()})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, plot_filename))
        plt.close()
        
        # If there are too many columns, also create a filtered version with only strong correlations
        if len(columns) > 15:
            plt.figure(figsize=(14, 12))
            
            # Create a copy and set weak correlations to zero
            strong_corr = corr_matrix.copy()
            strong_corr[(strong_corr < 0.5) & (strong_corr > -0.5)] = 0
            
            # Remove rows/columns that have no strong correlations
            has_strong = ~(strong_corr.abs().sum() == 1)  # Sum of 1 means only self-correlation
            strong_corr = strong_corr.loc[has_strong, has_strong]
            
            # Only proceed if we have strong correlations
            if has_strong.sum() > 1:
                sns.heatmap(
                    strong_corr,
                    cmap=cmap,
                    vmax=1.0,
                    vmin=-1.0,
                    center=0,
                    square=True,
                    linewidths=.5,
                    annot=True,
                    fmt='.2f'
                )
                
                plt.title(f'Strong Correlations Only ({method.capitalize()})', fontsize=16)
                plt.tight_layout()
                
                # Save with modified filename
                strong_filename = f"{save_filename}_strong.png" if save_filename else f"aggregate_correlation_strong_{timestamp}.png"
                plt.savefig(os.path.join(self.save_dir, strong_filename))
                plt.close()
                
        logger.info(f"Aggregate correlation matrix saved to {self.save_dir}")
        
    def compare_train_val_datasets(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        key_columns: List[str] = ["close", "volume"],
        additional_columns: Optional[List[str]] = None,
        save_filename_prefix: str = "train_val_comparison"
    ) -> None:
        """Compare training and validation datasets to detect distribution shifts.
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            key_columns: Primary columns to compare in detail
            additional_columns: Additional columns to include in statistical comparison
            save_filename_prefix: Prefix for saved files
        """
        if not key_columns:
            logger.warning("No key columns provided for train/val comparison.")
            return
            
        # Ensure we have common columns between datasets
        common_columns = [col for col in key_columns if col in train_data.columns and col in val_data.columns]
        
        if not common_columns:
            logger.warning("No common key columns found between training and validation data.")
            return
            
        # Add additional columns if specified
        if additional_columns:
            additional_cols = [col for col in additional_columns 
                              if col in train_data.columns and col in val_data.columns 
                              and col not in key_columns]
        else:
            additional_cols = []
            
        all_columns = common_columns + additional_cols
        
        # Timestamp for consistent filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Compare distributions for key columns
        for col in common_columns:
            plt.figure(figsize=(12, 6))
            
            # Create a histogram with KDE for each dataset
            sns.histplot(train_data[col].dropna(), color='blue', alpha=0.5, 
                         label='Training', kde=True, bins=50)
            sns.histplot(val_data[col].dropna(), color='red', alpha=0.5, 
                         label='Validation', kde=True, bins=50)
            
            # Add means as vertical lines
            train_mean = train_data[col].mean()
            val_mean = val_data[col].mean()
            plt.axvline(train_mean, color='blue', linestyle='--', 
                       label=f'Train Mean: {train_mean:.4f}')
            plt.axvline(val_mean, color='red', linestyle='--', 
                      label=f'Val Mean: {val_mean:.4f}')
            
            plt.title(f'Distribution Comparison for {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{save_filename_prefix}_{col}_dist.png"))
            plt.close()
            
        # 2. Compare time series characteristics for key columns
        if 'date' in train_data.columns and 'date' in val_data.columns:
            # Ensure date columns are datetime
            train_data_copy = train_data.copy()
            val_data_copy = val_data.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(train_data_copy['date']):
                train_data_copy['date'] = pd.to_datetime(train_data_copy['date'])
            if not pd.api.types.is_datetime64_any_dtype(val_data_copy['date']):
                val_data_copy['date'] = pd.to_datetime(val_data_copy['date'])
                
            # Group by date to get daily aggregates
            train_date_grouped = train_data_copy.groupby('date')
            val_date_grouped = val_data_copy.groupby('date')
            
            for col in common_columns:
                if col in train_data_copy.columns and col in val_data_copy.columns:
                    # Calculate daily means
                    train_means = train_date_grouped[col].mean()
                    val_means = val_date_grouped[col].mean()
                    
                    plt.figure(figsize=(14, 6))
                    
                    # Plot time series
                    plt.plot(train_means.index, train_means, color='blue', label='Training')
                    plt.plot(val_means.index, val_means, color='red', label='Validation')
                    
                    plt.title(f'Time Series Comparison for {col} (Daily Means)')
                    plt.xlabel('Date')
                    plt.ylabel(col)
                    plt.legend()
                    
                    plt.tight_layout()
                    fig = plt.gcf()
                    fig.autofmt_xdate()  # Rotate date labels
                    
                    # Save the plot
                    plt.savefig(os.path.join(self.save_dir, f"{save_filename_prefix}_{col}_ts.png"))
                    plt.close()
                    
        # 3. Statistical comparison table for all columns
        stats_data = []
        
        for col in all_columns:
            if col in train_data.columns and col in val_data.columns:
                train_stats = train_data[col].describe()
                val_stats = val_data[col].describe()
                
                # Calculate additional statistics
                train_skew = train_data[col].skew()
                val_skew = val_data[col].skew()
                train_kurt = train_data[col].kurtosis()
                val_kurt = val_data[col].kurtosis()
                
                # Calculate percent differences
                pct_diff = {}
                for stat in ['mean', 'std', '50%', 'min', 'max']:
                    if stat in train_stats and stat in val_stats and train_stats[stat] != 0:
                        pct_diff[stat] = ((val_stats[stat] - train_stats[stat]) / abs(train_stats[stat])) * 100
                    else:
                        pct_diff[stat] = float('nan')
                        
                stats_data.append({
                    'Column': col,
                    'Train Mean': train_stats['mean'],
                    'Val Mean': val_stats['mean'],
                    'Mean % Diff': pct_diff['mean'],
                    'Train Median': train_stats['50%'],
                    'Val Median': val_stats['50%'],
                    'Median % Diff': pct_diff['50%'],
                    'Train Std': train_stats['std'],
                    'Val Std': val_stats['std'],
                    'Std % Diff': pct_diff['std'],
                    'Train Min': train_stats['min'],
                    'Val Min': val_stats['min'],
                    'Train Max': train_stats['max'],
                    'Val Max': val_stats['max'],
                    'Train Skew': train_skew,
                    'Val Skew': val_skew,
                    'Train Kurt': train_kurt,
                    'Val Kurt': val_kurt
                })
                
        # Create a table figure with the stats data
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Format the table with highlighting for large differences
            fig, ax = plt.subplots(figsize=(20, len(stats_data)*0.8 + 2))
            ax.axis('off')
            
            # Highlight cells with large percent differences
            cell_colors = []
            for _, row in stats_df.iterrows():
                row_colors = ['white'] * len(row)  # Default color
                
                # Highlight mean % diff
                mean_diff_idx = stats_df.columns.get_loc('Mean % Diff')
                if abs(row['Mean % Diff']) > 10:
                    row_colors[mean_diff_idx] = 'salmon' if row['Mean % Diff'] > 0 else 'lightblue'
                elif abs(row['Mean % Diff']) > 5:
                    row_colors[mean_diff_idx] = 'lightsalmon' if row['Mean % Diff'] > 0 else 'lightcyan'
                    
                # Highlight median % diff
                median_diff_idx = stats_df.columns.get_loc('Median % Diff')
                if abs(row['Median % Diff']) > 10:
                    row_colors[median_diff_idx] = 'salmon' if row['Median % Diff'] > 0 else 'lightblue'
                elif abs(row['Median % Diff']) > 5:
                    row_colors[median_diff_idx] = 'lightsalmon' if row['Median % Diff'] > 0 else 'lightcyan'
                    
                # Highlight std % diff
                std_diff_idx = stats_df.columns.get_loc('Std % Diff')
                if abs(row['Std % Diff']) > 20:
                    row_colors[std_diff_idx] = 'salmon' if row['Std % Diff'] > 0 else 'lightblue'
                elif abs(row['Std % Diff']) > 10:
                    row_colors[std_diff_idx] = 'lightsalmon' if row['Std % Diff'] > 0 else 'lightcyan'
                    
                cell_colors.append(row_colors)
                
            # Format the data - round numeric values - using apply instead of deprecated applymap
            formatted_data = pd.DataFrame(index=stats_df.index, columns=stats_df.columns)
            for col in stats_df.columns:
                formatted_data[col] = stats_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
            
            # Create the table
            table = ax.table(
                cellText=formatted_data.values,
                colLabels=formatted_data.columns,
                cellLoc='center',
                loc='center',
                cellColours=cell_colors
            )
            
            # Adjust table formatting
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            plt.title('Statistical Comparison: Training vs Validation', fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(self.save_dir, f"{save_filename_prefix}_stats_table.png"), dpi=150)
            plt.close()
        
        logger.info(f"Training vs validation comparison saved to {self.save_dir}")
        
    def plot_grouped_time_series(
        self,
        data: pd.DataFrame,
        group_mappings: Optional[Dict[str, List[str]]] = None,
        column_patterns: Optional[Dict[str, List[str]]] = None,
        max_columns_per_group: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_filename: Optional[str] = None
    ) -> None:
        """Plot multiple time series columns grouped by category on a single figure per group.
        
        This method intelligently groups time series columns based on predefined categories or 
        pattern matching, then plots each group on a separate figure.
        
        Args:
            data: DataFrame containing the data
            group_mappings: Dictionary mapping group names to lists of column names
            column_patterns: Dictionary mapping group names to lists of regex patterns to match columns
            max_columns_per_group: Maximum number of columns to show in a single group
            start_date: Start date for time series (format: 'YYYY-MM-DD')
            end_date: End date for time series (format: 'YYYY-MM-DD')
            save_filename: Base filename to save the plots. If None, uses a timestamp
        """
        if 'date' not in data.columns:
            logger.error("DataFrame must have a 'date' column for time series plots.")
            return
            
        # Ensure date column is datetime
        data = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        # Filter by date range if provided
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
        
        # Create plot filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = save_filename if save_filename else f"grouped_time_series_{timestamp}"
        
        # Group related columns by category if not provided
        if group_mappings is None:
            # Default column groupings based on common financial indicators
            group_mappings = {}
            
            # Price-related columns
            price_cols = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'adj_close']]
            if price_cols:
                group_mappings['Price'] = price_cols
                
            # Volume-related columns
            volume_cols = [col for col in data.columns if col in ['volume']]
            if volume_cols:
                group_mappings['Volume'] = volume_cols
                
        # Apply pattern matching if provided
        if column_patterns:
            import re
            for group_name, patterns in column_patterns.items():
                matched_cols = []
                for pattern in patterns:
                    matched_cols.extend([col for col in data.columns 
                                        if re.search(pattern, col) and col not in matched_cols])
                if matched_cols:
                    if group_name in group_mappings:
                        group_mappings[group_name].extend(matched_cols)
                    else:
                        group_mappings[group_name] = matched_cols
                        
        # Auto-detect potential time series columns if no mappings are provided or found
        if not group_mappings:
            # Get numeric columns excluding non-feature columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            exclude_cols = ['day', 'timestamp']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Attempt to categorize by common naming patterns
            group_mappings = self._auto_categorize_columns(numeric_cols)
        
        # Plot each group
        for group_name, columns in group_mappings.items():
            # Skip if no columns in group
            if not columns:
                continue
                
            # Split into subgroups if too many columns
            for i in range(0, len(columns), max_columns_per_group):
                subgroup_columns = columns[i:i+max_columns_per_group]
                subgroup_suffix = f"_part{i//max_columns_per_group+1}" if len(columns) > max_columns_per_group else ""
                
                # Set up the figure
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Calculate a group-wide date aggregation (e.g., daily mean across all tickers)
                date_grouped = data.groupby('date')
                
                for col in subgroup_columns:
                    if col in data.columns:
                        # Calculate mean for each date
                        means = date_grouped[col].mean()
                        
                        # Plot the time series
                        ax.plot(means.index, means.values, label=col)
                
                ax.set_title(f'{group_name} Indicators{subgroup_suffix}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend(loc='best')
                ax.grid(True)
                
                # Format x-axis date ticks
                fig.autofmt_xdate()
                
                # Save the figure
                group_filename = f"{base_filename}_{group_name.lower().replace(' ', '_')}{subgroup_suffix}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, group_filename))
                plt.close()
        
        logger.info(f"Grouped time series plots saved to {self.save_dir}")
        
    def _auto_categorize_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Automatically categorize columns into groups based on naming patterns.
        
        Args:
            columns: List of column names to categorize
            
        Returns:
            Dictionary mapping category names to lists of column names
        """
        # Initialize categories using config groups plus "Other" for uncategorized columns
        categories = {
            'Price': [],
            'Volume': [],
            'Market': [],
            'Trend': [],
            'Momentum': [],
            'Volatility': [],
            'Volume Indicators': [],
            'Other': []
        }
        
        # Categorize each column
        for col in columns:
            col_lower = col.lower()
            
            # Check if column is in price columns
            if any(price_col in col_lower for price_col in PRICE_COLUMNS):
                categories['Price'].append(col)
                
            # Check if column is in volume columns
            elif any(vol_col in col_lower for vol_col in VOLUME_COLUMNS):
                categories['Volume'].append(col)
                
            # Check if column is in market indicators
            elif any(market_ind in col_lower for market_ind in MARKET_INDICATORS):
                categories['Market'].append(col)
                
            # Check if column is in trend indicators
            elif any(trend_ind in col_lower for trend_ind in TECHNICAL_INDICATOR_GROUPS['trend']):
                categories['Trend'].append(col)
                
            # Check if column is in momentum indicators
            elif any(momentum_ind in col_lower for momentum_ind in TECHNICAL_INDICATOR_GROUPS['momentum']):
                categories['Momentum'].append(col)
                
            # Check if column is in volatility indicators
            elif any(volatility_ind in col_lower for volatility_ind in TECHNICAL_INDICATOR_GROUPS['volatility']):
                categories['Volatility'].append(col)
                
            # Check if column is in volume indicators
            elif any(vol_ind in col_lower for vol_ind in TECHNICAL_INDICATOR_GROUPS['volume_indicators']):
                categories['Volume Indicators'].append(col)
                
            # Default category for unmatched columns
            else:
                categories['Other'].append(col)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
        
    def visualize_all(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame = None,
        normalized_data: pd.DataFrame = None,
        columns: Optional[List[str]] = None,
        time_series_columns: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        output_prefix: str = "",
        max_features_per_plot: int = 4,
        by_ticker: bool = False,  # Added parameter to control stock-specific plots
        auto_group_time_series: bool = True,  # New parameter to control automatic time series grouping
    ) -> None:
        """Run all visualizations on the provided datasets.
        
        Args:
            raw_data: DataFrame with raw data
            processed_data: DataFrame with processed data
            normalized_data: DataFrame with normalized data
            columns: Columns to visualize. If None, uses all numeric columns
            time_series_columns: Columns to use for time series visualization
            tickers: List of tickers to include. If None, includes all
            output_prefix: Prefix for output filenames
            max_features_per_plot: Maximum number of features to show in a single time series plot
            by_ticker: Whether to generate stock-specific plots (default: False)
            auto_group_time_series: Whether to automatically group time series plots (default: True)
        """
        # Select default time series columns if none provided
        if time_series_columns is None and 'close' in raw_data.columns:
            time_series_columns = ['close']
            
        # Generate prefix with timestamp if not provided
        if not output_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"analysis_{timestamp}"
            
        # Run visualizations on raw data
        logger.info("Generating raw data visualizations...")
        
        # Aggregate visualizations (not stock-specific)
        self.plot_aggregate_distributions(
            raw_data, columns,
            save_filename=f"{output_prefix}_raw_agg_dist"
        )
        
        if time_series_columns:
            self.plot_time_series_bands(
                raw_data, time_series_columns, max_features_per_plot,
                save_filename=f"{output_prefix}_raw_time_series_bands"
            )
            
            self.plot_feature_evolution(
                raw_data, time_series_columns, max_features_per_plot,
                save_filename=f"{output_prefix}_raw_feature_evolution"
            )
        
        # Add automatic time series grouping if enabled
        if auto_group_time_series:
            self.plot_grouped_time_series(
                raw_data,
                save_filename=f"{output_prefix}_raw_grouped_ts"
            )
            
        self.plot_aggregate_correlation(
            raw_data, columns,
            save_filename=f"{output_prefix}_raw_agg_corr"
        )
        
        self.plot_anomaly_detection(
            raw_data, columns,
            save_filename=f"{output_prefix}_raw_anomalies"
        )
        
        # Stock-specific visualizations (only if requested)
        if by_ticker:
            self.plot_distributions(
                raw_data, columns, by_ticker=True, 
                save_filename=f"{output_prefix}_raw_dist"
            )
            
            if time_series_columns:
                self.plot_time_series(
                    raw_data, time_series_columns, tickers, 
                    save_filename=f"{output_prefix}_raw_time_series"
                )
                
            if tickers and len(tickers) > 0:
                self.plot_correlations(
                    raw_data, columns, ticker=tickers[0],
                    save_filename=f"{output_prefix}_raw_corr"
                )
            
            self.plot_statistics(
                raw_data, columns, by_ticker=True,
                save_filename=f"{output_prefix}_raw_stats"
            )
            
        # Run visualizations on processed data if provided
        if processed_data is not None:
            logger.info("Generating processed data visualizations...")
            
            # Aggregate visualizations
            self.plot_aggregate_distributions(
                processed_data, columns,
                save_filename=f"{output_prefix}_processed_agg_dist"
            )
            
            if time_series_columns:
                self.plot_time_series_bands(
                    processed_data, time_series_columns, max_features_per_plot,
                    save_filename=f"{output_prefix}_processed_time_series_bands"
                )
                
                self.plot_feature_evolution(
                    processed_data, time_series_columns, max_features_per_plot,
                    save_filename=f"{output_prefix}_processed_feature_evolution"
                )
            
            # Add automatic time series grouping if enabled
            if auto_group_time_series:
                self.plot_grouped_time_series(
                    processed_data,
                    save_filename=f"{output_prefix}_processed_grouped_ts"
                )
                
            self.plot_aggregate_correlation(
                processed_data, columns,
                save_filename=f"{output_prefix}_processed_agg_corr"
            )
            
            self.plot_anomaly_detection(
                processed_data, columns,
                save_filename=f"{output_prefix}_processed_anomalies"
            )
            
            # Stock-specific visualizations (only if requested)
            if by_ticker:
                self.plot_distributions(
                    processed_data, columns, by_ticker=True,
                    save_filename=f"{output_prefix}_processed_dist"
                )
                
                if time_series_columns:
                    self.plot_time_series(
                        processed_data, time_series_columns, tickers,
                        save_filename=f"{output_prefix}_processed_time_series"
                    )
                    
                if tickers and len(tickers) > 0:
                    self.plot_correlations(
                        processed_data, columns, ticker=tickers[0],
                        save_filename=f"{output_prefix}_processed_corr"
                    )
                
                self.plot_statistics(
                    processed_data, columns, by_ticker=True,
                    save_filename=f"{output_prefix}_processed_stats"
                )
                
        # Run visualizations on normalized data if provided
        if normalized_data is not None:
            logger.info("Generating normalized data visualizations...")
            
            # Aggregate visualizations
            self.plot_aggregate_distributions(
                normalized_data, columns,
                save_filename=f"{output_prefix}_normalized_agg_dist"
            )
            
            if time_series_columns:
                self.plot_time_series_bands(
                    normalized_data, time_series_columns, max_features_per_plot,
                    save_filename=f"{output_prefix}_normalized_time_series_bands"
                )
                
                self.plot_feature_evolution(
                    normalized_data, time_series_columns, max_features_per_plot,
                    save_filename=f"{output_prefix}_normalized_feature_evolution"
                )
            
            # Add automatic time series grouping if enabled
            if auto_group_time_series:
                self.plot_grouped_time_series(
                    normalized_data,
                    save_filename=f"{output_prefix}_normalized_grouped_ts"
                )
                
            self.plot_aggregate_correlation(
                normalized_data, columns,
                save_filename=f"{output_prefix}_normalized_agg_corr"
            )
            
            self.plot_anomaly_detection(
                normalized_data, columns,
                save_filename=f"{output_prefix}_normalized_anomalies"
            )
            
            # Stock-specific visualizations (only if requested)
            if by_ticker:
                self.plot_distributions(
                    normalized_data, columns, by_ticker=True,
                    save_filename=f"{output_prefix}_normalized_dist"
                )
                
                if time_series_columns:
                    self.plot_time_series(
                        normalized_data, time_series_columns, tickers,
                        save_filename=f"{output_prefix}_normalized_time_series"
                    )
                    
                if tickers and len(tickers) > 0:
                    self.plot_correlations(
                        normalized_data, columns, ticker=tickers[0],
                        save_filename=f"{output_prefix}_normalized_corr"
                    )
                
                self.plot_statistics(
                    normalized_data, columns, by_ticker=True,
                    save_filename=f"{output_prefix}_normalized_stats"
                )
                
            # Compare raw vs normalized
            if raw_data is not None:
                self.compare_raw_vs_normalized(
                    raw_data, normalized_data, columns, tickers,
                    save_filename=f"{output_prefix}_raw_vs_normalized"
                )
                
        logger.info(f"All visualizations completed and saved to {self.save_dir}") 