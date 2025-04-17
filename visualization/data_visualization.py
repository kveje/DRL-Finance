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

# Logger
from utils.logger import Logger
logger = Logger.get_logger()

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
    
    def visualize_all(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame = None,
        normalized_data: pd.DataFrame = None,
        columns: Optional[List[str]] = None,
        time_series_columns: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        output_prefix: str = ""
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
        self.plot_distributions(
            raw_data, columns, by_ticker=True, 
            save_filename=f"{output_prefix}_raw_dist"
        )
        
        if time_series_columns:
            self.plot_time_series(
                raw_data, time_series_columns, tickers, 
                save_filename=f"{output_prefix}_raw_time_series"
            )
            
        self.plot_correlations(
            raw_data, columns, ticker=tickers[0] if tickers else None,
            save_filename=f"{output_prefix}_raw_corr"
        )
        
        self.plot_statistics(
            raw_data, columns, by_ticker=True,
            save_filename=f"{output_prefix}_raw_stats"
        )
            
        # Run visualizations on processed data if provided
        if processed_data is not None:
            logger.info("Generating processed data visualizations...")
            self.plot_distributions(
                processed_data, columns, by_ticker=True,
                save_filename=f"{output_prefix}_processed_dist"
            )
            
            if time_series_columns:
                self.plot_time_series(
                    processed_data, time_series_columns, tickers,
                    save_filename=f"{output_prefix}_processed_time_series"
                )
                
            self.plot_correlations(
                processed_data, columns, ticker=tickers[0] if tickers else None,
                save_filename=f"{output_prefix}_processed_corr"
            )
            
            self.plot_statistics(
                processed_data, columns, by_ticker=True,
                save_filename=f"{output_prefix}_processed_stats"
            )
                
        # Run visualizations on normalized data if provided
        if normalized_data is not None:
            logger.info("Generating normalized data visualizations...")
            self.plot_distributions(
                normalized_data, columns, by_ticker=True,
                save_filename=f"{output_prefix}_normalized_dist"
            )
            
            if time_series_columns:
                self.plot_time_series(
                    normalized_data, time_series_columns, tickers,
                    save_filename=f"{output_prefix}_normalized_time_series"
                )
                
            self.plot_correlations(
                normalized_data, columns, ticker=tickers[0] if tickers else None,
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